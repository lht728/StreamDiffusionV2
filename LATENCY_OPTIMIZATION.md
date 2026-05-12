# StreamDiffusionV2 延迟优化记录

> 记录围绕 step=1 流式推理场景，为降低端到端延迟所做的代码与配置改动。
> 硬件环境：NVIDIA H20 × 2，模型 T2V-1.3B，启用 `--fast --use_taehv --use_tensorrt`。
> 文档时间：2026-05-12。

---

## 1. 背景与基线

启动方式（典型）：

```bash
CUDA_VISIBLE_DEVICES=0,1 python demo/main.py \
  --num_gpus 2 --gpu_ids 0,1 --step 1 --model_type T2V-1.3B \
  --fast --use_taehv --use_tensorrt \
  --target-latency 0.4 --enable-metrics
```

观察指标（来源：`logs/server.log` 中的 `Batch N/M avg_latency=X.XXs`）：

- **batch 间隔（相邻打点 wall-time）**：反映稳态吞吐 / 单步耗时
- **avg_latency**：服务端记录的端到端单 batch 延迟
- 取 Batch ≥ 100 的稳态段计算 mean / median / p90

测试中发现：装上 FlashAttention 后延迟反而比未装时更大，由此触发本轮排查与优化。

---

## 2. 各 Attention 后端横向对比

| 模式 | batch 间隔 中位 | p90 | avg_latency 均值 | 备注 |
|---|---|---|---|---|
| 无 FA（before-fa3，旧 log） | 236 ms | 325 ms | 0.90 s ⚠️ | log 拼接含起停跳变；中位/p90 可信 |
| FA3 + KV-cache SDPA 混合 | **437 ms** | **444 ms** | **14.1 s** ⚠️ | 真实劣化，且出现阻塞 |
| FA2 全开 | 226 ms | 273 ms | 0.40 s | |
| **SDPA（FA 全禁，当前默认）** | **226 ms** | **272 ms** | **0.41 s** | 与 FA2 持平，路径更简单 |

结论：

- **FA3 在本场景下显著拖慢**（中位 +93%，且 avg_latency 出现 10s+ 异常）。
- **FA2 与 SDPA 性能等价**，但 SDPA 路径更简单、依赖更少、可控性更好。
- 因此目标方案：**默认禁用 flash-attn，统一走 SDPA**，但保留 wheel 与一键切换能力以便后续 A/B。

---

## 3. 改动层次总览

```
┌────────────────────────────────────────────────────────────────────┐
│  层次          改动                              原理         杠杆 │
├────────────────────────────────────────────────────────────────────┤
│  Kernel 后端   FA → SDPA (运行时开关)            匹配 H20 短序列 大 │
│  路径一致性    KV-cache 也走 SDPA                消除异构同步     大 │
│  默认配置      start.sh / run_v2v.sh 默认禁 FA   避免漏配         中 │
│  可观测        --enable-metrics                  数据驱动调优     中 │
│  调度反馈      --target-latency                  运行时延迟控制   小 │
│  运维封装      demo/run.sh + systemd unit        统一启动管理     —  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. 改动详情与原理

### 4.1 Kernel 后端：FlashAttention → SDPA（最大杠杆）

**改动文件**

- `models/wan/wan_base/modules/attention.py`
- `models/wan/causal_model.py`

**改动内容**

引入运行时开关 `STREAMDIFF_DISABLE_FLASH`：

```python
# attention.py 顶部
import os
_DISABLE_FLASH = os.environ.get("STREAMDIFF_DISABLE_FLASH", "").lower() in ("1", "true", "yes")

try:
    if _DISABLE_FLASH:
        raise ModuleNotFoundError("flash-attn disabled via STREAMDIFF_DISABLE_FLASH")
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    if _DISABLE_FLASH:
        raise ModuleNotFoundError("flash-attn disabled via STREAMDIFF_DISABLE_FLASH")
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
```

`causal_model.py` 中针对 KV-cache 路径的 `from flash_attn import flash_attn_interface` 同样套上开关。

`attention()` 主路径与 KV-cache 路径都已有 `try import + fallback to scaled_dot_product_attention` 的逻辑，因此只要让 import 失败就自动走 SDPA fallback。

**为什么禁用 FA 反而更快**

- **FA 的优势在长序列 + 大 batch + 训练**：tile + online softmax 把 attention 显存从 O(N²) 压到 O(N)，主要收益是减少 HBM 读写。
- **本场景为 step=1 推理 + KV cache + 短序列**：每步 q 长度 = 1，FA 的省显存优势用不上，反而暴露 **kernel 启动开销 + dispatch overhead**。
- **H20 是 H100 的算力阉割版**（FP8 / Tensor Core 削减）：FA2/FA3 的 kernel 是按 H100 调优的，迁到 H20 上性能模型不再匹配；FA3 实测中位耗时翻倍。
- **SDPA** 走 cuDNN / cuBLAS 内核，对短序列 + 小 q 长度时几何更友好、调度更轻。

### 4.2 路径一致性：消除 FA / SDPA 混合阻塞

FA3 那次配置实测 `avg_latency` 高达 **14.1 s**，远超 kernel 单步差异能解释的范围。原因：

- `attention.py` 主路径走 FA3，但 `causal_model.py` 的 **KV-cache 接口在 FA3 下仍 fallback 到 SDPA**；
- 同一 forward 内两条路径交替调用，CUDA stream 上同步点增多 + 显存格式来回转换，critical path 被拉长；
- 叠加 H20 上 FA3 单 kernel 本身就慢于 SDPA，效应被放大。

→ 当 `STREAMDIFF_DISABLE_FLASH=1` 时，两条路径都走 SDPA，**异构同步开销被消除**。这是除单 kernel 性能外的另一个收益来源。

### 4.3 默认配置：让最优配置成为默认

**`demo/start.sh`**

```bash
# Disable flash-attn at runtime by default (FA2/FA3 are no faster than SDPA on H20
# for step=1 + KV-cache + short sequences; FA3 was observed to be slower).
# Set STREAMDIFF_DISABLE_FLASH=0 to re-enable flash-attn for A/B comparison.
STREAMDIFF_DISABLE_FLASH="${STREAMDIFF_DISABLE_FLASH:-1}"
export STREAMDIFF_DISABLE_FLASH
```

并把开关显式写进 `python main.py` 调用行，与 `CUDA_VISIBLE_DEVICES` 风格一致。

同时新增以下环境变量，方便复用同一脚本应对不同部署形态：

| 变量 | 作用 | 默认 |
|---|---|---|
| `STREAMDIFF_DISABLE_FLASH` | 是否禁 FA | `1` |
| `SKIP_FRONTEND_BUILD` | 跳过 npm install / build | `0` |
| `PYTHON_BIN` | 指定 python 解释器（默认自动找 `.venv/bin/python`） | 自动 |
| `TARGET_LATENCY` | 透传 `--target-latency` | 不传 |
| `ENABLE_METRICS` | 透传 `--enable-metrics` | `0` |
| `EXTRA_ARGS` | 追加任意原生 CLI | 空 |

**`run_v2v.sh`**（offline V2V 入口）

```bash
export STREAMDIFF_DISABLE_FLASH="${STREAMDIFF_DISABLE_FLASH:-1}"
```

放在 `set -eu` 之后、所有模式分支之前，让脚本启动的三种模式（single / multi / longvideo）都默认禁 FA。

### 4.4 可观测：metrics + target-latency

启动参数固化为：

```
--target-latency 0.4 --enable-metrics
```

- `--enable-metrics`：周期性输出结构化日志 `Batch N/M avg_latency=X.XXs ...`，是所有横向对比的数据基础。没有这些日志，"中位/p90/均值"无从计算。
- `--target-latency`：给 scheduler 一个目标值，运行时根据延迟反馈做动态调度（如调整 batch 累积窗口），避免延迟漂移。

### 4.5 运维封装

**`demo/run.sh`（新增，容器友好）**

封装 `start / stop / restart / status / logs`：

- pid 文件管理与端口残留清理
- 日志路径标准化（`logs/server.log`）与按时间归档
- 默认透传 `STREAMDIFF_DISABLE_FLASH`，确保即使裸 `nohup` 风格也不会漏配

**`/etc/systemd/system/streamdiff-demo.service`（新增，裸机部署用）**

`Environment=STREAMDIFF_DISABLE_FLASH=1` 写入 unit；当前容器中 systemd 不可用，文件留作日后裸机部署直接 `systemctl enable --now`。

---

## 5. 使用指南

### 5.1 默认（禁 FA、SDPA 模式）

```bash
demo/run.sh start
demo/run.sh status        # 会显示 STREAMDIFF_DISABLE_FLASH 当前值
demo/run.sh logs          # 跟随日志
demo/run.sh stop
```

### 5.2 临时切回 FA2 做 A/B

```bash
STREAMDIFF_DISABLE_FLASH=0 demo/run.sh restart
```

确认当前模式（推流后看日志）：

- 出现 `flash_attn is not installed; falling back to scaled_dot_product_attention` → SDPA 模式 ✓
- 未出现该警告 → FA 模式

### 5.3 offline V2V

```bash
./run_v2v.sh single                       # 默认禁 FA
STREAMDIFF_DISABLE_FLASH=0 ./run_v2v.sh single   # 临时启用 FA
```

### 5.4 整机部署（裸机 systemd）

```bash
systemctl daemon-reload
systemctl enable --now streamdiff-demo

# 临时启用 FA：
mkdir -p /etc/systemd/system/streamdiff-demo.service.d
echo -e '[Service]\nEnvironment=STREAMDIFF_DISABLE_FLASH=0' \
  > /etc/systemd/system/streamdiff-demo.service.d/use-fa.conf
systemctl daemon-reload && systemctl restart streamdiff-demo
```

---

## 6. 验证流程

每次配置变更后，按以下步骤复测，确保数据可比：

1. 停掉旧 server，归档旧日志：`mv logs/server.log logs/server.log.<label>.<HHMMSS>`
2. 用目标配置启动，等 `Application startup complete`（H20 双卡冷启约 60–90 s）
3. 推流 ≥ 30 s，让 Batch 计数过 100（避开 warmup 与 torch.compile 期）
4. 用如下脚本统计 Batch ≥ 100 的稳态段：

```python
import re, statistics
ts = []
with open('logs/server.log') as f:
    for line in f:
        m = re.search(r'(\d\d:\d\d:\d\d,\d+).*Batch (\d+)/.*avg_latency=([\d.]+)s', line)
        if m:
            h, mn, s = m.group(1).split(':')
            sec, ms = s.split(',')
            t = int(h)*3600 + int(mn)*60 + int(sec) + int(ms)/1000
            ts.append((int(m.group(2)), t, float(m.group(3))))

stable = [x for x in ts if x[0] >= 100]
intervals = [stable[i+1][1] - stable[i][1] for i in range(len(stable)-1)]
lats = [x[2] for x in stable]
print(f"间隔 中位={statistics.median(intervals)*1000:.1f}ms "
      f"p90={sorted(intervals)[int(len(intervals)*0.9)]*1000:.1f}ms")
print(f"avg_latency 中位={statistics.median(lats):.4f}s "
      f"p90={sorted(lats)[int(len(lats)*0.9)]:.4f}s")
```

---

## 7. 经验教训

1. **不要默认相信"装了优化库就更快"**：FlashAttention 在训练 / 长序列下提速明显，但短序列推理 + 阉割版硬件场景需要实测。
2. **避免路径异构**：同一 forward 内 FA / SDPA 混用会引入同步与格式转换开销，量级可能远超 kernel 本身的差异。
3. **A/B 必须基线对齐**：torch.compile 的 autotune 编译期、客户端断流再连等都会污染数据；统一以 Batch ≥ 100 的稳态段做比较。
4. **保留切换能力优于"一刀切"**：用环境变量开关而不是卸载依赖，方便回归与硬件迁移后重新评估。
5. **指标要先于优化**：没有 `--enable-metrics` 输出的结构化日志，所有"快了 / 慢了"的判断都只是体感。

---

## 8. 改动文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `models/wan/wan_base/modules/attention.py` | 修改 | 顶部加 `STREAMDIFF_DISABLE_FLASH` 开关，禁 FA 时 import 抛 `ModuleNotFoundError` |
| `models/wan/causal_model.py` | 修改 | KV-cache 的 FA import 同样套开关 |
| `demo/start.sh` | 修改 | 默认 `STREAMDIFF_DISABLE_FLASH=1`；新增 `SKIP_FRONTEND_BUILD` / `PYTHON_BIN` / `TARGET_LATENCY` / `ENABLE_METRICS` / `EXTRA_ARGS` |
| `run_v2v.sh` | 修改 | 顶部 `export STREAMDIFF_DISABLE_FLASH="${STREAMDIFF_DISABLE_FLASH:-1}"` |
| `demo/run.sh` | 新增 | 容器友好的进程包装：`start / stop / restart / status / logs` |
| `/etc/systemd/system/streamdiff-demo.service` | 新增 | systemd unit，`Environment=STREAMDIFF_DISABLE_FLASH=1`，留作裸机部署 |
| `LATENCY_OPTIMIZATION.md` | 新增 | 本文档 |
