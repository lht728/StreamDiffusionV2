# Demo 启动指南

> StreamDiffusionV2 推流 demo 的启动 / 停止 / 排障一页纸。
> 配套延迟优化背景见同目录 `LATENCY_OPTIMIZATION.md`。

---

## TL;DR

```bash
# 推荐方式（容器/裸机通用，nohup 后台 + pidfile + 日志轮转）
/data/StreamDiffusionV2/demo/run.sh start
/data/StreamDiffusionV2/demo/run.sh status
/data/StreamDiffusionV2/demo/run.sh logs
/data/StreamDiffusionV2/demo/run.sh stop
/data/StreamDiffusionV2/demo/run.sh restart
```

启动成功后访问：`http://<本机IP>:7860`

---

## 一、三种启动方式

### 方式 A：`demo/run.sh`（推荐）

容器/进程管理友好，封装了 nohup、pidfile、日志轮转、端口清理。
**已内置最优默认**（`STREAMDIFF_DISABLE_FLASH=1`、`FAST=1`、`USE_TENSORRT=1`、`USE_TAEHV=1`、`GPU_IDS=0,1`、`TARGET_LATENCY=0.4`、`ENABLE_METRICS=1`、`SKIP_FRONTEND_BUILD=1`）。

```bash
cd /data/StreamDiffusionV2

demo/run.sh start     # 启动（后台），日志写 logs/server.log，pid 写 logs/demo.pid
demo/run.sh status    # 进程 + 端口 + STREAMDIFF_DISABLE_FLASH 值
demo/run.sh logs      # tail -F 实时日志
demo/run.sh stop      # 优雅停（TERM→KILL，回收 7860/29500 端口）
demo/run.sh restart   # 停 + 启
```

产物路径：

```
logs/server.log          当前日志
logs/server.log.YYYY...  上次启动时被轮转的日志
logs/demo.pid            当前 PID
```

### 方式 B：`demo/start.sh`（前台直跑）

调试用，输出直接打到当前终端，`Ctrl+C` 退出：

```bash
cd /data/StreamDiffusionV2/demo
./start.sh
```

`start.sh` 默认会跑 `npm install && npm run build` 构建前端。如果前端已构建（生产环境），用 `SKIP_FRONTEND_BUILD=1`：

```bash
SKIP_FRONTEND_BUILD=1 ./start.sh
```

### 方式 C：systemd（裸机/持久化部署）

仓库随附 `/etc/systemd/system/streamdiff-demo.service`：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now streamdiff-demo
sudo systemctl status streamdiff-demo
sudo journalctl -u streamdiff-demo -f       # 或 tail logs/server.log
sudo systemctl restart streamdiff-demo
```

> 容器内通常无 systemd，用 **方式 A** 即可。

---

## 二、环境变量

`run.sh` / `start.sh` / systemd unit 都识别以下变量。`run.sh` 自带"生产默认"，`start.sh` 默认更朴素。

| 变量 | run.sh 默认 | start.sh 默认 | 说明 |
|---|---|---|---|
| `PORT` | 7860 | 7860 | HTTP/WS 端口 |
| `HOST` | 0.0.0.0 | 0.0.0.0 | 监听地址 |
| `GPU_IDS` | `0,1` | `0` | 用哪几张卡（逗号分隔） |
| `STEP` | 1 | 1 | 推理步数 |
| `MODEL_TYPE` | `T2V-1.3B` | `T2V-1.3B` | 模型类型 |
| `USE_TAEHV` | 1 | 0 | TAEHV VAE，开后 → `--use_taehv` |
| `USE_TENSORRT` | 1 | 0 | TensorRT，开后 → `--use_tensorrt` |
| `FAST` | 1 | 0 | fast 路径，开后 → `--fast` |
| `TARGET_LATENCY` | 0.4 | —— | 目标延迟（秒），→ `--target-latency` |
| `ENABLE_METRICS` | 1 | 0 | batch 指标日志，→ `--enable-metrics` |
| `SKIP_FRONTEND_BUILD` | 1 | 0 | 1 = 跳过前端构建（已 build 过） |
| `PYTHON_BIN` | 自动找 `.venv/bin/python` | 同左 | 解释器路径 |
| **`STREAMDIFF_DISABLE_FLASH`** | **1** | **1** | **1=关 FA 走 SDPA（H20 推荐）；0=用 flash-attn** |
| `EXTRA_ARGS` | —— | —— | 透传给 `main.py` 的额外参数（高级） |

**布尔变量识别**：`1 / true / yes / on`（大小写不敏感）。

---

## 三、常用启动场景

### 1. 默认生产配置（最快）

```bash
demo/run.sh start
```

等价 CLI：

```text
python main.py --host 0.0.0.0 --port 7860 \
  --num_gpus 2 --gpu_ids 0,1 --step 1 --model_type T2V-1.3B \
  --fast --use_taehv --use_tensorrt \
  --target-latency 0.4 --enable-metrics
```

### 2. 单卡

```bash
GPU_IDS=0 demo/run.sh restart
```

### 3. 自定义端口

```bash
PORT=18860 demo/run.sh restart
```

### 4. A/B 对比：开回 flash-attn

```bash
STREAMDIFF_DISABLE_FLASH=0 demo/run.sh restart
demo/run.sh logs
# 用日志里的 avg_latency / batch 间隔对照 LATENCY_OPTIMIZATION.md §2 的表格
```

### 5. 前端首次/改动后构建

```bash
# run.sh 默认 SKIP_FRONTEND_BUILD=1，需要构建时显式覆盖：
SKIP_FRONTEND_BUILD=0 demo/run.sh restart
# 或直接走 start.sh 前台跑（默认会 build）
cd demo && ./start.sh
```

### 6. 关 TensorRT / fast 排查问题

```bash
USE_TENSORRT=0 FAST=0 demo/run.sh restart
```

### 7. 透传额外 CLI 参数

```bash
EXTRA_ARGS="--max-batch 8 --warmup 3" demo/run.sh restart
```

---

## 四、健康检查

启动后确认 4 件事：

```bash
# 1) 进程存活 + 端口监听
demo/run.sh status

# 2) 端口可达
ss -lntp | grep 7860
curl -sI http://127.0.0.1:7860/ | head -1

# 3) 当前生效的关键环境变量（特别是 FA 开关）
PID=$(cat /data/StreamDiffusionV2/logs/demo.pid)
tr '\0' '\n' </proc/$PID/environ | grep -E 'STREAMDIFF_DISABLE_FLASH|CUDA_VISIBLE_DEVICES|PORT'

# 4) 日志里有 metrics 输出（确认推理在跑）
tail -n 50 /data/StreamDiffusionV2/logs/server.log | grep -E 'avg_latency|Batch'
```

---

## 五、常见问题

### Q1：端口被占
```bash
ss -lntp | grep 7860       # 看是谁在用
demo/run.sh stop           # 优雅停（自带 fuser -k 7860/tcp 兜底）
# 还不行：
fuser -k 7860/tcp ; fuser -k 29500/tcp
```

### Q2：start 失败
```bash
demo/run.sh status         # not running 时看下面日志
tail -200 /data/StreamDiffusionV2/logs/server.log
```
常见原因：
- 没装好依赖 → 检查 `.venv/bin/python -m pip list`（参考 `requirements.lock.txt`）
- 显存不够 → 减少 `GPU_IDS` 或关 `USE_TENSORRT`
- 前端目录缺失 → `SKIP_FRONTEND_BUILD=0` 重跑一次

### Q3：FA / SDPA 切换没生效
检查环境变量是否被进程实际读到：
```bash
PID=$(cat /data/StreamDiffusionV2/logs/demo.pid)
tr '\0' '\n' </proc/$PID/environ | grep STREAMDIFF_DISABLE_FLASH
```
应输出 `STREAMDIFF_DISABLE_FLASH=1`（默认）或 `=0`。

### Q4：延迟回归
对照 `LATENCY_OPTIMIZATION.md` §2 表格，重点确认：
- `STREAMDIFF_DISABLE_FLASH=1`
- `FAST=1` `USE_TENSORRT=1` `USE_TAEHV=1`
- 日志中 `avg_latency` 是否在 0.4s 量级

---

## 六、相关文件

| 文件 | 用途 |
|---|---|
| `demo/run.sh` | 进程管理（start/stop/status/logs/restart） |
| `demo/start.sh` | 实际启动脚本（处理 env → CLI flags） |
| `demo/main.py` | 服务入口 |
| `/etc/systemd/system/streamdiff-demo.service` | systemd unit |
| `logs/server.log` | 运行日志（`run.sh` 自动轮转） |
| `logs/demo.pid` | 当前 PID 文件 |
| `LATENCY_OPTIMIZATION.md` | 延迟优化背景与各开关原理 |
| `V2V_TUNING_GUIDE.md` | V2V 模式调参指南 |
| `requirements.lock.txt` | 依赖快照（`pip freeze`） |
