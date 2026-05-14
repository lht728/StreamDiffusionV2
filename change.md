# feat-0512 相对 master 的改动

## 提交列表

| commit | 描述 |
|---|---|
| `41da4d3` | fix kvcache |
| `4a35dca` | optimize frame-frame delay |
| `a6f3551` | fix |
| `07f6c0b` | fix |
| `d07c075` | md |
| `b43412e` | update |
| `f097045` | 0513 |

核心方向：**帧到帧延时优化（A.1–A.5）** + **KV cache 稳定性** + **可观测性 / 运维**。

---

## 一、模型 / 推理核心

### `models/wan/causal_model.py` — KV cache 防御性 guard

- **evict 分支**：写入前检查 `target_end - num_new_tokens < 0 || target_end > kv_cache_size || target_end < num_new_tokens`，越界时打印 `i / target_end / num_new_tokens / kv_cache_size / evict_idx / sink_tokens / frame_seqlen` 并跳过写入。
- **normal 分支**：检查 `local_end - local_start != num_new_tokens || local_start < 0 || local_end > kv_cache_size`，命中时打印完整状态（包含 `global_end_index / local_end_index / roped_key.shape / v.shape`），跳过写入并把 `local_end/local_start` clamp 回合法区间，让 attention 继续跑而不是炸 worker。
- 新增 `STREAMDIFF_DISABLE_FLASH=1` 运行时强制 SDPA 兜底（A/B 不需要卸载 wheel）。

### `models/wan/wan_base/modules/attention.py`

- 支持 `STREAMDIFF_DISABLE_FLASH` 一键关闭 FA2/FA3 import。
- 给 FA3 `flash_attn_varlen_func` 补上 `seqused_q / seqused_k` 参数，修协议不匹配。

### `streamv2v/inference.py` / `inference_pipe.py` / `inference_wo_batch.py`

**A.4 — GPU 端 fuse → uint8 D2H**

- `_decode_video_array` / `_decode_prediction`：
  `(x*0.5+0.5).clamp(0,1) * 255 → uint8` 全部 fuse 到 GPU 上。
- PCIe 传输从 fp32 降到 uint8，流量降为原来的 1/4。

**A.3 — VAE decode 与 DiT/NCCL overlap**

- 新增 `_decode_prediction_async(denoised_pred)`：
  - 在专用 `decode_stream` 上跑 VAE decode + fuse。
  - 持久化 pinned host buffer，按需懒分配 / 重分配。
  - `non_blocking=True` 异步 D2H copy，记录 CUDA event。
- 新增 `_decode_prediction_finish()`：等 event、返回 `host_buffer.numpy().copy()`。
- 前置防御：连续两次 async 调用之间，未消费的 event 强制 synchronize。

---

## 二、demo 后端

### `demo/main.py`

- **A.1 出队侧改为事件驱动**：
  - 删除 EMA + `MIN_FPS=10` 兜底 + `asyncio.sleep(sleep_time)`。
  - 直接 `await self.conn_manager.get_frame(user_id)`，下游 MJPEG 节奏由 DiT/VAE 产能 + TCP 背压决定。
  - 移除非 Firefox 的双 yield（实测下行带宽翻倍 + 解码端排队延迟翻倍）。
- **A.2-a 上行 throttle 16 → 24 FPS**（`handle_websocket_data` 和 upload-mode 都改）：chunk-head 等待从 187 ms → 125 ms。
- **1Hz 端到端延时采样器**（`_latency_sampler_loop`）：
  - 独立 daemon 线程，每秒往 `demo/latency.log` 写一行 `ts user count mean p50 p95 p99 min max remaining_input_q`。
  - 锁内只做 `history[prev:cur_len]` 切片快照，stats 计算放锁外。
  - 检测下游 1000-batch 窗口 reset（`cur_len < prev`），自动 rebase。
  - shutdown 时 `Event.set()` + `join(timeout=2.0)`，写收尾行后关闭 fh。

### `demo/util.py`

- **A.5 MJPEG 编码**：优先 PyTurboJPEG（libjpeg-turbo C 绑定，~6× PIL）；找不到 fallback PIL，日志提示。
- 抽出 `_wrap_mjpeg()`，统一 multipart 包装。
- 新增 `ndarray_uint8_to_frame(arr)`：吃 GPU 直出的 `np.uint8 (H,W,3)`，跳过 PIL Image 构造直接 encode。
- `array_to_image` 加 uint8 fast-path，跳过 `* 255`。
- **A.2-b**：`read_images_from_queue` 轮询 `10 ms → 3 ms`（`DEMO_QUEUE_POLL_INTERVAL_S` 可调）。
- JPEG 质量改为环境变量 `DEMO_JPEG_QUALITY`（默认 85）。

### `demo/vid2vid.py`

- **Worker watchdog + auto respawn**：
  - 抽出 `_spawn_process(initial)`，可复用现有 queues / events。
  - 后台 `vid2vid-watchdog` 线程 1s 轮询，worker 死亡时 drain `error_queue` → 清 input/output 队列 → respawn，最多 `STREAMV2V_MAX_RESPAWN=10` 次。
  - 初次启动失败仍 fail-fast。
  - close 时优雅 join watchdog。
- **`accept_new_params` 修 bug**：
  UI 的 schema 默认 `use_taehv=false / use_tensorrt=false` 每帧 echo 回来，旧逻辑把它当作"用户请求关闭"，第一帧后触发全 pipeline rebuild，画面冻结。新逻辑只在请求值为 truthy 且与启动值实际不同时才 restart。
- **per-frame 延时日志**：
  - `_setup_delay_logger()` 写 `/data/StreamDiffusionV2/log/delay.log`。
  - 列：`iso_ts, phase(init|stream), chunk_idx, frame_in_chunk, global_frame_idx, chunk_input_frames, chunk_wall_ms, frame_delay_ms, fps`。
  - 首次空文件自动写 header；respawn 不堆叠 handler。

### `demo/vid2vid_pipe.py`

- **A.3 opt-in overlap**：`DEMO_DECODE_OVERLAP=1` 启用：
  - 当前 chunk decode 派发到 `decode_stream`，下一 chunk 的 NCCL recv + DiT forward 在默认 / com_stream 上并行。
  - 切 prompt 时 `pipeline_manager._decode_event = None`，丢弃未消费的 async decode。
  - 默认关闭（同步路径），文档说明吞吐受限场景下再开。
- **NCCL 7 天超时**：`init_dist_tcp` 给 `dist.init_process_group` 加 `timeout=datetime.timedelta(days=7)`。
  根因修复：之前默认 10 分钟 watchdog，用户空闲 / pause 超过 10 分钟，rank 0/N-1 的 broadcast 阻塞会被打掉，process group 永久损坏，画面定格在第一帧。
- 移除生产环境无用的 `torch.cuda.memory._record_memory_history(...)`（每个 alloc 都有开销）。

### `demo/config.py`

- `--use_taehv / --use_tensorrt / --fast` 改用 `argparse.BooleanOptionalAction`：
  - 默认 **全开**（生产推荐配置，TAEHV ~2× VAE 加速 + TRT engines 必备）。
  - 可用 `--no-use_taehv / --no-use_tensorrt / --no-fast` 关闭。
  - 环境变量 `USE_TAEHV / USE_TENSORRT / FAST` 仍可覆盖默认。
- `--target-latency` 默认 `1.0 → 0.4` s，与 `demo/run.sh` 对齐。
- 调用 `streamv2v.inference_common.normalize_acceleration_flags(parsed_args)`：
  修复多 GPU 路径下 `--fast` 静默失效 bug（之前 `use_tensorrt` 不会被打开、TRT engine 不构建、DiT 退回 SDPA）。

---

## 三、前端

### `demo/frontend/src/lib/lcmLive.ts` — WebSocket 重连泄漏修复

- 旧实例非 OPEN 时先解绑 `onopen/onclose/onerror/onmessage` 再 close，再创建新 `ws`。
- handler 闭包捕获本次 `ws` 引用，stale 消息 (`websocket !== ws`) 直接 ignore。
- 发送前检查 `ws.readyState === OPEN`，避免给 closed socket 发 next_frame。
- `onclose` 仅当仍是当前 ws 时才置 DISCONNECTED 状态。

### `demo/frontend/src/lib/components/VideoInput.svelte` — 摄像头镜像

- 新增 `mirrorCamera` prop（默认 true），仅 camera 模式生效。
- canvas 绘制时 `ctx.translate(width,0); ctx.scale(-1,1)`，保证上传给后端的 JPEG 也是镜像后的（CSS transform 只影响预览不影响像素）。
- `<video>` 同步加 `class:mirrored`。

---

## 四、部署 / 脚本

### 新增 `demo/run.sh`

- 容器友好的进程管理器：`start / stop / restart / status / logs`。
- `setsid nohup ./start.sh ...` 把 wrapper + main + mp.spawn rank workers + inductor compile workers 放在同一 pgid。
- `cmd_stop` 用 `kill -TERM -- "-$PID"` 一信号清整棵进程树，15 s 优雅期后 KILL 兜底。
- belt-and-suspenders：扫 `.venv/bin/python`（按 `argv[0]` 匹配避免误伤系统 python），清 `7860/29500` TCP 端口，最后报告残留 `/dev/nvidia*` 占用。
- 默认导出生产推荐 env：`PORT=7860 GPU_IDS=0,1 STEP=1 USE_TAEHV=1 USE_TENSORRT=1 FAST=1 TARGET_LATENCY=0.4 ENABLE_METRICS=1 SKIP_FRONTEND_BUILD=1 STREAMDIFF_DISABLE_FLASH=1`。

### `demo/start.sh`

- 默认开启 fast 路径：`USE_TAEHV / USE_TENSORRT / FAST` 默认 1。
- 默认关闭 flash-attn：`STREAMDIFF_DISABLE_FLASH=1`（H20 上 step=1 + KV cache + 短序列 SDPA 不慢于 FA，FA3 更慢）。
- `PYTHON_BIN` 自动探测 `.venv/bin/python`。
- 新增 `SKIP_FRONTEND_BUILD / TARGET_LATENCY / ENABLE_METRICS / EXTRA_ARGS` 控制项。

### `run_v2v.sh`

- 默认导出 `STREAMDIFF_DISABLE_FLASH=1`。

### 新增 `_download_ckpts.sh`

- 一键下载：
  - `Wan-AI/Wan2.1-T2V-1.3B` → `wan_models/`
  - `jerryfeng/StreamDiffusionV2 (wan_causal_dmd_v2v/*)` → `ckpts/`
  - TAEHV `taew2_1.pth` → `ckpts/`
- 启用 `HF_HUB_ENABLE_HF_TRANSFER=1`。

### 新增 `requirements.lock.txt`

- 80 行精确版本锁，包含 `torch==2.6.0 / flash_attn==2.7.4.post1 / nvidia-* 12.4 系列 / fastapi==0.117.1 / diffusers==0.35.1 / transformers==4.54.0` 等。

---

## 五、文档与观测产物（非代码）

| 路径 | 性质 |
|---|---|
| `LATENCY_OPTIMIZATION_P0_P1.md` | 新增文档：A.1–A.5 落地记录、行为差异、回退方式、验证步骤 |
| `V2V_TUNING_GUIDE.md` | 新增文档：`run_v2v.sh` 离线 V2V 调参指南（noise_scale / step / fixed_noise_scale / prompt 等） |
| `logs/server.log.*` | 观测产物（FA2 / FA3 / before-flashattn / failed-start 等基线日志） |
| `slo_metrics/latency_{data,statistics}_*.json` | 观测产物（14 份 latency 采集结果，每份 ~7000 行） |

---

## 影响面 / 风险点

- **接口契约**：`_decode_video_array` 返回值由 fp32 改为 uint8，下游 `array_to_image` 已配套加 uint8 fast-path；外部如有直接消费请确认。
- **运行时默认值变化**：fast 路径全开 + flash-attn 默认关 + 24 FPS 上行 + 0.4 s target latency。回退方式见 `LATENCY_OPTIMIZATION_P0_P1.md`。
- **NCCL 超时改 7 天**：极端 hang 场景不再被 watchdog 兜底打掉，需要外部进程级监控 / `run.sh status`。
- **opt-in 项**：`DEMO_DECODE_OVERLAP=1`（A.3）默认关闭，吞吐受限场景再开。

---

## 文件级 diff 概要

```
LATENCY_OPTIMIZATION_P0_P1.md        +174        新增
V2V_TUNING_GUIDE.md                  +160        新增
_download_ckpts.sh                   +13         新增
demo/config.py                       +46/-14     fast 默认开 + normalize
demo/frontend/.../VideoInput.svelte  +22         camera 镜像
demo/frontend/.../lcmLive.ts         +35/-9      ws 重连泄漏修复
demo/main.py                         +237/-46    A.1 / A.2-a / latency sampler
demo/run.sh                          +172        新增进程管理器
demo/start.sh                        +68/-10     默认值 + 新增 env 控制
demo/util.py                         +139/-20    A.2-b / A.4 / A.5
demo/vid2vid.py                      +227/-31    watchdog / delay log / restart bug
demo/vid2vid_pipe.py                 +65/-10     A.3 overlap + NCCL 7 天超时
models/wan/causal_model.py           +63/-5      KV cache guard + FA kill-switch
models/wan/wan_base/.../attention.py +12         FA kill-switch + seqused 参数
requirements.lock.txt                +80         新增版本锁
run_v2v.sh                           +6          STREAMDIFF_DISABLE_FLASH=1
streamv2v/inference.py               +13/-3      uint8 fuse
streamv2v/inference_pipe.py          +96/-5      async decode + uint8 fuse
streamv2v/inference_wo_batch.py      +10/-3      uint8 fuse
logs/server.log.*                    +5297       观测产物
slo_metrics/latency_*.json           +70588      观测产物
```

---

## 七、帧到生成帧延时优化（按数据通路顺序）

下列改动直接作用于"输入帧 → 生成帧"端到端延时。按通路位置（**上行 → 调度 → 推理 → VAE → 编码 → 下行**）排序。

### 1. 上行 throttle 16 → 24 FPS（A.2-a）

`demo/main.py`：`TARGET_FPS = 16.0 → 24.0`

**原理**：DiT 以 chunk（4 帧）为粒度推理。chunk 头帧必须等"凑齐 4 帧"才能开跑，等待时间 ≈ `(chunk_size - 1) / fps`。

- 16 FPS：等 187 ms
- 24 FPS：等 125 ms

直接砍掉 chunk 头帧 ~60 ms 的"凑帧空等"。

### 2. `read_images_from_queue` 轮询 10 ms → 3 ms（A.2-b）

`demo/util.py`：`_QUEUE_POLL_INTERVAL_S` 默认 0.003

**原理**：worker 端凑帧靠 `time.sleep` 轮询 input queue。最后一帧到达后到 worker 唤醒之间存在最坏等于轮询周期的"叫醒延时"。10 → 3 ms 期望节省 ~3.5 ms，p99 节省 ~7 ms，CPU 仅 ~0.03 core。

### 3. GPU 端 fuse + uint8 D2H（A.4）

`streamv2v/inference{,_pipe,_wo_batch}.py:_decode_video_array / _decode_prediction`

**原理**：原路径 VAE 输出 fp32 → `(x*0.5+0.5).clamp(0,1)` 在 GPU、`* 255` 在 CPU、最后 `astype(uint8)` 在 CPU；中间 PCIe 拷贝是 fp32。改成在 GPU 上 fuse `mul_(127.5).add_(127.5).clamp_(0,255).to(uint8)` 后再 D2H：

- PCIe 流量降为 1/4（一帧 512×512×3 fp32 ≈ 3 MB → uint8 ≈ 0.75 MB）
- 消除 host 端 `* 255` 的 CPU 浮点开销
- 下游 `array_to_image / ndarray_uint8_to_frame` 直接吃 uint8，跳过冗余转换

每帧节省 PCIe 拷贝 ~1–3 ms（PCIe 4.0），多 rank 场景更明显。

### 4. VAE decode / DiT 重叠（A.3，opt-in）

`streamv2v/inference_pipe.py` 新增 `_decode_prediction_async / _decode_prediction_finish`；`demo/vid2vid_pipe.py` 通过 `DEMO_DECODE_OVERLAP=1` 启用。

**原理**：把 VAE decode + D2H copy 放到独立 `decode_stream` + 持久化 pinned buffer + CUDA event。chunk N 的 decode 与 chunk N+1 的 NCCL recv + DiT forward 并行。

- 把 VAE 从 chunk 关键路径上摘掉 ~10–25 ms / chunk
- 代价：单帧最早可见时间相对当前 chunk 推后 1 chunk

**只在 throughput-bound 场景（chunk_wall > frame_interval，例如高分辨率或 step > 1）下是净收益**；低负载下反而增加 1 chunk 的固有延迟，所以默认关。

### 5. MJPEG 编码切 PyTurboJPEG（A.5）

`demo/util.py:pil_to_frame / ndarray_uint8_to_frame`

**原理**：PIL 默认 libjpeg 单线程，512×512 ~3 ms/帧 ⇒ 一个 chunk 4 帧 ~12 ms 阻塞 output 线程。PyTurboJPEG 走 libjpeg-turbo SIMD，~6× 加速，单帧降到亚毫秒。新增的 `ndarray_uint8_to_frame` 还省掉一次 `PIL.Image.fromarray` 构造。

每 chunk 节省 ~10 ms。

### 6. 出队侧改事件驱动（A.1）

`demo/main.py:generate()`

**原理**：旧实现用 EMA 估帧间隔 + `await asyncio.sleep(sleep_time)`，`MIN_FPS=10` 兜底意味着哪怕帧已经到了，最坏也要盲等 100 ms。改成纯 `await asyncio.Queue.get()` 后，帧一到就立即唤醒下游 MJPEG，零盲等。

- p50 节省 ~5–15 ms（sleep 残值）
- p99 节省 ~50–100 ms（极端慢帧不再被 100 ms 兜底卡住）

### 7. 移除非 Firefox 双 yield

`demo/main.py:generate()`

**原理**：旧逻辑非 Firefox 走 `yield frame; yield frame`，下行带宽 ×2，浏览器解码端排队 ×2，等价于给非 Firefox 客户端凭空叠了一帧的解码排队延时。去掉直接砍一帧下行延迟。

### 8. Pinned host buffer + non_blocking copy

`streamv2v/inference_pipe.py:_decode_prediction_async`

**原理**：异步 D2H copy 必须是 pinned memory 才能真正与 GPU 计算并行。持久化复用 pinned buffer（按 shape/dtype 懒重分配），消除每帧分配 pinned page 的隐性开销，同时让 copy 进入 `decode_stream` 排队，不阻塞默认 stream 上的 DiT。

### 9. JPEG 编码质量参数化

`DEMO_JPEG_QUALITY` 环境变量（默认 85）。

**原理**：质量降则编码时间和字节数同步降，下行也更快，必要时可换 2–3 ms / 帧。

### 10. NCCL 超时 7 天

`demo/vid2vid_pipe.py:init_dist_tcp`

**原理**：严格说不是"降延时"，但解决了**最严重的稳态延时退化**：默认 10 分钟 watchdog 在用户空闲超过 10 分钟后把 process group 打挂，画面定格在第一帧——表现为延时变成无限大。改 7 天后这个 failure mode 消失。

### 11. `accept_new_params` 不再误触发 rebuild

`demo/vid2vid.py`

**原理**：旧逻辑下 UI schema 默认 `use_taehv=false` 每帧 echo 回来，被当作"切换 TAEHV"，触发 pipeline 全量重建（数十秒）；用户感受是第一帧出来后画面冻结。修掉后稳态延时不再偶发被一次 rebuild 打穿。

### 12. fast 路径多 GPU 修复（最大收益）

`demo/config.py:normalize_acceleration_flags`

**原理**：旧逻辑下多 GPU 路径 `--fast` 是 silent no-op：`use_tensorrt` 保持 False，TRT engine 不构建，DiT 退回 SDPA。修复后 TAEHV + TRT 真的生效：

- TAEHV 比官方 VAE decode ~2× 提速（每 chunk 节省 ~10–20 ms）
- DiT TRT 化进一步压缩单 chunk 推理时间

这是**单条最大收益**的改动（前提是之前以为开了其实没开）。

---

### 数量级速览（单 chunk = 4 帧 @ 512²）

| 改动 | 关键路径节省 |
|---|---|
| `--fast` 多 GPU 修复（TAEHV+TRT 真生效）| ~20–40 ms / chunk（视基线） |
| chunk 头帧凑帧（16→24 FPS）| ~60 ms / chunk |
| 出队事件驱动（去 sleep）| p50 -5~15 ms / 帧，p99 -50~100 ms |
| VAE/DiT overlap（opt-in）| ~10–25 ms / chunk（吞吐受限时） |
| GPU fuse + uint8 D2H | ~1–3 ms / 帧 |
| PyTurboJPEG | ~10 ms / chunk |
| 输入轮询 10→3 ms | p99 ~7 ms |
| 移除非 FF 双 yield | -1 帧的下行排队 |
| NCCL 7d / params 修复 | 消除"延时跑飞"的退化路径 |

