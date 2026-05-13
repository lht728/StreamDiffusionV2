# P0 / P1 延时优化落地记录（A.1 / A.2 / A.3 / A.4 / A.5）

本文档记录针对帧到帧延时报告中"高 ROI"建议（A.1–A.5）的实际改动、行为差异、回退方式与验证步骤。

---

## 改动一览

| 项 | 文件 | 类型 | 默认是否启用 |
|---|---|---|---|
| A.1 出队侧改为事件驱动（去掉 EMA + sleep） | `demo/main.py:generate()` | 行为变更 | 是 |
| A.2-a 上行 throttle 24→32 FPS | `demo/main.py:handle_websocket_data` & upload-mode | 配置 | 是 |
| A.2-b `read_images_from_queue` 轮询 10ms→3ms | `demo/util.py` | 行为变更（可调） | 是 |
| A.4 GPU 端 fuse → uint8 D2H | `streamv2v/inference{,_pipe,_wo_batch}.py:_decode_video_array` & `inference_pipe.py:_decode_prediction` | 数据契约变更 | 是 |
| A.4 配套 `array_to_image` uint8 fast-path | `demo/util.py:array_to_image` | 兼容增强 | 是 |
| A.5 MJPEG 编码 turbojpeg 优先 | `demo/util.py:pil_to_frame` + 新增 `ndarray_uint8_to_frame` | 自动 fallback | 装了就用 |
| A.3 VAE decode / DiT overlap | `streamv2v/inference_pipe.py` 新增 `_decode_prediction_async` / `_decode_prediction_finish`，`demo/vid2vid_pipe.py:output_process` | opt-in | 否（`DEMO_DECODE_OVERLAP=1` 启用） |
| 移除非 Firefox 浏览器双 yield | `demo/main.py:generate()` | 行为变更 | 是 |

---

## 详细说明

### A.1 出队侧改为事件驱动

`generate()` 历史实现：
- EMA 估计帧间隔，`MIN_FPS=10` 兜底 ⇒ 最坏 100 ms `asyncio.sleep`
- 每轮先 `get_output_queue_size()` → `get_frame()` → `await asyncio.sleep(sleep_time)`

新实现：
- 直接 `await self.conn_manager.get_frame(user_id)`
- `output_queue` 已经是 `asyncio.Queue`，`await get()` 在帧到达瞬间唤醒，无需补 sleep
- 同步去除 `is_firefox` 反转分支带来的非 Firefox 双 yield（实测会让下行带宽翻倍且使解码端排队延迟翻倍）

**预期收益**：p50 -5~15 ms（去掉 sleep 残值），p99 -50~100 ms（极端慢帧时不再被 100 ms 兜底卡住）。

### A.2 降低 chunk-fill 头帧等待

| 维度 | 旧 | 新 |
|---|---|---|
| 上行 WS throttle | 24 FPS（chunk-fill 最坏 ≈125 ms） | **32 FPS**（chunk-fill 最坏 ≈94 ms） |
| `read_images_from_queue` 轮询 | 10 ms | **3 ms**（可通过 `DEMO_QUEUE_POLL_INTERVAL_S` 调整） |

32 FPS 上行与下游产能（DiT+VAE ~225 ms/chunk ⇒ 等效 ~17.8 fps）匹配，富余被 `connection_manager.max_output_queue_size` 兜住，不会导致上行队列无限堆积。

**预期收益**：chunk 头帧延时 -30 ms（125→94 ms），叠加 3 ms 轮询节省 jitter ~7 ms。

### A.4 GPU 端直接吐 uint8

```diff
- video = (video * 0.5 + 0.5).clamp(0, 1)
- video = video[0].permute(0, 2, 3, 1).contiguous()
- return video.detach().cpu().float().numpy()   # bf16→fp32→4×H×W×3 字节
+ video = video.mul(127.5).add_(127.5).clamp_(0, 255)
+ video = video[0].permute(0, 2, 3, 1).contiguous()
+ video = video.to(torch.uint8)                   # GPU 端 fuse
+ return video.cpu().numpy()                       # uint8 → 1×H×W×3 字节
```

PCIe D2H 流量降到 **1/4**（fp32 → uint8）。配套 `array_to_image` 自动检测 `dtype==uint8` 走零拷贝 `Image.fromarray` 路径，跳过 `*255 + astype` 的 host-side 复读。

**注意**：`run_inference` 离线写视频文件路径（`results[save_results] = video.cpu().float().numpy()`）保持 fp32，因为 `export_to_video` 期待 [0,1] float 输入；这条路径不在 demo 热路径，**未改动**。

### A.5 MJPEG 编码加速

`demo/util.py` 顶部尝试 `from turbojpeg import TurboJPEG`：
- 装了 PyTurboJPEG（`pip install -e .[fast-mjpeg]`，运行时还需要系统的 libjpeg-turbo）：JPEG 编码 ~6× PIL
- 没装：自动回退到 PIL；启动 logger 会打一条 INFO 提示

新增 `ndarray_uint8_to_frame(arr)` 直接消费 numpy uint8，跳过 PIL 中转——为后续 "GPU uint8 → 直接 JPEG"（不经 `array_to_image`）路径做准备，当前还未在主路径切换（保留 `pil_to_frame` 兼容），可在下一波改造里启用。

JPEG 质量默认 85（环境变量 `DEMO_JPEG_QUALITY` 调整）。

### A.3 VAE decode / DiT overlap（**默认关闭**）

`InferencePipelineManager.__init__` 新增 `self.decode_stream = torch.cuda.Stream()` + 持久 pinned host buffer。

```python
def _decode_prediction_async(self, denoised_pred):
    decode_stream.wait_stream(default_stream)
    with stream(decode_stream):
        video = vae.stream_decode_to_pixel(...)
        video = fuse_to_uint8(video)
        host_buffer.copy_(video, non_blocking=True)   # 真正异步 D2H（pinned）
        event.record(decode_stream)

def _decode_prediction_finish(self):
    event.synchronize()
    return host_buffer.numpy().copy()
```

`output_process` 使用：
- `DEMO_DECODE_OVERLAP=0`（默认）：保留旧的同步 `_decode_prediction`，已经享受 A.4 的 uint8 D2H 收益
- `DEMO_DECODE_OVERLAP=1`：流水化 —— chunk N decode 在 decode_stream，chunk N+1 的 NCCL recv + DiT forward 在 default + com_stream，**净 chunk wall -10~25 ms，但单帧最早可见时间相对推后 1 chunk**

何时开启：在你的部署上跑过 1000 batch 对比，确认 `chunk_wall > frame_interval`（即 throughput-bound）时开启；如果 chunk_wall 已经低于帧间隔（latency-bound），开启会得不偿失。

---

## 数据契约变化（重要）

`MultiGPUPipeline.output_queue` 与 `Pipeline.output_queue` 中的 numpy 帧从历史的 **`float32`/`[0,1]`** 变为 **`uint8`/`[0,255]`**。

链上消费方：
- `vid2vid.Pipeline.produce_outputs` → `array_to_image(...)` ✅ 已支持自动检测 uint8
- `streamdiffusionv2/pipeline.py:decode_chunk` → 调用 `_decode_video_array` ✅ 直接拿到 uint8
- 离线 `run_inference` 路径独立保持 fp32 ✅ 未变

如果你有外部脚本/测试直接消费这些 numpy（比如 `decode_chunks` 后接 `* 255` 二次缩放），需要同步去掉那次缩放。

---

## 验证清单

### 单元级别

```bash
cd /data/StreamDiffusionV2
.venv/bin/python -c "
import py_compile
for f in ['demo/main.py','demo/util.py','demo/vid2vid_pipe.py',
         'streamv2v/inference_pipe.py','streamv2v/inference.py','streamv2v/inference_wo_batch.py']:
    py_compile.compile(f, doraise=True); print('OK', f)
"
```

### 端到端（双卡 H20）

1. **基线**（同步路径，A.1/A.2/A.4/A.5 默认生效）：
   ```bash
   cd /data/StreamDiffusionV2/demo
   ENABLE_METRICS=1 TARGET_LATENCY=0.4 \
     ../.venv/bin/python main.py --gpu-ids 0,1 --step 1 --fast --enable-metrics \
       --target-latency 0.4 2>&1 | tee logs/server.log.p0_baseline
   ```
   连接前端跑满 1000 batch，看 `slo_metrics/.../statistics_*.json` 的 `mean_latency / p50 / p95 / p99 / deadline_miss_rate`。

2. **opt-in overlap**：
   ```bash
   DEMO_DECODE_OVERLAP=1 ../.venv/bin/python main.py ...
   ```

3. **可选：装 PyTurboJPEG 进一步压缩 CPU 编码时间**：
   ```bash
   sudo apt-get install libturbojpeg
   ../.venv/bin/pip install -e ".[fast-mjpeg]"
   ```
   启动时 logger 应打：`MJPEG encoder: PyTurboJPEG (quality=85)`。

### 预期数值（H20×2，512×512，step=1，fast）

| 指标 | 当前基线 | A.1+A.2+A.4 后 | + A.5 (turbojpeg) | + A.3 (overlap=1) |
|---|---|---|---|---|
| mean | 416 ms | ~330 ms | ~315 ms | ~290 ms |
| p99 | 588 ms | ~480 ms | ~460 ms | ~430 ms |
| chunk wall (median) | 226 ms | 220 ms | 215 ms | **~190 ms** |
| jitter p90 | 178 ms | ~120 ms | ~115 ms | ~110 ms |

> 数值为根据延时模型推算的目标范围，**实际请以 1000-batch 实测为准**。
> 若 p99 较预期差 >50 ms，先排查 `slo_metrics` 中 batch ≥ 100 的样本是否被预热样本污染（参考 LATENCY_OPTIMIZATION.md 中 B.8 异常样本过滤）。

---

## 回退方法

| 变更 | 回退 |
|---|---|
| A.1 事件驱动 | revert `demo/main.py:generate()` |
| A.2-a 32 FPS | `TARGET_FPS = 24.0` 改回 |
| A.2-b 3 ms 轮询 | 设 `DEMO_QUEUE_POLL_INTERVAL_S=0.01` |
| A.3 overlap | `DEMO_DECODE_OVERLAP=0`（默认即关） |
| A.4 uint8 路径 | revert `_decode_video_array` / `_decode_prediction`；同步把 `array_to_image` 旧版本恢复 |
| A.5 turbojpeg | 卸载 `PyTurboJPEG` 即自动 fallback PIL |
| 移除非 Firefox 双 yield | revert `generate()` 中 `is_firefox` 那一行 |
