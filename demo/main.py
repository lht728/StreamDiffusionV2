from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2

import logging
import uuid
import time
from types import SimpleNamespace
import asyncio
import os
import mimetypes
import threading
import multiprocessing as mp
import signal
import sys
from collections import deque

from config import config, Args
from util import pil_to_frame, bytes_to_pil, is_firefox
from connection_manager import ConnectionManager, ServerFullException

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120
LOGGER = logging.getLogger(__name__)


class App:
    def __init__(self, config: Args, pipeline):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.prediction_workers = {}
        self.shutdown_event = asyncio.Event()
        self.demo_root = os.path.dirname(os.path.abspath(__file__))
        self.frontend_public_dir = os.path.join(self.demo_root, "frontend", "public")
        # Initialize metrics collection only if enabled
        self.enable_metrics = config.enable_metrics
        self.target_latency = config.target_latency  # Target latency in seconds for deadline miss rate
        self.step = config.step  # Pipeline step parameter
        self.gpu_ids = config.gpu_ids  # GPU IDs (e.g., "0,1" or "0")
        if self.enable_metrics:
            # Simple timestamp queue for input frames (FIFO)
            self.user_input_timestamps = {}  # user_id -> deque of input timestamps
            self.user_metrics_lock = threading.Lock()  # Lock for thread-safe timestamp tracking
            # Track metrics collection count per user (number of batches collected)
            self.user_batch_count = {}  # user_id -> count of batches collected
            self.user_latency_history = {}  # user_id -> list of latencies for statistics
            self.user_raw_data = {}  # user_id -> list of raw batch data (for logging)
            self.metrics_log_dir = "./slo_metrics"
            os.makedirs(self.metrics_log_dir, exist_ok=True)
            # 1Hz end-to-end (input -> generated) latency sampler. Writes one
            # line per second to demo/latency.log summarising the latencies
            # that were *recorded during the last 1s window* (i.e. only the
            # newly-appended entries in user_latency_history). This is a
            # read-only consumer of the history list and uses
            # user_metrics_lock just long enough to snapshot a slice; the
            # heavy stats math runs outside the lock.
            self.latency_log_path = os.path.join(self.demo_root, "latency.log")
            self._latency_sampler_stop = threading.Event()
            self._latency_sampler_offsets = {}  # user_id -> last seen len(history)
            self._latency_sampler_thread = None
        self.init_app()

    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            finally:
                await self._stop_prediction_worker(user_id)
                # Do not block shutdown here; schedule disconnect
                asyncio.create_task(self.conn_manager.disconnect(user_id, self.pipeline))
                # Clean up metrics and timestamp tracking for this user
                if self.enable_metrics:
                    with self.user_metrics_lock:
                        self.user_input_timestamps.pop(user_id, None)
                        self.user_batch_count.pop(user_id, None)
                        self.user_latency_history.pop(user_id, None)
                        self.user_raw_data.pop(user_id, None)
                        # Drop sampler cursor so a future user_id with the
                        # same value (extremely unlikely with uuid4 but
                        # defensive) cannot inherit a stale offset.
                        self._latency_sampler_offsets.pop(user_id, None)
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")
            last_time = time.time()
            last_frame_time = None
            # Latency tuning: bump 16 -> 24 FPS so chunks (4 frames) fill up faster.
            # At 24 FPS chunk wait ~ (4-1)/24 = 125 ms (was 187 ms @ 16 FPS).
            TARGET_FPS = 24.0
            min_frame_interval = 1.0 / TARGET_FPS
            last_frame_received_time = None
            try:
                while True:
                    if (
                        self.args.timeout > 0
                        and time.time() - last_time > self.args.timeout
                    ):
                        await self.conn_manager.send_json(
                            user_id,
                            {
                                "status": "timeout",
                                "message": "Your session has ended",
                            },
                        )
                        await self.conn_manager.disconnect(user_id, self.pipeline)
                        return
                    data = await self.conn_manager.receive_json(user_id)
                    # Refresh idle timer on any client control message
                    last_time = time.time()
                    # Handle stop/pause without closing socket: go idle and wait
                    if data and data.get("status") == "pause":
                        params = SimpleNamespace(**{"restart": True})
                        await self.conn_manager.update_data(user_id, params)
                        continue
                    if data and data.get("status") == "resume":
                        await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                        continue
                    # Mark upload completion: after this, don't receive image bytes again
                    if data and data.get("status") == "upload_done":
                        self.conn_manager.set_video_upload_completed(user_id, True)
                        LOGGER.info("Upload completed for user %s", user_id)
                        await self.conn_manager.send_json(user_id, {"status": "upload_done_ack"})
                        continue
                    if not data or data.get("status") != "next_frame":
                        await asyncio.sleep(THROTTLE)
                        continue

                    params = await self.conn_manager.receive_json(user_id)
                    params = self.pipeline.InputParams(**params)
                    info = self.pipeline.Info()
                    params = self.pipeline.params_to_namespace(params)
                    
                    # Check if upload mode is enabled
                    is_upload_mode = params.__dict__.get('input_mode') == 'upload' or params.__dict__.get('upload_mode', False)
                    self.conn_manager.set_upload_mode(user_id, is_upload_mode)
                    if is_upload_mode:
                        LOGGER.debug("Upload mode detected for user %s", user_id)
                    
                    if info.input_mode == "image":
                        upload_completed = self.conn_manager.is_video_upload_completed(user_id)
                        # Only receive image bytes if not in upload mode, or upload not completed yet
                        if (not is_upload_mode) or (is_upload_mode and not upload_completed):
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if len(image_data) == 0:
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                # await asyncio.sleep(sleep_time)
                                continue
                            
                            # 16 FPS throttling: only process frames at 16 FPS rate
                            current_time = time.time()
                            if last_frame_received_time is not None:
                                time_since_last_frame = current_time - last_frame_received_time
                                if time_since_last_frame < min_frame_interval:
                                    # Skip this frame to maintain 16 FPS
                                    await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                                    continue
                            
                            last_frame_received_time = current_time
                            
                            # If upload mode and not completed, append frames to cache for later reuse
                            if is_upload_mode and not upload_completed:
                                await self.conn_manager.add_video_frame(user_id, image_data)
                                LOGGER.debug("Buffered uploaded frame for user %s", user_id)
                            # For camera mode, set current image directly
                            if not is_upload_mode:
                                params.image = bytes_to_pil(image_data)
                        else:
                            # Upload already completed: do not receive more bytes; image will be fed from cached frames
                            pass
                    await self.conn_manager.update_data(user_id, params)
                    await self.conn_manager.send_json(user_id, {"status": "wait"})
                    if last_frame_time is None:
                        last_frame_time = time.time()
                    else:
                        # print(f"Frame time: {time.time() - last_frame_time}")
                        last_frame_time = time.time()

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")
                await self.conn_manager.disconnect(user_id, self.pipeline)

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})
        
        @self.app.get("/api/metrics/{user_id}")
        async def get_metrics(user_id: uuid.UUID, window_size: int = 100):
            """Get SLO metrics for a specific user"""
            if not self.enable_metrics:
                return JSONResponse({"error": "Metrics collection is not enabled"}, status_code=400)
            try:
                import numpy as np
                with self.user_metrics_lock:
                    if user_id not in self.user_latency_history or len(self.user_latency_history[user_id]) == 0:
                        return JSONResponse({"error": "No metrics data available"}, status_code=404)
                    
                    latencies = np.array(self.user_latency_history[user_id][-window_size:])
                    
                    stats = {
                        "mean_latency": float(np.mean(latencies)),
                        "median_latency": float(np.median(latencies)),
                        "p95_latency": float(np.percentile(latencies, 95)),
                        "p99_latency": float(np.percentile(latencies, 99)),
                        "min_latency": float(np.min(latencies)),
                        "max_latency": float(np.max(latencies)),
                        "std_latency": float(np.std(latencies)),
                        "sample_count": len(latencies),
                        "remaining_frames": len(self.user_input_timestamps.get(user_id, deque())),
                        "batch_count": self.user_batch_count.get(user_id, 0)
                    }
                    
                    return JSONResponse(stats)
            except Exception as e:
                logging.error(f"Error getting metrics: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:
                async def push_frames_to_pipeline():
                    last_params = SimpleNamespace()
                    sleep_time = 1 / 20  # Initial guess
                    # 16 FPS throttling for upload mode
                    TARGET_FPS = 24.0
                    min_frame_interval = 1.0 / TARGET_FPS
                    last_frame_sent_time = None
                    while True:
                        # Check if upload mode is enabled
                        video_status = self.conn_manager.get_video_queue_status(user_id)
                        is_upload_mode = video_status.get("is_upload_mode", False)
                        
                        if is_upload_mode:
                            # Upload mode: get next frame from video queue with 16 FPS throttling
                            current_time = time.time()
                            if last_frame_sent_time is not None:
                                time_since_last_frame = current_time - last_frame_sent_time
                                if time_since_last_frame < min_frame_interval:
                                    # Wait to maintain 16 FPS
                                    await asyncio.sleep(min_frame_interval - time_since_last_frame)
                            
                            video_frame = await self.conn_manager.get_next_video_frame(user_id)
                            if video_frame:
                                last_frame_sent_time = time.time()
                                # Create params object with video frame
                                params = SimpleNamespace()
                                params.image = bytes_to_pil(video_frame)
                                # Copy other parameters
                                if vars(last_params):
                                    for key, value in last_params.__dict__.items():
                                        if key != 'image' and key != '_frame_id':
                                            setattr(params, key, value)
                                
                                if params.__dict__ != last_params.__dict__:
                                    # Record input timestamp when frame is added to pipeline queue
                                    if self.enable_metrics:
                                        input_timestamp = time.time()
                                        with self.user_metrics_lock:
                                            if user_id not in self.user_input_timestamps:
                                                self.user_input_timestamps[user_id] = deque()
                                            self.user_input_timestamps[user_id].append(input_timestamp)
                                    
                                    last_params = params
                                    self.pipeline.accept_new_params(params)
                                    LOGGER.debug("Sent cached upload frame to pipeline for user %s", user_id)
                                # Yield control without delaying to maximize fluency
                                # await asyncio.sleep(sleep_time)
                            else:
                                # No frame available, wait a bit
                                await asyncio.sleep(sleep_time)
                        else:
                            # Camera mode: normal processing
                            params = await self.conn_manager.get_latest_data(user_id)
                            if params is None:
                                break
                            if vars(params) and params.__dict__ != last_params.__dict__:
                                last_params = params
                                # Record input timestamp when frame is added to pipeline queue
                                if self.enable_metrics:
                                    input_timestamp = time.time()
                                    with self.user_metrics_lock:
                                        if user_id not in self.user_input_timestamps:
                                            self.user_input_timestamps[user_id] = deque()
                                        self.user_input_timestamps[user_id].append(input_timestamp)
                                self.pipeline.accept_new_params(params)
                            await self.conn_manager.send_json(
                                user_id, {"status": "send_frame"}
                            )
                            # Yield control without delaying
                            # await asyncio.sleep(sleep_time)

                async def generate():
                    """
                    Event-driven MJPEG generator.

                    历史实现使用 EMA 估计帧间隔 + asyncio.sleep 进行节流，
                    在 produce_predictions 偶发慢一拍时会引入最坏 ~100 ms 的盲等
                    （MIN_FPS=10），显著推高 p99 与抖动。

                    现在改为纯事件驱动：`get_frame()` 内部已是 `await asyncio.Queue.get()`，
                    会精确在帧到达时唤醒。下游 MJPEG multipart 流的节奏天然由
                    DiT/VAE 产能 + WebSocket TCP 写入背压共同决定，无需应用层补 sleep。

                    丢弃 EMA / sleep_time / queue_size 轮询逻辑后：
                      • p50 节省 ~5–15 ms 的 sleep 残值；
                      • 极端慢帧时 p99 节省可达 ~100 ms（之前 MIN_FPS 兜底）。

                    `is_firefox` 历史在“非 Firefox”分支上额外多 yield 一次帧，初衷是
                    绕过 Chromium MJPEG 渲染器的双缓冲卡顿。这条会让非 Firefox 客户端
                    每帧下行带宽翻倍，并使解码端排队延迟翻倍——实测弊大于利，去除。
                    """
                    user_agent = request.headers.get("user-agent", "")
                    _ = is_firefox(user_agent)  # 保留兼容性钩子；当前不再做双 yield

                    last_frame_time = None
                    frame_time_list = []
                    while True:
                        try:
                            # 阻塞式拿帧——asyncio.Queue.get() 会在 produce_predictions
                            # put 后立即唤醒；上游若仍未生产则 await 让出事件循环，
                            # 不会忙轮询，也不会盲等。
                            frame = await self.conn_manager.get_frame(user_id)
                            if frame is None:
                                break

                            yield frame

                            now = time.time()
                            if last_frame_time is not None:
                                frame_time_list.append(now - last_frame_time)
                                if len(frame_time_list) > 100:
                                    frame_time_list.pop(0)
                            last_frame_time = now
                        except Exception as e:
                            LOGGER.error("Frame fetch error for user %s: %s", user_id, e)
                            break

                def produce_predictions(user_id, loop, stop_event):
                    while not stop_event.is_set():
                        images = self.pipeline.produce_outputs()
                        if len(images) == 0:
                            time.sleep(THROTTLE)
                            continue
                        
                        # Calculate latency for each output frame using FIFO timestamp queue
                        if self.enable_metrics:
                            output_timestamp = time.time()
                            batch_latencies = []
                            
                            with self.user_metrics_lock:
                                if user_id in self.user_input_timestamps:
                                    # For each output frame, get corresponding input timestamp (FIFO)
                                    for _ in range(len(images)):
                                        if len(self.user_input_timestamps[user_id]) > 0:
                                            input_timestamp = self.user_input_timestamps[user_id].popleft()
                                            latency = output_timestamp - input_timestamp
                                            batch_latencies.append(latency)
                                            
                                            # Add to history for statistics
                                            if user_id not in self.user_latency_history:
                                                self.user_latency_history[user_id] = []
                                            self.user_latency_history[user_id].append(latency)
                                    
                                    # Print batch statistics
                                    if len(batch_latencies) > 0:
                                        avg_latency = sum(batch_latencies) / len(batch_latencies)
                                        remaining_frames = len(self.user_input_timestamps[user_id])
                                        
                                        # Get batch count
                                        if user_id not in self.user_batch_count:
                                            self.user_batch_count[user_id] = 0
                                        self.user_batch_count[user_id] += 1
                                        batch_num = self.user_batch_count[user_id]
                                        
                                        # Prepare raw batch data
                                        raw_batch_data = {
                                            "batch_num": batch_num,
                                            "current_frames": len(batch_latencies),
                                            "avg_latency": avg_latency,
                                            "remaining": remaining_frames,
                                            "data_count": len(self.user_latency_history[user_id])
                                        }
                                        
                                        # Store raw data
                                        if user_id not in self.user_raw_data:
                                            self.user_raw_data[user_id] = []
                                        self.user_raw_data[user_id].append(raw_batch_data)
                                        
                                        LOGGER.info(
                                            "[Metrics] Batch %s/1000: current_frames=%s, avg_latency=%.4fs, remaining=%s, data_count=%s",
                                            batch_num,
                                            len(batch_latencies),
                                            avg_latency,
                                            remaining_frames,
                                            len(self.user_latency_history[user_id]),
                                        )
                                        
                                        # Log after 1000 batches
                                        if batch_num >= 1000:
                                            self._log_metrics_to_file(user_id)
                                            # Reset for next 1000 batches
                                            self.user_batch_count[user_id] = 0
                                            self.user_latency_history[user_id] = []
                                            self.user_raw_data[user_id] = []
                        
                        asyncio.run_coroutine_threadsafe(
                            self.conn_manager.put_frames_to_output_queue(
                                user_id,
                                list(map(pil_to_frame, images))
                            ),
                            loop
                        )

                await self._start_prediction_worker(
                    user_id,
                    produce_predictions,
                    asyncio.get_running_loop(),
                )
                asyncio.create_task(push_frames_to_pipeline())
                await self.conn_manager.send_json(user_id, {"status": "send_frame"})

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )

            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
                # Stop prediction thread on error
                await self._stop_prediction_worker(user_id)
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            info_schema = self.pipeline.Info.schema()
            info = self.pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = self.pipeline.InputParams.schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                }
            )

        os.makedirs(self.frontend_public_dir, exist_ok=True)

        self.app.mount(
            "/", StaticFiles(directory=self.frontend_public_dir, html=True), name="public"
        )

        # Add shutdown event handler
        @self.app.on_event("shutdown")
        async def shutdown_event():
            LOGGER.info("Shutdown event triggered, cleaning up...")
            await self.cleanup()

        # Start the 1Hz end-to-end latency sampler. Done at the end of
        # init_app() so every dependency (lock, history dict, log path)
        # is fully initialised. Daemon=True so it never blocks process exit
        # if cleanup is skipped (e.g. SIGKILL).
        if self.enable_metrics:
            self._latency_sampler_thread = threading.Thread(
                target=self._latency_sampler_loop,
                name="latency-sampler-1hz",
                daemon=True,
            )
            self._latency_sampler_thread.start()
            LOGGER.info(
                "[Metrics] 1Hz latency sampler started -> %s",
                self.latency_log_path,
            )

    def _latency_sampler_loop(self):
        """Once per second, emit a line to demo/latency.log summarising the
        end-to-end (input frame enqueue -> generated frame emit) latencies
        observed during the past ~1s window, per user.

        Implementation notes:
        - We track a per-user offset into self.user_latency_history. Each
          tick we read the *new tail* (history[offset:]) under the metrics
          lock as a Python slice (cheap copy of float refs), then release
          the lock before computing stats.
        - The downstream code in generate() periodically resets
          self.user_latency_history[user_id] = [] when a 1000-batch window
          completes. We detect that as len < offset and rebase to 0, so
          stats stay correct across resets.
        - One log line per active user per tick. If no user produced any
          new samples in the last second we still emit a heartbeat line
          ("no_new_samples") so the cadence is visible in the file.
        """
        import math
        log_path = self.latency_log_path
        # Open in line-buffered append mode so each row is flushed even if
        # the process is killed; survives demo restarts (we want history).
        try:
            log_fh = open(log_path, "a", buffering=1)
        except OSError as e:
            LOGGER.error("[Metrics] cannot open %s: %s", log_path, e)
            return
        try:
            log_fh.write(
                f"# latency sampler started at {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(unix={time.time():.3f}); columns: ts user count mean p50 p95 p99 min max remaining_input_q\n"
            )
        except OSError:
            pass

        def _percentile(sorted_vals, q):
            # Linear-interp percentile on a pre-sorted list. q in [0,100].
            if not sorted_vals:
                return float("nan")
            if len(sorted_vals) == 1:
                return sorted_vals[0]
            k = (len(sorted_vals) - 1) * (q / 100.0)
            lo = math.floor(k)
            hi = math.ceil(k)
            if lo == hi:
                return sorted_vals[int(k)]
            return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)

        next_tick = time.monotonic()
        while not self._latency_sampler_stop.is_set():
            next_tick += 1.0
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                # Use Event.wait so shutdown wakes us immediately.
                if self._latency_sampler_stop.wait(timeout=sleep_for):
                    break
            else:
                # We fell behind (GC pause etc.); resync to now to avoid a
                # burst of catch-up ticks.
                next_tick = time.monotonic()

            ts = time.time()
            # Snapshot phase: minimise time under the lock.
            snapshots = []  # list of (user_id, new_slice, remaining_input_q)
            try:
                with self.user_metrics_lock:
                    user_ids = list(self.user_latency_history.keys())
                    for uid in user_ids:
                        hist = self.user_latency_history.get(uid, [])
                        prev = self._latency_sampler_offsets.get(uid, 0)
                        cur_len = len(hist)
                        # Detect window reset (downstream truncates list).
                        if cur_len < prev:
                            prev = 0
                        if cur_len == prev:
                            new_slice = []
                        else:
                            # Slice copy of float refs is O(n) on n new
                            # samples; n is at most a few dozen per second
                            # at expected throughput, so this is cheap.
                            new_slice = hist[prev:cur_len]
                        self._latency_sampler_offsets[uid] = cur_len
                        remaining = len(self.user_input_timestamps.get(uid, deque()))
                        snapshots.append((uid, new_slice, remaining))
            except Exception as e:
                # Never let the sampler crash the server; just record it.
                LOGGER.warning("[Metrics] sampler snapshot error: %s", e)
                continue

            # Stats + write phase: outside the lock.
            try:
                if not snapshots:
                    log_fh.write(f"{ts:.3f} - no_active_users\n")
                    continue
                for uid, samples, remaining in snapshots:
                    if not samples:
                        log_fh.write(
                            f"{ts:.3f} {uid} count=0 no_new_samples remaining_input_q={remaining}\n"
                        )
                        continue
                    s = sorted(samples)
                    n = len(s)
                    mean = sum(s) / n
                    p50 = _percentile(s, 50)
                    p95 = _percentile(s, 95)
                    p99 = _percentile(s, 99)
                    log_fh.write(
                        f"{ts:.3f} {uid} count={n} "
                        f"mean={mean:.4f} p50={p50:.4f} p95={p95:.4f} "
                        f"p99={p99:.4f} min={s[0]:.4f} max={s[-1]:.4f} "
                        f"remaining_input_q={remaining}\n"
                    )
            except Exception as e:
                LOGGER.warning("[Metrics] sampler write error: %s", e)

        try:
            log_fh.write(
                f"# latency sampler stopped at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            log_fh.close()
        except OSError:
            pass

    def _log_metrics_to_file(self, user_id: uuid.UUID):
        """Log metrics to file after collecting 1000 batches"""
        try:
            import json
            import numpy as np
            
            # Get latency history
            if user_id not in self.user_latency_history or len(self.user_latency_history[user_id]) == 0:
                LOGGER.info("[Metrics] No latency data to log for user %s", user_id)
                return
            
            latencies = np.array(self.user_latency_history[user_id])
            
            # Calculate statistics
            stats = {
                "mean_latency": float(np.mean(latencies)),
                "median_latency": float(np.median(latencies)),
                "p50_latency": float(np.percentile(latencies, 50)),
                "p90_latency": float(np.percentile(latencies, 90)),
                "p95_latency": float(np.percentile(latencies, 95)),
                "p99_latency": float(np.percentile(latencies, 99)),
                "p99_9_latency": float(np.percentile(latencies, 99.9)),
                "min_latency": float(np.min(latencies)),
                "max_latency": float(np.max(latencies)),
                "std_latency": float(np.std(latencies)),
                "sample_count": len(latencies)
            }
            
            # Calculate deadline miss rate using target latency
            deadline = self.target_latency
            missed = np.sum(latencies > deadline)
            deadline_stats = {
                "deadline_seconds": deadline,
                "deadline_miss_rate": float(missed / len(latencies)) if len(latencies) > 0 else 0.0,
                "missed_frames": int(missed),
                "total_frames": len(latencies)
            }
            
            # Calculate jitter (variation in consecutive latencies)
            if len(latencies) > 1:
                jitter = np.abs(np.diff(latencies))
                jitter_stats = {
                    "mean_jitter": float(np.mean(jitter)),
                    "std_jitter": float(np.std(jitter)),
                    "max_jitter": float(np.max(jitter)),
                    "min_jitter": float(np.min(jitter)),
                    "p50_jitter": float(np.percentile(jitter, 50)),
                    "p90_jitter": float(np.percentile(jitter, 90)),
                    "p95_jitter": float(np.percentile(jitter, 95)),
                    "p99_jitter": float(np.percentile(jitter, 99)),
                    "p99.9_jitter": float(np.percentile(jitter, 99.9)),
                    "jitter_variance": float(np.var(jitter))
                }
            else:
                jitter_stats = {}
            
            # Create timestamp folder (YYYYMMDD_HHMM_step{step}_gpu{gpu_ids} format)
            timestamp = time.strftime("%Y%m%d_%H%M")
            # Format GPU IDs: replace commas with underscores for folder naming
            gpu_str = self.gpu_ids.replace(",", "_")
            folder_name = f"{timestamp}_step{self.step}_gpu{gpu_str}"
            session_dir = os.path.join(self.metrics_log_dir, folder_name)
            os.makedirs(session_dir, exist_ok=True)
            
            # Prepare raw data file content
            raw_data_content = {
                "user_id": str(user_id),
                "timestamp": timestamp,
                "target_latency": self.target_latency,
                "batches": self.user_raw_data.get(user_id, [])
            }
            
            # Prepare statistics file content
            statistics_content = {
                "user_id": str(user_id),
                "timestamp": timestamp,
                "target_latency": self.target_latency,
                "batch_count": 1000,
                "total_frames": len(latencies),
                "latency_stats": stats,
                "deadline_miss_rate": deadline_stats,
                "jitter_distribution": jitter_stats,
                "tail_latency": {
                    "p90_latency": stats["p90_latency"],
                    "p95_latency": stats["p95_latency"],
                    "p99_latency": stats["p99_latency"],
                    "p99_9_latency": stats["p99_9_latency"],
                    "max_latency": stats["max_latency"],
                    "mean_latency": stats["mean_latency"],
                    "median_latency": stats["median_latency"]
                }
            }
            
            # Write raw data file
            raw_data_filename = os.path.join(session_dir, f"raw_data_{user_id}.json")
            with open(raw_data_filename, 'w') as f:
                json.dump(raw_data_content, f, indent=2)
            
            # Write statistics file
            statistics_filename = os.path.join(session_dir, f"statistics_{user_id}.json")
            with open(statistics_filename, 'w') as f:
                json.dump(statistics_content, f, indent=2)
            
            LOGGER.info("[Metrics] Logged metrics to %s/", session_dir)
            LOGGER.info("[Metrics]   - Raw data: raw_data_%s.json", user_id)
            LOGGER.info("[Metrics]   - Statistics: statistics_%s.json", user_id)
            LOGGER.info(
                "[Metrics] Summary: mean=%.4fs, p95=%.4fs, miss_rate=%.2f%%",
                stats["mean_latency"],
                stats["p95_latency"],
                deadline_stats["deadline_miss_rate"] * 100,
            )
            
        except Exception as e:
            logging.error(f"Error logging metrics to file: {e}")
    
    async def cleanup(self):
        """Clean up all resources on shutdown"""
        LOGGER.info("Starting cleanup process...")
        
        # Set shutdown event
        self.shutdown_event.set()

        # Stop the 1Hz latency sampler thread (if running). Done early so
        # any further latency stats print only the existing data and we
        # don't keep an open file handle once user state is being torn
        # down.
        if getattr(self, "_latency_sampler_thread", None) is not None:
            self._latency_sampler_stop.set()
            self._latency_sampler_thread.join(timeout=2.0)
            if self._latency_sampler_thread.is_alive():
                LOGGER.warning("[Metrics] latency sampler did not stop in 2s")
            else:
                LOGGER.info("[Metrics] latency sampler stopped")

        # Stop all background tasks
        for user_id in list(self.prediction_workers):
            await self._stop_prediction_worker(user_id)
        LOGGER.info("Stopped prediction tasks")
        
        # Close all WebSocket connections and pipeline
        LOGGER.info("Closing %s active connections...", len(self.conn_manager.active_connections))
        try:
            await self.conn_manager.disconnect_all(self.pipeline)
        except Exception as e:
            LOGGER.error("Error during disconnect_all: %s", e)
        
        LOGGER.info("Cleanup completed")

    async def _start_prediction_worker(self, user_id, produce_predictions, loop):
        await self._stop_prediction_worker(user_id)
        stop_event = threading.Event()
        task = asyncio.create_task(
            asyncio.to_thread(produce_predictions, user_id, loop, stop_event)
        )
        self.prediction_workers[user_id] = (stop_event, task)

    async def _stop_prediction_worker(self, user_id):
        worker = self.prediction_workers.pop(user_id, None)
        if worker is None:
            return

        stop_event, task = worker
        stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# Global app instance for signal handler
app_instance = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    LOGGER.info("Received signal %s, shutting down gracefully...", signum)
    if app_instance:
        # Trigger cleanup in a separate thread to avoid blocking
        import threading
        def trigger_cleanup():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(app_instance.cleanup())
                loop.close()
            except Exception as e:
                LOGGER.error("Error during cleanup: %s", e)
        
        cleanup_thread = threading.Thread(target=trigger_cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        cleanup_thread.join(timeout=5)  # Wait up to 5 seconds for cleanup
    
    sys.exit(0)

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    mp.set_start_method("spawn", force=True)

    config.pretty_print()
    if config.num_gpus > 1:
        from vid2vid_pipe import MultiGPUPipeline
        pipeline = MultiGPUPipeline(config)
    else:
        from vid2vid import Pipeline
        pipeline = Pipeline(config)

    app_obj = App(config, pipeline)
    app = app_obj.app
    app_instance = app_obj  # Set global reference for signal handler

    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            reload=False,
            ssl_certfile=config.ssl_certfile,
            ssl_keyfile=config.ssl_keyfile,
        )
    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, shutting down...")
        # Trigger cleanup
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(app_obj.cleanup())
            loop.close()
        except Exception as e:
            LOGGER.error("Error during cleanup: %s", e)
        sys.exit(0)
    except Exception as e:
        LOGGER.error("Fatal error: %s", e)
        sys.exit(1)
