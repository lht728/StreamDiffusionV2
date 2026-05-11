import sys
import os
import logging
import queue
import threading
import time
import traceback
from datetime import datetime
from multiprocessing import Queue, Manager, Event, Process
from typing import Literal

DEMO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEMO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util import (
    array_to_image,
    clear_queue,
    dump_pydantic_model,
    image_to_array,
    read_images_from_queue,
    resolve_worker_device,
    select_stream_execution_mode,
)

import torch

from pydantic import BaseModel, Field
from PIL import Image
from typing import List
from streamv2v.inference import SingleGPUInferencePipeline as StreamBatchInferencePipeline
from streamv2v.inference_wo_batch import SingleGPUInferencePipeline as StreamNoBatchInferencePipeline
from streamv2v.inference_common import merge_cli_config

LOGGER = logging.getLogger(__name__)
STARTUP_TIMEOUT_SECONDS = 180.0

default_prompt = "Cyberpunk-inspired figure, neon-lit hair highlights, augmented cybernetic facial features, glowing interface holograms floating around, futuristic cityscape reflected in eyes, vibrant neon color palette, cinematic sci-fi style"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusionV2</h1>
<p class="text-sm">
    This demo showcases
    <a
    href="https://streamdiffusionv2.github.io/"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusionV2
</a>
video-to-video pipeline with a MJPEG stream server.
</p>
"""


def set_config_value(config, key: str, value) -> None:
    if isinstance(config, dict):
        config[key] = value
        return
    setattr(config, key, value)


def sync_pydantic_field_default(model_cls, field_name: str, value) -> None:
    if hasattr(model_cls, "model_fields") and field_name in model_cls.model_fields:
        model_cls.model_fields[field_name].default = value
    if hasattr(model_cls, "__fields__") and field_name in model_cls.__fields__:
        model_cls.__fields__[field_name].default = value


def build_single_gpu_pipeline_manager(args, device: torch.device):
    mode_info = select_stream_execution_mode(args, device)
    pipeline_cls = (
        StreamBatchInferencePipeline
        if mode_info["mode"] == "stream_batch"
        else StreamNoBatchInferencePipeline
    )
    pipeline_manager = pipeline_cls(args, device)
    pipeline_manager.load_model(args.checkpoint_folder)
    pipeline_manager.logger.info(
        "Online single-GPU worker selected mode=%s, use_taehv=%s, use_tensorrt=%s",
        mode_info["mode"],
        bool(getattr(args, "use_taehv", False)),
        bool(getattr(args, "use_tensorrt", False)),
    )
    return pipeline_manager, mode_info

class Pipeline:
    class Info(BaseModel):
        name: str = "StreamV2V"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        
        prompt: str = Field(
            default_prompt,
            title="Update your prompt here",
            field="textarea",
            id="prompt",
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        restart: bool = Field(
            default=False,
            title="Restart",
            description="Restart the streaming",
        )
        input_mode: Literal["camera", "upload"] = Field(
            default="camera",
            title="Input Mode",
            hide=True,
            id="input_mode",
        )
        upload_mode: bool = Field(
            default=False,
            title="Upload Mode",
            hide=True,
            id="upload_mode",
        )
        use_taehv: bool = Field(
            default=False,
            title="Use TAEHV VAE",
            description="Use the lightweight TAEHV decoder for online inference",
            field="checkbox",
            hide=True,
            id="use_taehv",
        )
        use_tensorrt: bool = Field(
            default=False,
            title="Use TensorRT",
            description="Enable available TensorRT acceleration paths for online inference",
            field="checkbox",
            hide=True,
            id="use_tensorrt",
        )

    def __init__(self, args):
        torch.set_grad_enabled(False)

        config = merge_cli_config(args.config_path, args._asdict())
        sync_pydantic_field_default(self.InputParams, "use_taehv", bool(getattr(config, "use_taehv", False)))
        sync_pydantic_field_default(self.InputParams, "use_tensorrt", bool(getattr(config, "use_tensorrt", False)))
        params = self.InputParams()
        config["height"] = params.height
        config["width"] = params.width

        self.prompt = params.prompt
        self.args = config
        self.prepare()

    def prepare(self):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.prepare_event = Event()
        self.stop_event = Event()
        self.restart_event = Event()
        self.error_queue = Queue()
        self.runtime_state = Manager().dict()
        self.runtime_state["prompt"] = self.prompt
        self.runtime_state["use_taehv"] = bool(getattr(self.args, "use_taehv", False))
        self.runtime_state["use_tensorrt"] = bool(getattr(self.args, "use_tensorrt", False))

        # Watchdog state
        self._watchdog_thread = None
        self._respawn_lock = threading.Lock()
        self._respawn_count = 0
        self._max_respawn = int(os.environ.get("STREAMV2V_MAX_RESPAWN", "10"))

        self._spawn_process(initial=True)
        self._start_watchdog()

    def _spawn_process(self, initial: bool = False):
        """Create + start the generate_process worker. Reuses existing queues/events
        so the main HTTP layer can keep running across restarts."""
        # Ensure events are in a clean state for the new worker
        self.prepare_event.clear()
        self.restart_event.clear()

        self.process = Process(
            target=generate_process,
            args=(
                self.args,
                self.runtime_state,
                self.prepare_event,
                self.restart_event,
                self.stop_event,
                self.input_queue,
                self.output_queue,
                self.error_queue,
            ),
            daemon=True
        )
        self.process.start()
        self.processes = [self.process]

        try:
            wait_for_processes_ready(
                processes=self.processes,
                ready_events=[self.prepare_event],
                error_queue=self.error_queue,
            )
        except Exception:
            if initial:
                # First-time startup failure: bubble up so launcher fails fast.
                raise
            LOGGER.error(
                "Worker respawn failed to become ready:\n%s",
                traceback.format_exc(),
            )
            raise

    def _start_watchdog(self):
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="vid2vid-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self):
        """Background monitor: respawns the worker if it dies unexpectedly."""
        while not self.stop_event.is_set():
            time.sleep(1.0)
            proc = getattr(self, "process", None)
            if proc is None:
                continue
            if proc.is_alive():
                continue
            if self.stop_event.is_set():
                return
            # Drain any pending error report from the dead worker
            try:
                while True:
                    worker_name, error_message = self.error_queue.get_nowait()
                    LOGGER.error(
                        "Worker '%s' reported error before exit:\n%s",
                        worker_name,
                        error_message,
                    )
            except queue.Empty:
                pass

            with self._respawn_lock:
                if self.stop_event.is_set():
                    return
                if self._respawn_count >= self._max_respawn:
                    LOGGER.error(
                        "Worker died and respawn limit (%d) reached; giving up.",
                        self._max_respawn,
                    )
                    return
                self._respawn_count += 1
                LOGGER.warning(
                    "Worker process exited (exitcode=%s). Respawning (%d/%d)...",
                    proc.exitcode,
                    self._respawn_count,
                    self._max_respawn,
                )
                # Clear stale data so the new worker starts on fresh frames
                clear_queue(self.input_queue)
                clear_queue(self.output_queue)
                try:
                    self._spawn_process(initial=False)
                    LOGGER.info("Worker respawned successfully.")
                except Exception:
                    LOGGER.error(
                        "Failed to respawn worker:\n%s", traceback.format_exc()
                    )
                    # Back off a bit before next attempt
                    time.sleep(2.0)

    def accept_new_params(self, params: "Pipeline.InputParams"):
        if hasattr(params, "image"):
            image_array = image_to_array(params.image, self.args.width, self.args.height)
            self.input_queue.put(image_array)

        if hasattr(params, "prompt") and params.prompt and self.prompt != params.prompt:
            self.prompt = params.prompt
            self.runtime_state["prompt"] = self.prompt

        if hasattr(params, "use_taehv"):
            requested_use_taehv = bool(params.use_taehv)
            if requested_use_taehv != bool(self.runtime_state.get("use_taehv", False)):
                self.runtime_state["use_taehv"] = requested_use_taehv
                self.restart_event.set()
                clear_queue(self.output_queue)

        if hasattr(params, "use_tensorrt"):
            requested_use_tensorrt = bool(params.use_tensorrt)
            if requested_use_tensorrt != bool(self.runtime_state.get("use_tensorrt", False)):
                self.runtime_state["use_tensorrt"] = requested_use_tensorrt
                self.restart_event.set()
                clear_queue(self.output_queue)

        if hasattr(params, "restart") and params.restart:
            self.restart_event.set()
            clear_queue(self.output_queue)

    @staticmethod
    def params_to_namespace(params: "Pipeline.InputParams"):
        from types import SimpleNamespace

        return SimpleNamespace(**dump_pydantic_model(params))

    def produce_outputs(self) -> List[Image.Image]:
        qsize = self.output_queue.qsize()
        results = []
        for _ in range(qsize):
            results.append(array_to_image(self.output_queue.get()))
        return results

    def close(self):
        LOGGER.info("Setting stop event for the single-GPU demo worker")
        self.stop_event.set()

        LOGGER.info("Waiting for demo worker shutdown")
        # Always operate on the most recently spawned process as well as any tracked.
        procs = list(getattr(self, "processes", []))
        if getattr(self, "process", None) is not None and self.process not in procs:
            procs.append(self.process)
        for i, process in enumerate(procs):
            try:
                process.join(timeout=1.0)
                if process.is_alive():
                    LOGGER.warning("Process %s did not terminate gracefully; terminating", i)
                    process.terminate()
                    process.join(timeout=0.5)
                    if process.is_alive():
                        LOGGER.error("Force killing process %s", i)
                        process.kill()
            except Exception:
                LOGGER.exception("Error while shutting down process %s", i)

        wd = getattr(self, "_watchdog_thread", None)
        if wd is not None and wd.is_alive():
            wd.join(timeout=2.0)

        LOGGER.info("Pipeline closed successfully")


def _maybe_raise_worker_error(error_queue):
    try:
        worker_name, error_message = error_queue.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"{worker_name} failed during startup:\n{error_message}")


def wait_for_processes_ready(processes, ready_events, error_queue, timeout_seconds: float = STARTUP_TIMEOUT_SECONDS):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        _maybe_raise_worker_error(error_queue)
        if all(event.is_set() for event in ready_events):
            return
        dead_processes = [process.pid for process in processes if not process.is_alive()]
        if dead_processes:
            raise RuntimeError(f"Demo worker processes exited before becoming ready: {dead_processes}")
        time.sleep(0.1)

    _maybe_raise_worker_error(error_queue)
    raise TimeoutError(f"Timed out waiting for demo workers to become ready after {timeout_seconds:.0f}s")


def report_worker_error(error_queue, worker_name: str) -> None:
    error_queue.put((worker_name, traceback.format_exc()))


def _setup_delay_logger():
    """Per-frame inference delay logger.

    Writes one line per output frame to /data/StreamDiffusionV2/log/delay.log
    in CSV-ish format:
        <iso_ts>,<phase>,<chunk_idx>,<frame_in_chunk>,<global_frame_idx>,<chunk_input_frames>,<chunk_wall_ms>,<frame_delay_ms>,<fps>
    Phase is one of: init / stream.
    Returns a stdlib Logger that writes only to the delay file (no propagation).
    """
    delay_log_path = "/data/StreamDiffusionV2/log/delay.log"
    try:
        os.makedirs(os.path.dirname(delay_log_path), exist_ok=True)
    except Exception:
        pass
    logger = logging.getLogger("vid2vid.delay")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Avoid stacking handlers on respawn
    if not any(getattr(h, "_v2v_delay_handler", False) for h in logger.handlers):
        try:
            handler = logging.FileHandler(delay_log_path, mode="a", encoding="utf-8")
            handler._v2v_delay_handler = True  # type: ignore[attr-defined]
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            # Header (only if file is empty / new run)
            try:
                if os.path.getsize(delay_log_path) == 0:
                    logger.info(
                        "iso_ts,phase,chunk_idx,frame_in_chunk,global_frame_idx,"
                        "chunk_input_frames,chunk_wall_ms,frame_delay_ms,fps"
                    )
            except OSError:
                pass
        except Exception:
            # If we can't open the file, fall back to a no-op logger so the
            # worker keeps running.
            pass
    return logger


def generate_process(args, runtime_state, prepare_event, restart_event, stop_event, input_queue, output_queue, error_queue):
    torch.set_grad_enabled(False)
    try:
        device = resolve_worker_device(args.gpu_ids, rank=0)
        torch.cuda.set_device(device)

        delay_logger = _setup_delay_logger()
        chunk_idx = 0
        global_frame_idx = 0

        current_use_taehv = bool(runtime_state.get("use_taehv", getattr(args, "use_taehv", False)))
        current_use_tensorrt = bool(runtime_state.get("use_tensorrt", getattr(args, "use_tensorrt", False)))
        set_config_value(args, "use_taehv", current_use_taehv)
        set_config_value(args, "use_tensorrt", current_use_tensorrt)
        pipeline_manager, _ = build_single_gpu_pipeline_manager(args, device)
        chunk_size = pipeline_manager.base_chunk_size * args.num_frame_per_block
        first_batch_num_frames = 1 + chunk_size
        is_running = False
        prompt = runtime_state["prompt"]
        session = None

        prepare_event.set()

        while not stop_event.is_set():
            requested_use_taehv = bool(runtime_state.get("use_taehv", current_use_taehv))
            requested_use_tensorrt = bool(runtime_state.get("use_tensorrt", current_use_tensorrt))
            if requested_use_taehv != current_use_taehv or requested_use_tensorrt != current_use_tensorrt:
                pipeline_manager.logger.info(
                    "Rebuilding online single-GPU worker for use_taehv=%s, use_tensorrt=%s",
                    requested_use_taehv,
                    requested_use_tensorrt,
                )
                current_use_taehv = requested_use_taehv
                current_use_tensorrt = requested_use_tensorrt
                set_config_value(args, "use_taehv", current_use_taehv)
                set_config_value(args, "use_tensorrt", current_use_tensorrt)
                clear_queue(input_queue)
                clear_queue(output_queue)
                del pipeline_manager
                torch.cuda.empty_cache()
                pipeline_manager, _ = build_single_gpu_pipeline_manager(args, device)
                chunk_size = pipeline_manager.base_chunk_size * args.num_frame_per_block
                first_batch_num_frames = 1 + chunk_size
                prompt = runtime_state["prompt"]
                session = None
                is_running = False
                restart_event.clear()
                continue

            # Prepare first batch
            if not is_running or runtime_state["prompt"] != prompt or restart_event.is_set():
                prompt = runtime_state["prompt"]
                if restart_event.is_set():
                    clear_queue(input_queue)
                    restart_event.clear()
                images = read_images_from_queue(input_queue, first_batch_num_frames, device, stop_event)

                _t_init0 = time.time()
                session, initial_video = pipeline_manager.start_stream_session(
                    prompt=prompt,
                    images=images,
                    noise_scale=args.noise_scale,
                )
                _init_wall_ms = (time.time() - _t_init0) * 1000.0
                _init_frames_total = sum(len(v) for v in initial_video) if initial_video is not None else 0
                _init_fps = (_init_frames_total / (_init_wall_ms / 1000.0)) if _init_wall_ms > 0 and _init_frames_total > 0 else 0.0
                _frame_in_chunk = 0
                for image in initial_video:
                    output_queue.put(image)
                    try:
                        delay_logger.info(
                            f"{datetime.utcnow().isoformat()},init,{chunk_idx},"
                            f"{_frame_in_chunk},{global_frame_idx},"
                            f"{first_batch_num_frames},{_init_wall_ms:.2f},"
                            f"{_init_wall_ms:.2f},{_init_fps:.2f}"
                        )
                    except Exception:
                        pass
                    _frame_in_chunk += 1
                    global_frame_idx += 1
                chunk_idx += 1
                is_running = True

            images = read_images_from_queue(input_queue, chunk_size, device, stop_event)
            if images is None:
                break

            _t_chunk0 = time.time()
            _frame_in_chunk = 0
            for decoded_video in pipeline_manager.run_stream_batch(session, images):
                for image in decoded_video:
                    _now = time.time()
                    _frame_delay_ms = (_now - _t_chunk0) * 1000.0
                    output_queue.put(image)
                    try:
                        # chunk_wall_ms is the same as frame_delay_ms for the *last* frame of the chunk;
                        # for intermediate frames it represents how long since chunk start.
                        delay_logger.info(
                            f"{datetime.utcnow().isoformat()},stream,{chunk_idx},"
                            f"{_frame_in_chunk},{global_frame_idx},"
                            f"{chunk_size},{_frame_delay_ms:.2f},"
                            f"{_frame_delay_ms:.2f},"
                            f"{(global_frame_idx + 1) / max(_now - _t_chunk0, 1e-6):.2f}"
                        )
                    except Exception:
                        pass
                    _frame_in_chunk += 1
                    global_frame_idx += 1
            chunk_idx += 1
    except Exception:
        report_worker_error(error_queue, "single_gpu_demo_worker")
        raise
