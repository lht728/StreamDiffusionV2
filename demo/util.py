"""Small helpers shared by the demo pipelines."""

from PIL import Image
import io
import logging
import os
import time
import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JPEG encoder selection
#
# Hot path: produce_predictions 每个 chunk 把 4 个 numpy uint8 帧编码为 MJPEG 段。
# PIL 默认实现是单线程 libjpeg，约 3 ms/帧 @ 512×512 ⇒ 12 ms/chunk，会阻塞
# 输出生产线程。
#
# 优先级：
#   1) PyTurboJPEG（libjpeg-turbo C 绑定，~6× PIL）—— 推荐生产环境安装。
#   2) Pillow-SIMD（drop-in 替换 PIL，~2–4× PIL）—— 已通过 import PIL 自动生效。
#   3) PIL（兜底，保证可运行）。
#
# 当 numpy 帧从 GPU 直出 uint8 时（A.4 优化），路径是
# `np.uint8 (H,W,3)` → `encode_jpeg_bytes` → bytes，无需 PIL 中转。
# ---------------------------------------------------------------------------

_JPEG_QUALITY = int(os.environ.get("DEMO_JPEG_QUALITY", "85"))

try:
    from turbojpeg import TurboJPEG, TJPF_RGB, TJSAMP_420  # type: ignore

    _TURBO_JPEG = TurboJPEG()
    _USE_TURBOJPEG = True
    LOGGER.info("MJPEG encoder: PyTurboJPEG (quality=%s)", _JPEG_QUALITY)
except Exception as _exc:  # noqa: BLE001 - turbojpeg is optional
    _TURBO_JPEG = None
    _USE_TURBOJPEG = False
    LOGGER.info(
        "MJPEG encoder: PIL fallback (PyTurboJPEG not available: %s; quality=%s). "
        "Install `PyTurboJPEG` (and libjpeg-turbo) for ~6x faster encoding.",
        type(_exc).__name__,
        _JPEG_QUALITY,
    )

BF16_BYTES = 2
KV_HEAD_DIM = 128
STREAM_BATCH_HEADROOM_BYTES = 1024**3
STREAM_BATCH_SAFETY_FACTOR = 1.15
MODEL_LAYOUTS = {
    "T2V-1.3B": {"num_transformer_blocks": 30, "num_heads": 12},
    "T2V-14B": {"num_transformer_blocks": 40, "num_heads": 40},
}

_MJPEG_HEADER_PREFIX = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
_MJPEG_HEADER_SUFFIX = b"\r\n\r\n"
_MJPEG_TRAILER = b"\r\n"


def _wrap_mjpeg(jpeg_bytes: bytes) -> bytes:
    """Wrap a JPEG payload in an MJPEG multipart-form chunk."""
    return (
        _MJPEG_HEADER_PREFIX
        + str(len(jpeg_bytes)).encode("ascii")
        + _MJPEG_HEADER_SUFFIX
        + jpeg_bytes
        + _MJPEG_TRAILER
    )


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image


def pil_to_frame(image: Image.Image) -> bytes:
    """Encode a PIL image into a single MJPEG multipart frame.

    Uses PyTurboJPEG when available (significantly faster than PIL's libjpeg),
    otherwise falls back to ``Image.save(format="JPEG")``.
    """
    if _USE_TURBOJPEG:
        # TurboJPEG accepts ndarray; ensure RGB uint8 contiguous.
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
        jpeg_bytes = _TURBO_JPEG.encode(
            arr,
            quality=_JPEG_QUALITY,
            pixel_format=TJPF_RGB,
            jpeg_subsample=TJSAMP_420,
        )
    else:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        jpeg_bytes = buf.getvalue()
    return _wrap_mjpeg(jpeg_bytes)


def ndarray_uint8_to_frame(arr: np.ndarray) -> bytes:
    """Encode a HxWx3 uint8 numpy array directly into an MJPEG frame.

    Fast path used when the inference pipeline produces uint8 frames on the
    GPU and copies them straight to numpy (see A.4 optimization), avoiding
    an intermediate PIL Image construction.
    """
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Defensive: fall back to PIL for unusual layouts.
        return pil_to_frame(Image.fromarray(arr))

    if _USE_TURBOJPEG:
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        jpeg_bytes = _TURBO_JPEG.encode(
            arr,
            quality=_JPEG_QUALITY,
            pixel_format=TJPF_RGB,
            jpeg_subsample=TJSAMP_420,
        )
    else:
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=_JPEG_QUALITY)
        jpeg_bytes = buf.getvalue()
    return _wrap_mjpeg(jpeg_bytes)


def is_firefox(user_agent: str) -> bool:
    return "Firefox" in user_agent


# Polling cadence used while waiting for the input queue to fill enough frames
# for one DiT chunk. Tuned for low end-to-end latency:
#   • 10 ms (legacy) wastes up to ~10 ms after the last needed frame arrives,
#     contributing measurable jitter on chunk-head frames.
#   • 3 ms keeps CPU usage trivial (~0.03 core) but cuts that wake-up jitter
#     by ~7 ms on average, with similar bound improvements on p99.
_QUEUE_POLL_INTERVAL_S = float(os.environ.get("DEMO_QUEUE_POLL_INTERVAL_S", "0.003"))


def read_images_from_queue(queue, num_frames_needed, device, stop_event=None):
    # Wait until we have enough frames. Tight poll interval keeps wake-up
    # latency low while remaining cheap on CPU; see _QUEUE_POLL_INTERVAL_S.
    while queue.qsize() < num_frames_needed:
        if stop_event and stop_event.is_set():
            return None
        time.sleep(_QUEUE_POLL_INTERVAL_S)

    # Read exactly num_frames_needed frames in order (FIFO), don't discard any frames.
    images = []
    for _ in range(num_frames_needed):
        images.append(queue.get())

    # Stack images in order (FIFO)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).unsqueeze(0)
    images = images.permute(0, 4, 1, 2, 3).to(dtype=torch.bfloat16).to(device=device)
    return images


def clear_queue(queue):
    while queue.qsize() > 0:
        queue.get()


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def dump_pydantic_model(model) -> dict:
    """Serialize a Pydantic model across v1/v2 without deprecation warnings."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def parse_gpu_ids(gpu_ids: str) -> list[int]:
    return [int(gpu_id.strip()) for gpu_id in gpu_ids.split(",") if gpu_id.strip()]


def resolve_worker_device(gpu_ids: str, rank: int) -> torch.device:
    """
    Resolve the correct CUDA device for a worker rank.

    When `CUDA_VISIBLE_DEVICES` is set, torch renumbers the visible devices to
    `cuda:0..N-1`. The demo still accepts physical GPU IDs, so this helper maps
    them back to the correct local index.
    """
    requested_ids = parse_gpu_ids(gpu_ids)
    target_id = requested_ids[rank]
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    if not visible_env:
        return torch.device(f"cuda:{target_id}")

    visible_ids = parse_gpu_ids(visible_env)
    if target_id in visible_ids:
        return torch.device(f"cuda:{visible_ids.index(target_id)}")

    if 0 <= target_id < len(visible_ids):
        return torch.device(f"cuda:{target_id}")

    LOGGER.warning(
        "GPU id %s is not present in CUDA_VISIBLE_DEVICES=%s; falling back to local rank %s",
        target_id,
        visible_env,
        rank,
    )
    return torch.device(f"cuda:{rank}")


def compute_stream_token_shapes(width: int, height: int) -> dict[str, int]:
    """Compute the VAE-latent and DiT-token grids for the current video size."""
    latent_width = width // 8
    latent_height = height // 8
    token_width = width // 16
    token_height = height // 16
    return {
        "latent_width": latent_width,
        "latent_height": latent_height,
        "token_width": token_width,
        "token_height": token_height,
        "token_count": token_width * token_height,
    }


def get_model_layout(config) -> dict[str, int]:
    model_type = _config_value(config, "model_type", "T2V-1.3B")
    return MODEL_LAYOUTS.get(model_type, MODEL_LAYOUTS["T2V-1.3B"])


def get_num_transformer_blocks(config) -> int:
    return int(get_model_layout(config)["num_transformer_blocks"])


def infer_stream_dimensions(config) -> tuple[int, int]:
    """Infer pixel-space width/height from explicit config or latent-shape metadata."""
    width = _config_value(config, "width")
    height = _config_value(config, "height")
    if width is not None and height is not None:
        return int(width), int(height)

    image_or_video_shape = _config_value(config, "image_or_video_shape")
    if image_or_video_shape and len(image_or_video_shape) >= 5:
        latent_height = int(image_or_video_shape[-2])
        latent_width = int(image_or_video_shape[-1])
        return latent_width * 8, latent_height * 8

    raise ValueError("Unable to infer demo stream width/height from config")


def estimate_stream_batch_extra_memory_bytes(config, width: int, height: int) -> int:
    """
    Estimate the extra CUDA memory required by stream-batch over no-batch mode.

    The main delta comes from repeating the KV cache across denoising steps in
    `prepare(..., batch_denoise=True)`, plus the batched hidden-state buffer.
    """
    non_terminal_steps = [
        int(step)
        for step in _config_value(config, "denoising_step_list", [])
        if int(step) != 0
    ]
    num_steps = len(non_terminal_steps)
    if num_steps <= 1:
        return 0

    model_layout = get_model_layout(config)
    shapes = compute_stream_token_shapes(width, height)

    num_frame_per_block = int(_config_value(config, "num_frame_per_block", 1))
    num_kv_cache = int(_config_value(config, "num_kv_cache", 6))
    kv_cache_length = shapes["token_count"] * num_kv_cache

    kv_bytes_per_step = (
        model_layout["num_transformer_blocks"]
        * kv_cache_length
        * model_layout["num_heads"]
        * KV_HEAD_DIM
        * 2  # K and V
        * BF16_BYTES
    )
    hidden_state_bytes = (
        num_steps
        * num_frame_per_block
        * 16
        * shapes["latent_height"]
        * shapes["latent_width"]
        * BF16_BYTES
    )
    return (num_steps - 1) * kv_bytes_per_step + hidden_state_bytes


def select_stream_execution_mode(config, device: torch.device) -> dict[str, object]:
    """
    Choose between batched and no-batch online inference based on free CUDA memory.
    """
    width, height = infer_stream_dimensions(config)
    shapes = compute_stream_token_shapes(width=width, height=height)
    estimate_bytes = estimate_stream_batch_extra_memory_bytes(
        config,
        width=width,
        height=height,
    )
    required_bytes = int(estimate_bytes * STREAM_BATCH_SAFETY_FACTOR) + STREAM_BATCH_HEADROOM_BYTES

    if device.type != "cuda":
        return {
            "mode": "stream_batch",
            "free_bytes": None,
            "required_bytes": required_bytes,
            "estimated_extra_bytes": estimate_bytes,
            "shapes": shapes,
        }

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    use_stream_batch = free_bytes >= required_bytes
    mode = "stream_batch" if use_stream_batch else "stream_wo_batch"

    LOGGER.info(
        "Online inference mode=%s, free=%.2f GiB, required=%.2f GiB, estimated_extra=%.2f GiB, latent=%sx%s, tokens=%sx%s",
        mode,
        free_bytes / 1024**3,
        required_bytes / 1024**3,
        estimate_bytes / 1024**3,
        shapes["latent_width"],
        shapes["latent_height"],
        shapes["token_width"],
        shapes["token_height"],
    )
    return {
        "mode": mode,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "required_bytes": required_bytes,
        "estimated_extra_bytes": estimate_bytes,
        "shapes": shapes,
    }


def image_to_array(
        image: Image.Image,
        width: int,
        height: int,
        normalize: bool = True
    ) -> np.ndarray:
        image = image.convert("RGB").resize((width, height))
        image_array = np.array(image)
        if normalize:
            image_array = image_array / 127.5 - 1.0
        return image_array


def array_to_image(image_array: np.ndarray, normalize: bool = True) -> Image.Image:
    """Convert a HxWxC numpy array to a PIL Image.

    The ``normalize`` flag is **best-effort**: if the array is already
    ``uint8`` it is treated as ready-to-display pixels regardless of the
    flag, avoiding a redundant CPU-side ``* 255`` after the inference
    pipeline started returning uint8 directly (see the GPU-side fuse in
    ``streamv2v/inference{,_pipe,_wo_batch}.py:_decode_video_array``).
    """
    if image_array.dtype == np.uint8:
        # Fast path: pipeline already produced quantized [0,255] uint8 frames
        # on the GPU — no host-side multiplication needed.
        return Image.fromarray(image_array, mode="RGB" if image_array.ndim == 3 and image_array.shape[-1] == 3 else None)

    if normalize:
        image_array = image_array * 255.0
    image_array = image_array.astype(np.uint8)
    return Image.fromarray(image_array)
