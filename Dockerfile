# syntax=docker/dockerfile:1.6
#
# StreamDiffusionV2 portable image.
#
# Stage 1 (frontend-builder): build the Svelte demo frontend (Node 18).
# Stage 2 (runtime):          CUDA 12.4 + Python 3.10 + project deps + code.
#
# Build:
#   docker build -t mirrors.tencent.com/gputest/streamdiffusionv2:<tag> .
#
# Run (mount checkpoints + outputs from host so the image stays slim):
#   docker run --gpus all -it --rm \
#     -p 7860:7860 \
#     -v /data/ckpts:/workspace/StreamDiffusionV2/ckpts \
#     -v /data/wan_models:/workspace/StreamDiffusionV2/wan_models \
#     -v /data/outputs:/workspace/StreamDiffusionV2/outputs \
#     mirrors.tencent.com/gputest/streamdiffusionv2:<tag>
#
# Notes:
# - torch==2.6.0+cu124 is pinned in pyproject.toml. The base image uses CUDA
#   12.4 to match the wheel's runtime exactly.
# - Host driver must be >= 550 for CUDA 12.4 user-mode runtime. If your driver
#   is 535.x (CUDA 12.2 max), switch the base tag to 12.1.1 and reinstall torch
#   from the cu121 index (see DOCKER.md).
# - flash-attn is OPTIONAL. It is compiled from source and adds ~20-30 minutes
#   to the build. It is gated by the BUILD_FLASH_ATTN build-arg (default 0)
#   because run_v2v.sh and demo/run.sh both default to STREAMDIFF_DISABLE_FLASH=1.

############################
# Stage 1: frontend builder
############################
FROM node:18-bookworm-slim AS frontend-builder

WORKDIR /build/frontend

# Install only what is needed to resolve the lockfile first, so this layer
# caches across code changes that don't touch package.json/package-lock.json.
COPY demo/frontend/package.json demo/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

# Now bring in the rest of the frontend sources and build.
COPY demo/frontend/ ./
RUN npm run build


############################
# Stage 2: runtime
############################
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_FLASH_ATTN=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # Default to SDPA fallback (matches run_v2v.sh / demo/run.sh defaults).
    STREAMDIFF_DISABLE_FLASH=1 \
    # Frontend is already built into demo/frontend/public during image build,
    # so demo/run.sh and demo/start.sh must NOT rerun npm install/build.
    SKIP_FRONTEND_BUILD=1

# System deps:
# - python3.10 + venv: project requires >=3.10,<3.13 and pins torch 2.6.0
# - ffmpeg / libsndfile1: imageio/av decode mp4
# - git, curl, ca-certificates: hf cli + downloader scripts
# - build-essential, ninja-build: needed if BUILD_FLASH_ATTN=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates curl git \
        ffmpeg libsndfile1 \
        build-essential ninja-build \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip toolchain once.
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/StreamDiffusionV2

# ---- Python deps layer (cached separately from source) ----
# Copy only files that affect dependency resolution to maximise cache reuse.
COPY pyproject.toml requirements.lock.txt README.md LICENSE ./

# Install torch first (CUDA 12.4 wheels) so the rest of the resolve doesn't
# accidentally pull a CPU-only build from a fallback index.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Install the rest of the runtime deps from the lockfile (excludes torch line
# which was already satisfied above; pip will skip already-satisfied entries).
RUN pip install -r requirements.lock.txt

# huggingface CLI for _download_ckpts.sh inside the container.
RUN pip install "huggingface_hub[cli,hf_transfer]"

# Optional: build flash-attn from source. Disabled by default; turn on with
#   docker build --build-arg BUILD_FLASH_ATTN=1 ...
RUN if [ "$BUILD_FLASH_ATTN" = "1" ]; then \
        pip install packaging && \
        MAX_JOBS=4 pip install flash_attn==2.7.4.post1 --no-build-isolation ; \
    fi

# ---- Source code layer ----
# Copy the rest of the repo. Anything matched by .dockerignore is skipped
# (ckpts, wan_models, logs, outputs, .venv, __pycache__, build, *.egg-info...).
COPY . .

# Drop the pre-built frontend artefacts from stage 1 over the (intentionally
# empty due to .dockerignore) demo/frontend tree.
COPY --from=frontend-builder /build/frontend/public ./demo/frontend/public

# Install the package itself in editable mode so `python -m streamv2v.*` and
# the `streamdiffusionv2-*` entrypoints work without re-resolving deps.
RUN pip install --no-deps -e .

# Make helper scripts executable (COPY preserves bits, but this is belt-and-
# suspenders for hosts that strip them, e.g. some CI checkouts on Windows).
RUN chmod +x run_v2v.sh demo/run.sh demo/start.sh _download_ckpts.sh || true

EXPOSE 7860

# Default command: drop into bash. Override with `docker run ... <cmd>` to
# launch run_v2v.sh / demo/run.sh / etc. See DOCKER.md for ready-made recipes.
CMD ["/bin/bash"]
