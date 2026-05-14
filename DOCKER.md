# 用 Docker 跨 GPU 机器复用 StreamDiffusionV2 环境

本仓库提供完整的 Docker 化方案，目标镜像：

```
mirrors.tencent.com/gputest/streamdiffusionv2:<tag>
mirrors.tencent.com/gputest/streamdiffusionv2:latest
```

镜像里包含：

- CUDA 12.4 + cuDNN runtime（与 `torch==2.6.0+cu124` 匹配）
- Python 3.10 + 全部 `requirements.lock.txt` 依赖
- 仓库源码（不含权重 / 日志 / 输出）
- **预构建好的** demo 前端静态产物（`demo/frontend/public`）
- `ffmpeg`、`huggingface_hub[cli,hf_transfer]`

镜像里**故意不含**：

- `ckpts/`、`wan_models/`（几十 GB，运行时挂载）
- `outputs/`、`logs/`、`demo/slo_metrics/`（运行时产物）
- `flash-attn`（默认关闭，`STREAMDIFF_DISABLE_FLASH=1`；按需用 build-arg 启用）

---

## 1. 构建并推送（在一台装了 docker 的机器上做一次）

### 1.1 准备工作

```bash
# 装 docker（如果还没装）
curl -fsSL https://get.docker.com | sh

# 装 nvidia-container-toolkit（用于 docker 跑 GPU；构建阶段不需要 GPU）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 登录腾讯云镜像仓库（一次即可，凭据存到 ~/.docker/config.json）
docker login mirrors.tencent.com
```

### 1.2 构建并推送

```bash
cd /data/StreamDiffusionV2

# 默认：tag 用 git short sha，并同步打 :latest，自动推送
./docker-build-push.sh

# 指定 tag
./docker-build-push.sh v0512

# 也启用 flash-attn（编译耗时 +20~30 分钟，镜像会大几百 MB）
BUILD_FLASH_ATTN=1 ./docker-build-push.sh v0512-fa

# 只构建不推送，先本地测试
PUSH=0 ./docker-build-push.sh v0512
```

构建完成后镜像就长这样：

```
mirrors.tencent.com/gputest/streamdiffusionv2:v0512
mirrors.tencent.com/gputest/streamdiffusionv2:latest
```

> 预计大小：~8–10 GB（torch + cuDNN + transformers/diffusers 占大头）。开启 flash-attn 再多 ~500 MB。

---

## 2. 在新 GPU 机器上拉取并运行

### 2.1 拉镜像

```bash
docker login mirrors.tencent.com   # 仅首次
docker pull mirrors.tencent.com/gputest/streamdiffusionv2:latest
```

### 2.2 准备宿主机上的权重目录

权重可以**只下载一次**，放到一个固定的宿主目录里（例如 NFS 上），多台机器共用：

```bash
mkdir -p /data/sdv2_assets/{ckpts,wan_models,outputs}
```

### 2.3 用 `docker-run.sh` 启动

把 `docker-run.sh` 拷到目标机器（或直接 git clone 仓库拿）：

```bash
# 1) 第一次：下权重到挂载目录（在容器里跑，避免再装一次 hf cli）
HOST_CKPTS=/data/sdv2_assets/ckpts \
HOST_WAN=/data/sdv2_assets/wan_models \
HOST_OUTPUTS=/data/sdv2_assets/outputs \
./docker-run.sh download

# 2) 启动 web demo（默认 7860 端口）
HOST_CKPTS=/data/sdv2_assets/ckpts \
HOST_WAN=/data/sdv2_assets/wan_models \
HOST_OUTPUTS=/data/sdv2_assets/outputs \
GPU_IDS=0,1 PORT=7860 \
./docker-run.sh demo

# 3) 离线单卡推理
HOST_CKPTS=/data/sdv2_assets/ckpts \
HOST_WAN=/data/sdv2_assets/wan_models \
HOST_OUTPUTS=/data/sdv2_assets/outputs \
GPU_IDS=0 \
./docker-run.sh single

# 4) 进容器调试
./docker-run.sh shell
```

### 2.4 不用 `docker-run.sh`，纯 docker 命令

```bash
docker run --gpus all -it --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 7860:7860 \
  -v /data/sdv2_assets/ckpts:/workspace/StreamDiffusionV2/ckpts \
  -v /data/sdv2_assets/wan_models:/workspace/StreamDiffusionV2/wan_models \
  -v /data/sdv2_assets/outputs:/workspace/StreamDiffusionV2/outputs \
  -e GPU_IDS=0,1 \
  mirrors.tencent.com/gputest/streamdiffusionv2:latest \
  bash -lc "cd demo && ./start.sh"
```

---

## 3. 关于驱动版本的提醒

镜像基于 **CUDA 12.4**，`torch==2.6.0+cu124` 需要宿主机 NVIDIA 驱动 **≥ 550.54.14**。

当前 build 机器驱动为 `535.247.01`（最高 CUDA 12.2），如果**目标 GPU 机器的驱动也是 535.x**，会在 `import torch` 时报：

```
CUDA error: forward compatibility was attempted on non supported HW
```

两种解决办法（任选其一）：

### 方案 A：升级目标机器驱动到 550+（推荐）

最干净，对未来兼容性也最好。

### 方案 B：把镜像降级到 CUDA 12.1

修改 `Dockerfile`：

```diff
- FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS runtime
+ FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS runtime
```

并把 torch 安装行换成 cu121：

```diff
- RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
-         torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
+ RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
+         torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

> torch 2.6.0 同时发布了 cu121 和 cu124 wheel；cu121 兼容 535.x 驱动。

---

## 4. 常见问题

### 4.1 镜像太大，pull 太慢？

- 把权重彻底从镜像里剥离（已经做了，挂载即可）。
- 如果还想再瘦，可以把 base 换成 `12.4.1-cudnn-runtime-ubuntu22.04`（runtime 版而非 devel），但那样就**不能再打开** `BUILD_FLASH_ATTN=1`，因为 flash-attn 编译需要 nvcc。
- 多机内网 pull 时考虑就近的 mirror 或 docker registry mirror。

### 4.2 demo 启动后前端 404？

镜像里已经预先 `npm run build` 过，且默认 `SKIP_FRONTEND_BUILD=1`。如果你修改了前端代码，需要重新构建镜像；不要在容器里重新跑 `npm install`（容器里没装 node）。

### 4.3 想在容器里改代码热调试

```bash
docker run --gpus all -it --rm \
  -v /data/StreamDiffusionV2:/workspace/StreamDiffusionV2 \
  -v /data/sdv2_assets/ckpts:/workspace/StreamDiffusionV2/ckpts \
  -v /data/sdv2_assets/wan_models:/workspace/StreamDiffusionV2/wan_models \
  mirrors.tencent.com/gputest/streamdiffusionv2:latest \
  /bin/bash
```

把整个仓库挂进去就能改代码即时生效（包是 `pip install -e .` 安装的）。

### 4.4 想启 flash-attn

构建时加 `BUILD_FLASH_ATTN=1`，运行时 `STREAMDIFF_DISABLE_FLASH=0`：

```bash
BUILD_FLASH_ATTN=1 ./docker-build-push.sh v0512-fa

docker run --gpus all -it --rm \
  -e STREAMDIFF_DISABLE_FLASH=0 \
  ...
  mirrors.tencent.com/gputest/streamdiffusionv2:v0512-fa
```

---

## 5. 文件清单

| 文件 | 作用 |
|---|---|
| `Dockerfile` | 多阶段镜像定义（前端构建 + Python runtime） |
| `.dockerignore` | 排除 ckpts / 日志 / venv / node_modules，避免污染镜像 |
| `docker-build-push.sh` | 一键构建并推送到 `mirrors.tencent.com/gputest/streamdiffusionv2` |
| `docker-run.sh` | 一键在目标机器拉起 shell / demo / 离线推理 / 下权重 |
| `DOCKER.md` | 本文件 |
