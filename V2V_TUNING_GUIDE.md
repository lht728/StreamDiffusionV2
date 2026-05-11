# StreamDiffusionV2 离线 V2V 调参指南

针对 `run_v2v.sh` + `streamv2v.inference` 的离线视频到视频 (V2V) 推理参数说明，覆盖所有可调项及调参决策树。

> 默认调用入口：`./run_v2v.sh <mode> [extra args...]`，通过环境变量传 I/O，通过 extra args 传模型/采样参数。

---

## 一、改写强度（决定"像不像目标"）—— 最关键

| 参数 | 默认 | 取值范围 | 作用 | 调高的效果 | 调低的效果 |
|---|---|---|---|---|---|
| `--noise_scale` | 0.8 | 0.0–1.1 | **每个 chunk 注入的噪声强度**。值越大越"忘掉"输入视频的结构 | 形态改写更彻底（人能变成猫）；但会丢失原姿态/构图 | 高度保留原视频结构；改写无力（还是动物/人形） |
| `--fixed_noise_scale` | off | flag | 固定 `noise_scale`（默认会按 chunk 衰减）。**与 `--noise_scale ≥ 0.9` 配合是把"改写"踩到底的开关** | 全程强改写，每个 chunk 都重写 | （off 时）后续 chunk 改写力度自然下降，时序更稳 |
| `--step` | 2 | 1–4（实际） | 去噪步数。每步 = 一次结构修正机会 | 结构改写更彻底、细节更精；但耗时线性增加 | 更快；但可能保留太多原结构 |
| `--t2v` | off | flag | **完全忽略输入视频**，只按 prompt 文生视频（仍生成相同帧数） | 100% 听 prompt，结构最自由 | （off 时）走 V2V，受输入约束 |
| `--seed` | 0 | 任意整数 | 初始噪声随机种子 | 同 prompt + 不同 seed = 不同变体；遇到糟糕结果先换 seed 再说 | — |

**经验组合**：

- 形态接近（人 → 人，动物 → 动物）：`--step 2 --noise_scale 0.7`，不加 fixed
- 跨类别（人 → 猫，动物 → 小女孩）：`--step 3 --noise_scale 1.0 --fixed_noise_scale`
- 极端跨类别还不行：`--step 3 --noise_scale 1.05 --fixed_noise_scale`，或直接 `--t2v`
- prompt 也要同步加强（`NOT human, NOT animal` 这类负面对抗描述放前面）

---

## 二、Prompt（不是 CLI 参数，但本质是参数）

| 项 | 作用 |
|---|---|
| `prompt.txt` 内容 | 决定目标外观。**关键词靠前 + 明确否定原类别**（如 `human girl, NOT animal`）效果最强 |
| 关键词强度 | 形容词堆叠（`big round eyes, long twin tails, pink hoodie...`）会显著提升特征出现概率 |

prompt 文件路径通过 `PROMPT_FILE_PATH` 环境变量传入。

**模板示例**（人 → 小猫）：

```
A cute fluffy cartoon kitten, NOT human, NOT person,
big round eyes, small pink nose, soft fur, anime style,
replacing the original human characters, same poses and motions,
clean background
```

---

## 三、时间/分辨率（决定基础质量与速度）

| 参数 | 默认 | 作用 | 备注 |
|---|---|---|---|
| `--height` / `--width` | 480 / 832 | 推理分辨率 | **必须 8 的倍数**；模型按 480p 训练，提到 720p 不一定更好且显存翻倍 |
| `--fps` | 16 | 输出帧率 | 16 是模型基准；提到 24 会做时序插值，可能模糊 |
| `--target_fps` | None | 重采样输入的帧率 | 不设 = 用原视频；设了会改变实际推理帧数 |
| `--num_frames` | 81 | 单次推理 chunk 长度 | 一般不动；增大 → 显存涨、时序更连贯；减小 → 显存省、可能 chunk 接缝 |

---

## 四、模型/权重（换底层能力）

| 参数 | 默认 | 作用 |
|---|---|---|
| `--config_path` | `configs/wan_causal_dmd_v2v.yaml` | 模型配置。换 `wan_causal_dmd_v2v_14b.yaml` 用 14B 模型，**结构改写能力强很多**，但需先下权重 |
| `--checkpoint_folder` | `ckpts/wan_causal_dmd_v2v` | 权重目录，必须和 config 对应 |
| `--model_type` | `T2V-1.3B` | 模型类型标识 |

通过环境变量 `CONFIG_PATH` / `CHECKPOINT_FOLDER` 传入。

---

## 五、加速/资源（不影响画面，只影响速度和显存）

| 参数 | 默认 | 作用 |
|---|---|---|
| `--use_taehv` | off | 用轻量 TAEHV VAE 编解码，**加速 ~2x，画质几乎无损**，强烈建议开 |
| `--use_tensorrt` | off | TensorRT 加速路径，需环境支持（首次会编译，慢；之后很快） |
| `--fast` | off | 等价于 `--use_taehv --use_tensorrt` |
| `--profile` | off | 同步打印吞吐日志，调参看耗时用 |
| `--gpu_id` | None | 指定 GPU。也可用 `CUDA_VISIBLE_DEVICES=N` 实现等效效果 |

---

## 六、运行模式（`run_v2v.sh` 第一个参数）

| 模式 | 作用 |
|---|---|
| `single` | 单卡推理（`inference.py`，支持 batch/chunk pipeline，常用） |
| `single_wo_batch` | 单卡无 batch 版（`inference_wo_batch.py`），更省显存但慢 |
| `multi` | 多卡 `torchrun`，吞吐高（用 `NPROC_PER_NODE` 控制卡数，`CUDA_VISIBLE_DEVICES=6,7` 选卡） |

---

## 七、I/O（环境变量，`run_v2v.sh` 读取）

| 变量 | 作用 |
|---|---|
| `VIDEO_PATH` | 输入视频路径 |
| `PROMPT_FILE_PATH` | prompt 文本路径 |
| `OUTPUT_FOLDER` | 输出目录（生成 `output_000.mp4`） |
| `MASTER_PORT` | 多卡模式 torchrun 端口（多任务并跑要错开） |
| `CUDA_VISIBLE_DEVICES` | 选 GPU |
| `HEIGHT` / `WIDTH` / `FPS` / `STEP` | 同名 CLI 参数的环境变量入口 |

---

## 八、调参决策树（效果不理想时按顺序试）

1. **完全没改写、还是原物**：`--noise_scale` 0.8 → **1.0** + 加 `--fixed_noise_scale`；prompt 加 `NOT <原类别>`
2. **改写了但不像目标**：`--step` 2 → **3**；prompt 把目标特征前置并堆叠形容词
3. **还是不行**：`--noise_scale 1.05`，或 `--seed` 换 3-5 个值挑最好
4. **依然不行**：`--t2v` 完全无视输入（会丢动作/构图）
5. **跨类别太大（人 ↔ 动物）**：换 **14B 模型**（`configs/wan_causal_dmd_v2v_14b.yaml`），改写能力质变
6. **风格过头/失真**：反向回调，`--step 2`、`--noise_scale 0.85`、去掉 `--fixed_noise_scale`

---

## 九、参考调用命令

### 标准试跑（8 秒片段，单卡）

```bash
cd /data/StreamDiffusionV2 && source venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=2 \
  VIDEO_PATH=outputs/<task>/clip_8s.mp4 \
  PROMPT_FILE_PATH=outputs/<task>/prompt.txt \
  OUTPUT_FOLDER=outputs/<task>/trial \
  HEIGHT=480 WIDTH=832 FPS=16 STEP=3 \
  setsid bash -c 'exec ./run_v2v.sh single \
      --use_taehv --noise_scale 1.0 --fixed_noise_scale --seed 42 \
      > /tmp/v2v_<task>_trial.log 2>&1' < /dev/null & disown
```

### 强改写（极端跨类别）

```bash
./run_v2v.sh single --use_taehv --step 3 \
  --noise_scale 1.05 --fixed_noise_scale --seed 42
```

### 完全文生（忽略输入视频结构）

```bash
./run_v2v.sh single --use_taehv --step 3 --t2v --seed 42
```

### 多卡加速（2 卡）

```bash
CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 \
  ./run_v2v.sh multi --use_taehv --step 3 --noise_scale 1.0 --fixed_noise_scale
```

### 监控/控制

```bash
tail -f /tmp/v2v_<task>_trial.log     # 看日志
pgrep -af streamv2v.inference          # 看进程
nvidia-smi -i 2                        # 看 GPU
pkill -f 'streamv2v.inference'         # 终止
```
