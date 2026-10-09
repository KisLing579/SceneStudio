# generate_shots.py — LTX-2.5 分镜视频生成

## 关键场景前后端工作台

新增 `scene_studio.py` 和 `scene_studio.html`，整合关键帧标记、场景提示词融合、即梦重抽、LTX 视频生成和勾选拼接。

先安装 Miniconda 或 Anaconda。在 SceneStudio 项目根目录中，安装依赖前先创建并激活独立环境：

```bash
conda create -n scenestudio python=3.11 pip -y
conda activate scenestudio
```

环境只需创建一次；每次打开新终端运行工作台命令前，执行 `conda activate scenestudio`。该环境用于 SceneStudio 工作台，视频生成仍需单独配置 `LTX_WORKDIR` 指向的 LTX 运行环境。

在已激活的环境中安装依赖并启动：

```bash
python -m pip install -r requirements-studio.txt
python scene_studio.py --project-root /root/project --port 7200
```

打开 `http://127.0.0.1:7200`，输入 project_id，读取 `/root/project/<project_id>/meta.json`。
输入方式可选择「Project ID」或「绝对路径」。绝对路径支持服务器上的项目目录（例如 `/root/project/proj2`）
或该项目的 `meta.json` 文件（例如 `/root/project/proj2/meta.json`），也支持 Windows 服务端的盘符绝对路径。
加载后页面显示实际项目目录，后续重抽、生成、保存和拼接均使用该目录；同一目录通过两种方式加载时共享任务状态。
后续所有项目数据操作均以该项目目录为基础：重抽图像写回项目目录，提示词写回该目录的 meta.json，
视频和日志写入该目录下的 key_scene_out/、key_scene_work/。原图读取与视频首帧读取使用同一套路径解析，
迁移前的绝对图像路径会回退到当前项目下的同名文件，不读取其它项目的图像。
`LTX_WORKDIR` 仅用于定位模型运行环境，不改变上述项目数据路径。
本机示例：`python scene_studio.py --project-root ./work --port 7200`，项目 ID 填 `proj2`。

Mindawaker 的 LTX-Scene-Studio 入口通过 `/?project_path=<URL编码后的绝对目录>` 自动加载当前章节。
链接同时传入 `locale=en` 或 `locale=zh`，工作台自动使用对应界面语言；未指定时使用中文。
例如 `/?locale=en&project_path=<URL编码后的绝对目录>`。项目内容、提示词和原始任务日志不做翻译。
先启动本服务，默认链接使用与 Mindawaker 前端相同的主机名及 7200 端口。
如地址不同，可在 Mindawaker 前端环境变量中设置 `NEXT_PUBLIC_SCENE_STUDIO_URL` 后重启前端。
两项服务需能访问同一个项目目录；链接不会复制或上传项目文件。

- 加载项目后展示真正缩小到最大 192×128 的图像缩略图。
- 「创建关键场景帧」依次调用 `generate_key_frame_flags.py`、`merge_frame_scene.py`，选取现有关键帧图像并生成融合提示词。再次创建会重新计算融合提示词。
- 每组上方编辑图像提示词并「重抽」，下方编辑场景提示词；「保存提示词」或任何操作按钮都会保存当前编辑内容到 meta.json。
- `max-frames`、`frames-delta` 默认分别为 480、3；delta 的单位沿用原脚本，为 8 帧。
- 「批量生成视频段」按顺序生成全部组，覆盖已有视频段；完成一段显示一段。首次生成前，各组按钮显示「开始生成」且可用，无需先批量生成。
- 单组生成或批量生成任务提交成功后，对应按钮立即改为「重新生成」；执行期间暂时禁用，失败后可重试。服务运行期间刷新页面也保留已提交状态；服务重启后按已有视频判断。
- 「开始生成」和「重新生成」传入 `--scene`、`--key-frame`、`--max-frames`、`--frames-delta` 和 `--force`，只更新当前组。
- 右上角复选框控制拼接范围；按项目中的场景顺序拼接，并沿用 `key_scene_editor.py` 的淡入淡出处理。结果显示在页面底部。
- 视频保存在项目的 `key_scene_out/`，日志和中间帧保存在 `key_scene_work/`；拼接结果为 `key_scene_out/concat_fade.mp4`。
- 同一个项目的任务串行执行，运行期间禁用编辑和操作按钮；页面轮询显示进度与错误日志。服务重启后不恢复运行中的任务，可重新提交。

环境配置：融合对白需要 `DEEPSEEK_API_KEY`，角色文件使用项目内的 `artifacts/story/character_specs.json`。
如仅需离线融合视觉提示词，可添加 `--offline`（不抽取对白）。
图像重抽需要 `VOLC_ACCESS_KEY` / `VOLC_SECRET_KEY`，或 `IMAGE_MODEL_PASSWORD=AccessKey**SecretKey`。
视频生成需要已有 LTX 模型、`uv`、`LTX_WORKDIR`；视频拼接需要 `ffmpeg` / `ffprobe`。
服务默认仅监听本机，没有账号认证；适合本地或受信任的内网使用。

本次仅完成代码检查，实际图像服务、GPU 视频生成与浏览器流程由使用者测试。

按 `meta.json` 逐分镜生成 LTX-2.5 视频（image-to-video，首帧续接），支持断点续跑、
单分镜切段、按场景/分镜选择性生成，并可选地把配音合成进最终视频。

## 前置条件

- Python 3.10+，`uv`（执行 `uv run`）
- `ffmpeg` / `ffprobe`（在 PATH 中，或通过环境变量 `FFMPEG` / `FFPROBE` 指定）
- LTX 工作目录：含 `pyproject.toml` 与 `ltx_pipelines` 包，通过 `LTX_WORKDIR` 指定命令执行目录。
- 模型权重根目录：修改 `LTX-test/studio_config.json` 的 `model_directory`，不从环境变量读取。
  支持绝对路径（Windows 建议写成 `E:/models/ltx-2.5`）或相对配置文件目录的路径。
  该目录下应包含 `diffusion_models`、`text_encoders`、`vae`、`latent_upscale_models`。
  scene_studio 和两个视频生成脚本共用此配置，修改后建议重启 scene_studio。

## 快速开始

```bash
# 预览计划（不触发 GPU）
python generate_shots.py --dry-run

# 正式生成全部分镜（默认保留原生音轨）
python generate_shots.py

# 生成并用 audios 配音替换原生音轨
python generate_shots.py --tts

# 只生成某个场景 / 某个分镜
python generate_shots.py --scene 3
python generate_shots.py --scene 3 --shot 2
```

## 模块化脚本（分步执行）

除一键脚本 `generate_shots.py` 外，另提供 4 个独立脚本，可单独执行某一阶段（共用一个
`ltx_common.py`，模型路径、帧数换算、抽帧、拼接等逻辑与 `generate_shots.py` 完全一致）：

| 脚本 | 功能 | 用法 |
| --- | --- | --- |
| `generate_clips.py` | 生成 clips 视频段（不含拼接与配音） | `python generate_clips.py [meta.json] [--dry-run] [--force] [--scene N] [--shot M] [--prompt_with_text]` |
| `concat_clips.py` | 按 meta.json 顺序拼接 clips 为结果视频（纯视频） | `python concat_clips.py [meta.json] [--scene N] [--shot M] [--output 输出]` |
| `concat_audio.py` | 按顺序拼接 audios 为一段 AAC 配音 | `python concat_audio.py [meta.json] [--scene N] [--shot M] [--output 输出]` |
| `add_audio.py` | 把配音 mux 进视频 | `python add_audio.py <视频> <音频> [--output 输出]` |

典型流程：

```bash
python generate_clips.py --dry-run        # 先预览计划
python generate_clips.py                  # 1. 生成分镜视频段

python concat_clips.py                    # 2. 拼接结果视频（纯视频 result.mp4）

python concat_audio.py                    # 3. 拼接配音（可选 -> work/result_audio.m4a）
python add_audio.py result.mp4 work/result_audio.m4a   # 4. 配音合成进视频（可选）
```

- 各脚本的 `--scene` / `--shot` 筛选逻辑与 `generate_shots.py` 一致。
- `add_audio.py` 默认输出 `<视频名>_with_audio.mp4`（如 `result_with_audio.mp4`），
  不会覆盖纯视频结果；需要覆盖时用 `--output result.mp4` 显式指定。

## 参数

| 参数 | 说明 |
| --- | --- |
| `meta`（位置参数） | `meta.json` 路径，默认取脚本同目录下的 `meta.json` |
| `--dry-run` | 只打印每个分镜的参数与命令，不真正执行 |
| `--force` | 强制重新生成已存在的输出（默认跳过已完成的视频段，便于续跑） |
| `--scene N` | 只生成第 N 个场景的所有分镜（编号与 `shot_id` 的 `sceneN` 一致，如 `--scene 3`） |
| `--shot M` | 只生成指定场景内的第 M 个分镜，需与 `--scene` 配合（如 `--scene 3 --shot 2`） |
| `--tts` | 用 `audios` 配音替换原生音轨（默认保留 LTX 原生音轨；与 `--prompt_with_text` 互斥） |
| `--prompt_with_text` | 把每个分镜对应的 `lines` 中文文本翻译成英文，追加到该分镜 `prompt` 末尾（默认关闭；需 `DEEPSEEK_API_KEY`） |
| `-h, --help` | 查看帮助 |

> `--scene` / `--shot` 兼容 `3` 与 `scene3` 两种写法（内部提取数字）。

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `LTX_WORKDIR` | 执行 `uv run` 的工作目录（含 `pyproject.toml` 与 `ltx_pipelines` 包）；默认取脚本所在目录 |
| `FFMPEG` / `FFPROBE` | 可选，指定 `ffmpeg` / `ffprobe` 可执行文件路径 |
| `DEEPSEEK_API_KEY` | 可选，`--prompt_with_text` 时用于中→英翻译（DeepSeek） |

## 工作原理

1. 依次读取 `meta.json` 中 `task_data` 下的 `prompts` / `images` / `durations` / `audios`
   （列表长度动态读取，`images`、`audios` 须与 `prompts` 长度一致）。
2. 每个分镜四个参数：

   | 参数 | 来源 |
   | --- | --- |
   | `PROMPT` | `image_prompt` |
   | `IDX` | `shot_idx` |
   | `FRAMES` | `8 * round(duration * 3) + 1`（24fps，帧数满足 LTX 的 8n+1） |
   | `IMAGE` | 场景内第一个分镜用 `images[i]`；同场景后续分镜用上一分镜视频的最后一帧 |

3. GPU 单次最多生成 121 帧；换算出的帧数超过 121 时，切成多段“首尾相接”生成
   （上一段最后一帧作为下一段起始图），再用 ffmpeg 合并为完整分镜视频。
4. 逐条执行 `uv run python -m ltx_pipelines.distilled`，等待结束再继续。
5. 全部分镜生成后，按顺序拼接为 `result.mp4`，默认保留 LTX 原生音轨；加 `--tts` 时，
   把 `audios` 按顺序拼接成 AAC 并合成进 `result.mp4`，替换原生音轨。

## 输出目录结构

```
LTX-test/
├── clips/                     # 分镜视频段（不存在则自动创建）
│   ├── output_0.mp4           # 分镜最终视频（output_{shot_idx}.mp4）
│   ├── output_1_seg0.mp4      # 切段中间视频（多段分镜才有）
│   └── ...
├── work/                      # 中间产物（不存在则自动创建）
│   ├── output_0_last.png      # 末帧续接图
│   ├── logs/                  # 每个视频段的生成日志
│   └── result_video.mp4       # 纯视频结果（合成前）
└── result.mp4                 # 最终结果视频（--tts 时含配音）
```

## 断点续跑

- 每个视频段生成前会检查 `clips/` 中是否已有对应文件：已存在且非空、且未加 `--force` 则跳过。
- 中断后直接重跑即可：已完成的段会跳过，只补缺失的段；多段分镜会重新合并，末帧续接图缺失时自动补抽。
- 想完全重跑某个场景/分镜，用 `--force` 覆盖。
- 注意：跳过判断只看文件名、不看 `prompt`。切换 `--prompt_with_text`（或改动 prompt/seed）后，已存在的视频段不会自动重生成，需加 `--force`。

## 说明与注意

- 帧率固定 24fps，`--num-frames` 恒为 8n+1；单段上限 121 帧。
- `--seed` 固定为 42；`--image` 后的强度参数为 `0 0.8`（可在脚本顶部常量处修改）。
- 模型路径由 `studio_config.json` 指定，与 `LTX_WORKDIR` 和素材项目目录无关。
- 分镜多段合并与结果拼接均采用 `-c copy`（无损拷贝视频流 + 音轨，保留 LTX 生成的原声）；
  加 `--tts` 时改用 `audios` 配音替换原生音轨。配音按顺序直接拼接，未做逐分镜的帧级对齐，
  多段累计后尾部可能有轻微错位。

## 图像 Prompt 编辑与重抽（网页）

`prompt_editor.py` 提供一个本地网页，用于编辑 `meta.json` 每个分镜的图像 prompt，并用火山引擎
（即梦）重抽图像、把结果写回 `meta.json`（重抽逻辑参考 `Mindawaker/app/image_engine/volc_engine.py`）。

```bash
python prompt_editor.py [--host 127.0.0.1] [--port 8000]
```

浏览器打开 `http://127.0.0.1:8000`，输入 `meta.json` 的绝对路径点「加载」：

- 展示 `images` 指向的所有图像，每张图旁是一个可编辑的 prompt 输入框（初始值为该分镜的 `image_prompt`）；
- 「保存 prompt」：把该分镜修改后的 prompt 写回 `meta.json`（不重抽）；
- 「重抽图像」：按输入框里的 prompt 用即梦重新生成图像，并把新图像路径与 prompt 一并写回；
- 「保存全部 prompt」：一次性保存所有输入框的修改。

写回时会同步更新 `images[i]`、`prompts[i].image_prompt`、`images_configs[i].prompt`（若存在）。
重抽使用 `VOLC_REQ_KEY`（默认 `jimeng_t2i_v40`）与 `images_configs[i].size`（默认 `1024*1024`），
随机 seed；需环境变量 `VOLC_ACCESS_KEY` / `VOLC_SECRET_KEY`（或 `IMAGE_MODEL_PASSWORD=AccessKey**SecretKey`）。

> 重抽后的图像与 prompt 已更新进 `meta.json`，但下游视频仍需重新运行 `generate_shots.py` /
> `generate_clips.py`（建议加 `--force`）才会按新首图与新 prompt 重新生成。

## 视频教程
观看这个分步演示教程，了解如何使用 Novel Engine 创建项目、生成章节并导出故事；使用 Mindawaker 将小说转换为多媒体素材；最后通过 Scene Studio + LTX 2.5 将这些素材制作成完整的视频。

[▶ 观看视频](https://youtu.be/Cjfw1-kFnxc)

