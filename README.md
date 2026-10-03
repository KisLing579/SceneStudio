# SceneStudio

[中文说明](README_zh.md)

SceneStudio is a project-based workbench for turning storyboard images into LTX-2.5 videos. It brings key-frame grouping, image and scene-prompt editing, Jimeng image redraw, LTX clip generation, and selected-clip assembly into one browser interface.

The workbench runs as a Flask service with a standalone HTML frontend. It can open Mindawaker projects directly or load a compatible project directory containing `meta.json`.

## Features

- Load a project by ID, absolute directory, or absolute `meta.json` path, including Windows server paths.
- Browse generated image thumbnails, resized to fit within 192 x 128 pixels.
- Create key-scene groups from existing frames and merge their visual prompts, with optional DeepSeek dialogue extraction.
- Edit image prompts and scene-video prompts, save them to the project, and redraw selected images with Volcengine / Jimeng.
- Generate all key-scene clips sequentially, then regenerate individual groups as needed.
- Preview completed clips as generation progresses and select clips for a montage with fades.
- Use English or Chinese UI text through the `locale` URL parameter.
- Inspect progress and errors while project-level locks prevent overlapping work on the same project.

## Workflow

```text
Project meta.json + images + optional character specifications
  -> key-frame flags -> merged key-scene prompts
  -> prompt editing / optional image redraw
  -> LTX image-to-video clips
  -> select clips -> concatenate with fades -> preview
```

Creating key-scene frames selects existing images; it does not redraw them automatically. Repeating this action recomputes the merged prompts. Save the desired prompt edits before proceeding; operation buttons also submit the current edits.

## Requirements

- Python 3.10+ and the packages in `requirements-studio.txt`.
- FFmpeg and ffprobe on `PATH`, or paths supplied through `FFMPEG` and `FFPROBE`.
- An installed LTX runtime with `uv`, its `pyproject.toml`, and the `ltx_pipelines` package.
- Compatible LTX-2.5 model weights and a GPU/runtime capable of loading them. The web requirements file does not install the model runtime or weights.
- `DEEPSEEK_API_KEY` for dialogue extraction during prompt merging, unless offline merging is selected.
- Volcengine credentials only when using image redraw.

## Model and runtime configuration

Edit `studio_config.json` in the SceneStudio directory:

```json
{
  "model_directory": "/path/to/models/ltx-2.5"
}
```

A relative directory is resolved against the configuration file's directory. On Windows, a path such as `E:/models/ltx-2.5` is supported. Model weights are resolved by `model_config.py` from this file, not from an environment variable:

```text
<model_directory>/
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
  vae/ltx-2.5-video-vae-bf16.safetensors
  vae/ltx-2.5-audio-vae-bf16.safetensors
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
```

Set `LTX_WORKDIR` to the separate LTX runtime directory. This determines where `uv run` executes; it does not relocate the project's media or choose the model-weight directory. Restart SceneStudio after changing model configuration.

## Start the workbench

Install Miniconda or Anaconda first. From the SceneStudio directory, create and activate a dedicated environment before installing dependencies:

```bash
conda create -n scenestudio python=3.11 pip -y
conda activate scenestudio
```

Create the environment only once. In a new terminal, run `conda activate scenestudio` before running workbench commands. This environment is for SceneStudio; the separate LTX runtime configured through `LTX_WORKDIR` is still required for video generation.

Install dependencies and start the workbench in the activated environment:

```bash
python -m pip install -r requirements-studio.txt

# Linux/macOS
export LTX_WORKDIR=/path/to/LTX-runtime
# Windows PowerShell instead:
# $env:LTX_WORKDIR = "E:/path/to/LTX-runtime"

python scene_studio.py --project-root /path/to/projects --port 7200
```

Open `http://127.0.0.1:7200/?locale=en`.

For a project ID such as `demo`, SceneStudio loads `<project-root>/demo/meta.json`. Alternatively, select absolute-path input and enter the server-side project directory or its `meta.json` file. The resolved project directory is shown in the UI.

To merge visual prompts without model-based dialogue extraction:

```bash
python scene_studio.py --project-root /path/to/projects --port 7200 --offline
```

Offline mode applies to merging only; image redraw and LTX generation still require their respective services and runtime.

## Using the editor

1. Load a project containing storyboard prompts and images.
2. Choose **Create key-scene frames** to run key-frame flagging and prompt merging.
3. Review each group. Edit its image prompt, redraw if needed, and edit its scene-video prompt.
4. Set the frame controls, then run batch video generation. Existing group videos are overwritten; completed clips become available individually.
5. Use **Generate** on any group to create its first clip without a batch run. After a single or batch generation request is accepted, the affected buttons read **Regenerate**, including when generation fails and needs a retry.
6. Select the clips to include and concatenate them. Assembly follows project scene order, with fades, and displays the result below the groups.

The workbench defaults to `max-frames=480` and `frames-delta=3`. Delta uses units of eight frames. The generation scripts normalize segment lengths to LTX's `8n+1` frame constraint; the configured cap is not a promise of that exact frame count.

Tasks are serialized per project, and editing is disabled while a task runs. Job state is held in memory and running tasks are not restored after a service restart. Submit the required operation again after checking existing outputs. Different projects may run concurrently, so account for shared GPU capacity.

## Mindawaker integration

Mindawaker's **LTX-Scene-Studio** button opens the active project or chapter using:

```text
/?project_path=<URL-encoded-absolute-project-directory>&locale=en
```

`locale=zh` selects Chinese; Chinese is also the default when no locale is supplied. Project content, prompts, and raw task logs are not translated.

Mindawaker defaults to the frontend's hostname on port 7200. Set `NEXT_PUBLIC_SCENE_STUDIO_URL` in the Mindawaker frontend environment to change this address, then restart or rebuild that frontend. Both services must have access to the same project directory: the link does not transfer files.

## Project data and outputs

The workbench reads storyboard data from `task_data` in `meta.json` (or the top-level task object where supported), including `prompts`, `images`, and related timing and audio data. Merged key-scene entries use the existing field name `frame_secene_prompt`; retain this spelling for compatibility. Character information used by prompt merging lives at `artifacts/story/character_specs.json`.

```text
<project-directory>/
  meta.json
  <source and redrawn images>
  artifacts/story/character_specs.json
  key_scene_out/
    output_kf<index>.mp4      Generated group videos
    concat_fade.mp4           Selected montage
  key_scene_work/             Generation logs and intermediate frames
```

All workbench edits, redraws, videos, and intermediate files use the loaded project directory. Loading the same directory by ID or absolute path shares its task state. Image lookup can fall back from an old absolute image path to a matching filename in the current project, without borrowing images from another project.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `PROJECT_ROOT` | Default project parent directory; defaults to `/root/project`, overridden by `--project-root` |
| `LTX_WORKDIR` | Working directory for the installed LTX runtime |
| `DEEPSEEK_API_KEY` | Dialogue extraction and supported prompt-text translation operations |
| `VOLC_ACCESS_KEY`, `VOLC_SECRET_KEY` | Preferred Jimeng image-redraw credentials |
| `IMAGE_MODEL_PASSWORD` | Legacy image credential fallback in `AccessKey**SecretKey` form |
| `VOLC_REQ_KEY` | Image service request key; defaults to `jimeng_t2i_v40` |
| `FFMPEG`, `FFPROBE` | Optional executable paths |

Set variables in the environment that launches SceneStudio. A Mindawaker `.env` file is not automatically shared with this separate service.

## Command-line tools

Use explicit metadata paths to avoid depending on a script's default input location.

| Tool | Purpose |
| --- | --- |
| `generate_key_frame_flags.py` | Mark key frames in project metadata |
| `merge_frame_scene.py` | Build merged key-scene prompts; supports offline merging |
| `generate_key_scenes.py` | Generate key-scene videos; supports scene/key-frame selection and custom output directories |
| `generate_shots.py` | Generate shot videos, concatenate results, and optionally replace native audio with TTS |
| `generate_clips.py` | Generate shot clips without final assembly |
| `concat_clips.py` | Assemble clips in metadata order |
| `concat_audio.py` | Assemble project audio into AAC |
| `add_audio.py` | Mux an audio file into a video |
| `prompt_editor.py` | Standalone per-shot image-prompt editor and redraw UI |
| `key_scene_editor.py` | Key-scene editing/assembly support used by the workbench |

For the shot-based pipeline:

```bash
python generate_shots.py /path/to/project/meta.json --dry-run
python generate_shots.py /path/to/project/meta.json
python generate_shots.py /path/to/project/meta.json --scene 3 --shot 2 --force
python generate_shots.py /path/to/project/meta.json --tts
```

The shot generator uses 24 fps, splits long shots into `8n+1`-frame segments, and uses the previous segment's last frame for continuation. Its default frame cap is 121; `--max-frames` accepts a cap up to 480. It also exposes frame delta, retry, and frame-avoidance controls; see `--help` for the full options.

Without `--force`, completed nonempty outputs are reused. Reuse is based on output files rather than a prompt hash: use `--force` after changing prompts, source images, or generation settings. Native LTX audio is retained by default. `--tts` replaces it with project audio; `--prompt_with_text` adds translated dialogue to prompts, requires DeepSeek, and is mutually exclusive with `--tts`.

Output locations differ between tool families: `generate_shots.py` writes to SceneStudio's `out/` and `work/`, while the modular clip tools use `clips/`, `work/`, and `result.mp4` via `ltx_common.py`. The web workbench instead passes explicit project-local `key_scene_out/` and `key_scene_work/` paths. Do not assume these output sets are interchangeable without checking the scripts.

## API and operational notes

| Endpoint | Purpose |
| --- | --- |
| `POST /api/load` | Register/load a project path |
| `GET /api/projects/{project_id}` | Read groups, media links, and current job status |
| `POST /api/projects/{project_id}/{action}` | Save edits or request a workbench operation |
| `GET /api/projects/{project_id}/media/{kind}/{index}` | Serve thumbnails, images, clips, or assembled output |

The service defaults to `127.0.0.1` and has no account authentication. It accepts server-side project paths and is intended for local or trusted-network use. The web dependency list alone is insufficient for GPU generation: validate the LTX runtime, configured weights, and FFmpeg independently when diagnosing failed jobs.
