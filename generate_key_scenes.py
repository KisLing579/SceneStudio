#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 meta.json 的 frame_secene_prompt（关键帧段）逐段生成 LTX-2.5 视频（image-to-video）。

与 generate_shots.py 方法一致，但不再参考 task_data.key_frame_flags 与 task_data.prompts，
而是直接参考 merge_frame_scene.py 生成的顶层 frame_secene_prompt 列表：

  - scene_id          -> --scene 过滤（一个 scene_id 可能对应多个视频段）；
  - frame_image_idx   -> 该段覆盖的分镜下标列表，frame_image_idx[0] 即关键帧下标，
                        用 task_data.images[frame_image_idx[0]] 作为起始帧图像；
  - prompt            -> LTX --prompt 提示词参数（已合并的英文视觉 prompt）；
  - duration          -> 按 8n+1 换算帧数（frame_segments），超过 --max-frames 时切多段；
  - dialogue          -> 该段对白（「角色英文名：对白」列表），翻译成英文后插入到角色外观
                        描述子句处（不显示角色名）；需 DEEPSEEK_API_KEY，无 key 或翻译失败
                        时仅用 prompt，不影响生成。
  - text              -> 中文原文，--prompt_with_text 时抽取人物对白并分配到子段。

这里已经没有“分镜（shot）”的概念——每个 frame_secene_prompt 条目即一个独立视频段，
段与段之间不再首尾续接，各自从自己的关键帧图像重新开始。

输出：每条生成 out/output_kf{关键帧下标}.mp4（多段时先切成 output_kf{..}_seg{k}.mp4 再合并），
全部跑完后拼接为 out/result.mp4（保留 LTX 原生音轨）。

用法：
  python generate_key_scenes.py [meta.json 路径] [--scene N] [--force]
                                [--max-frames N] [--frames-delta N] [--retries N]
                                [--language en|zh] [--prompt_with_text]

  --scene N       只生成第 N 个场景的所有关键帧段（数字，如 --scene 3）。
  --force         强制重新生成已存在的输出（默认跳过已完成的最终视频，便于续跑）。
  --max-frames N  单段生成帧数上限，默认 121，最大 480（LTX 单段上限）。
  --frames-delta N 帧数偏移（单位：8 帧，可正可负），全局平移所有帧数，如 137->145 用 1、137->129 用 -1。
  --retries N     单段生成失败时帧数 +8 重试的最大总尝试次数，默认 3（如 137 失败 -> 145 -> 153 ...）。
  --language L    目标对白语言：zh=中文（默认，保留原文，不翻译）；en=英文（把中文对白翻译成英文）。
  --prompt_with_text  默认开启，从 text 抽取对白、过滤旁白，固定角色音色并按句分配子段（需 DEEPSEEK_API_KEY）。
  --no-prompt-with-text  关闭文本对白模式，沿用 dialogue 融合；切换模式后用 --force 重生成已有视频。

环境变量：
  LTX_WORKDIR  执行 uv run 的工作目录（含 pyproject.toml 与 ltx_pipelines 包）。
  模型权重目录从同目录的 studio_config.json 的 model_directory 读取。
  FFMPEG / FFPROBE  可选，指定 ffmpeg / ffprobe 可执行文件路径。
  DEEPSEEK_API_KEY  默认文本对白模式下，无论目标语言均需此 key 来抽取对白。
                    关闭文本对白模式后，仅 en 翻译需要；zh 不需要，en 缺失时仅用 prompt 生成。
"""

import argparse
import json
from logging import config
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from generate_shots import (
    build_character_registry, build_segment_prompts, _derive_voice,
    _deepseek_chat as _dialogue_chat, _is_narration_label,
)

# Windows 控制台默认编码可能是 cp1252/gbk，统一 UTF-8 输出，避免中文报错
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# --------------------------------------------------------------------------- #
# 可配置项（与 generate_shots.py 一致）
# --------------------------------------------------------------------------- #
FPS = 24                 # LTX 视频帧率
MAX_FRAMES = 121         # 默认单段生成帧数上限（可用 --max-frames 覆盖）
MAX_FRAMES_LIMIT = 480   # LTX 单段帧数上限（--max-frames 不允许超过此值）
SEED = 42                # 固定种子（对应命令里的 --seed）
IMAGE_START = "0"        # --image 后的第一个位置参数
IMAGE_STRENGTH = "0.8"   # --image 后的第二个位置参数
TIMEOUT = 3600           # 单次生成超时（秒），超时会杀掉子进程
RETRY_ATTEMPTS = 3       # 单段生成失败时，帧数 +8 重试的最大总尝试次数

# Model weights are configured separately from the LTX command working directory.
from model_config import load_ltx_directory, load_model_paths 

MODELS = load_model_paths()
LTX_DIR = load_ltx_directory()

SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = os.environ.get("LTX_WORKDIR") or str(SCRIPT_DIR)
OUT_DIR = SCRIPT_DIR / "out"           # 输出目录：关键帧段视频与结果视频
WORK_DIR = SCRIPT_DIR / "work"         # 续接帧、日志等中间产物
RESULT_PATH = OUT_DIR / "result.mp4"   # 全部视频段拼接后的结果视频


def _which_or_env(name, env_var):
    """优先使用环境变量指定的可执行文件，否则回退到 PATH 查找。"""
    exe = os.environ.get(env_var)
    if exe:
        return exe
    found = shutil.which(name)
    return found or name


FFMPEG = _which_or_env("ffmpeg", "FFMPEG")
FFPROBE = _which_or_env("ffprobe", "FFPROBE")


# --------------------------------------------------------------------------- #
# meta.json 读取（frame_secene_prompt + images）
# --------------------------------------------------------------------------- #
def resolve_project_image(directory, value):
    """Resolve project assets locally, including paths from a migrated meta.json."""
    directory = Path(directory).resolve()
    raw = str(value or "").replace("\\", "/")
    path = (directory / raw).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        path = (directory / Path(raw).name).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"图像不存在或不在项目目录内：{value}")
    return path


def load_key_scenes(meta_path):
    """读取顶层 frame_secene_prompt 列表与 task_data.images。

    返回 (entries, images)。entries 每项含 scene_id / world_scene_id /
    frame_image_idx / text / dialogue / prompt / duration。
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    task = data.get("task_data") or {}
    images = task.get("images") or []
    entries = data.get("frame_secene_prompt") or []
    if not entries:
        raise RuntimeError("meta.json 中未找到 frame_secene_prompt（请先运行 merge_frame_scene.py）")
    if not images:
        raise RuntimeError("meta.json 中未找到 task_data.images")
    return entries, images


# --------------------------------------------------------------------------- #
# 帧数换算与切段（与 generate_shots.py 一致，去掉 --avoid-frames）
# --------------------------------------------------------------------------- #
def frame_segments(duration, fps=FPS, max_frames=MAX_FRAMES, delta=0):
    """由秒数换算成若干段 num_frames 列表，每段均为 8n+1 且 <= max_frames。

    delta 为帧数偏移（单位：8 帧，可正可负），用于全局平移所有帧数。
    """
    duration = float(duration)
    frames = 8 * (round(duration * fps / 8) + delta) + 1
    frames = max(frames, 1)
    if frames <= max_frames:
        return [frames]
    # 拆分“有效负载”帧（去掉每段的 +1 头帧），每段负载为 8 的倍数且 <= max_frames-1
    payload = frames - 1
    cap = max_frames - 1
    segments = []
    while payload > 0:
        chunk = min(payload, cap)
        chunk -= chunk % 8
        segments.append(chunk + 1)
        payload -= chunk
    return segments


# --------------------------------------------------------------------------- #
# 命令构建与执行（与 generate_shots.py 一致）
# --------------------------------------------------------------------------- #
def build_command(prompt, image_path, frames, output_path):
    return [
        "uv",  "--project", LTX_DIR,"run", "python", "-m", "ltx_pipelines.distilled",
        "--transformer-path", MODELS["transformer"],
        "--text-encoder-path", MODELS["text_encoder"],
        "--video-vae-path", MODELS["video_vae"],
        "--audio-vae-path", MODELS["audio_vae"],
        "--spatial-upsampler-path", MODELS["spatial_upsampler"],
        "--num-frames", str(frames),
        "--seed", str(SEED),
        "--output-path", str(output_path),
        "--prompt", prompt,
        "--image", str(image_path), IMAGE_START, IMAGE_STRENGTH,
    ]


def run_generation(command, log_path):
    """执行一条生成命令并等待结束；失败时抛出带日志尾部的异常。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as logf:
        logf.write((" ".join(command) + "\n").encode("utf-8"))
        logf.flush()
        try:
            proc = subprocess.run(
                command,
                cwd=WORKDIR,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"生成超时（>{TIMEOUT}s），日志见 {log_path}")
    if proc.returncode != 0:
        tail = _tail(log_path)
        raise RuntimeError(f"生成失败（返回码 {proc.returncode}），日志见 {log_path}\n{tail}")


def _tail(log_path, n=2000):
    try:
        return Path(log_path).read_text(encoding="utf-8", errors="replace")[-n:]
    except OSError:
        return ""


def _generate_with_retry(prompt, image_path, nf, output_path, log_path, label,
                         retries=RETRY_ATTEMPTS, max_frames=MAX_FRAMES_LIMIT):
    """生成单个视频段；失败时帧数 +8 重试，最多尝试 retries 次。"""
    cur_nf = nf
    attempt = 0
    while attempt < retries:
        attempt += 1
        command = build_command(prompt, image_path, cur_nf, output_path)
        try:
            run_generation(command, log_path)
            if cur_nf != nf:
                print(f"    {label} 帧数 {nf} 失败后改 {cur_nf} 帧重试成功", flush=True)
            return True
        except Exception as e:
            if attempt >= retries:
                raise
            next_nf = cur_nf + 8
            if next_nf > max_frames:
                print(f"    {label} 帧数 {cur_nf} 再 +8 将超过单段上限 {max_frames}，无法绕开", flush=True)
                raise
            try:
                Path(output_path).unlink()
            except OSError:
                pass
            err_line = " ".join(str(e).split())[:200]
            print(f"    {label} 第 {attempt} 次失败（{err_line}），"
                  f"帧数 {cur_nf} -> {next_nf} 重试 ...", flush=True)
            cur_nf = next_nf
    raise RuntimeError(f"{label} 重试 {retries} 次仍失败")


# --------------------------------------------------------------------------- #
# 最后一帧抽取（逻辑同 extract_last_frame.py / generate_shots.py）
# --------------------------------------------------------------------------- #
def extract_last_frame(video_path, image_path):
    """用 ffprobe 统计总帧数，ffmpeg 抽取最后一帧。"""
    video_path = str(video_path)
    image_path = str(image_path)
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)

    probe_cmd = [
        FFPROBE, "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    r = subprocess.run(probe_cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    value = r.stdout.strip()
    if not value.isdigit():
        raise RuntimeError(f"无法读取视频帧数: {r.stderr.strip() or value}")
    total = int(value)
    last_index = total - 1

    ff_cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"select=eq(n\\,{last_index})",
        "-frames:v", "1", "-y", image_path,
    ]
    r2 = subprocess.run(ff_cmd, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    if r2.returncode != 0:
        raise RuntimeError(r2.stderr.strip())
    if not os.path.isfile(image_path):
        raise RuntimeError("抽帧失败: 未生成图像文件")
    return image_path


# --------------------------------------------------------------------------- #
# 视频段合并（ffmpeg concat demuxer，保留音轨）
# --------------------------------------------------------------------------- #
def concat_videos(seg_paths, output_path, keep_audio=False):
    output_path = str(output_path)
    list_file = output_path + ".concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in seg_paths:
            f.write(f"file '{str(p).replace(chr(92), '/')}'\n")
    codec_args = ["-c", "copy"] if keep_audio else ["-c:v", "copy", "-an"]
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        *codec_args, output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if os.path.exists(list_file):
        os.remove(list_file)
    if r.returncode != 0:
        raise RuntimeError(f"合并视频失败: {r.stderr.strip()}")
    if not os.path.isfile(output_path):
        raise RuntimeError("合并视频失败: 未生成输出文件")
    return output_path


# --------------------------------------------------------------------------- #
# 对白融合：中→英翻译 + 插入角色外观描述处（不显示角色名）
# --------------------------------------------------------------------------- #
def _deepseek_chat(system_prompt, user_prompt, temperature=0.3):
    """调用 DeepSeek chat completions，返回 content；需环境变量 DEEPSEEK_API_KEY。"""
    import requests  # 惰性导入，仅翻译对白时才需要

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("对白翻译需要环境变量 DEEPSEEK_API_KEY")

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            json={
                "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        raise RuntimeError(f"DeepSeek 调用失败: {e}") from e

    if not out:
        raise RuntimeError("DeepSeek 返回结果为空")
    return out


def translate_to_english(text):
    """把中文文本翻译成英文（对白融合用）。失败时抛出 RuntimeError。"""
    text = (text or "").strip()
    if not text:
        return ""
    return _deepseek_chat(
        "你是专业翻译。把下面的中文文本翻译成英文，只输出英文译文，不要任何解释、前缀或引号。",
        text,
        temperature=0.3,
    )


_warned_no_key = False  # 只在第一次缺少 key 时提示，避免刷屏


def load_character_desc(meta_path):
    """读取 meta.json 所在目录 artifacts/story 下的 character_specs.json，返回 {英文名: 外观描述}。

    用于对白融合时按角色英文名定位 prompt 里的外观描述子句。文件缺失时返回空字典。
    """
    char_file = Path(meta_path).parent / "artifacts" / "story" / "character_specs.json"
    if not char_file.is_file():
        return {}
    specs = json.loads(char_file.read_text(encoding="utf-8"))
    out = {}
    for s in specs:
        name = (s.get("name") or "").strip()
        desc = (s.get("visual_core") or s.get("description") or "").strip()
        if name:
            out[name] = desc
    return out


def _split_dialogue(line):
    """把「角色英文名：对白」拆成 (name, 对白)。对白可含冒号，取第一个分隔。"""
    line = (line or "").strip()
    if not line:
        return "", ""
    m = re.search(r"[:：]\s*", line)
    if not m:
        return "", line
    return line[:m.start()].strip(), line[m.end():].strip()


def _insert_after_clause(prompt, desc, snippets):
    """在 prompt 里找到 desc（角色外观描述）后插入 snippets（若干 saying 子句）。

    desc 可能跨多个逗号子句：优先精确子串定位，失败退化为按 desc 首个子句定位；
    desc 为空（说话者不在角色名单）或仍未找到时，插到风格后缀之前（否则追加到末尾）。
    snippets 按顺序用 ", " 连接。
    """
    prompt = (prompt or "").strip()
    if not snippets:
        return prompt
    combined = ", ".join(snippets)

    def _insert_at(end):
        nxt = prompt.find(",", end)
        if nxt != -1:
            return prompt[:nxt] + ", " + combined + prompt[nxt:]
        return prompt[:end] + ", " + combined + prompt[end:]

    if desc:
        idx = prompt.find(desc)
        if idx != -1:
            return _insert_at(idx + len(desc))
        for clause in (c.strip() for c in desc.split(",") if c.strip()):
            ci = prompt.find(clause)
            if ci != -1:
                return _insert_at(ci + len(clause))
    si = prompt.lower().find("highly detailed")
    if si != -1:
        return (prompt[:si].rstrip().rstrip(",").strip() + ", " + combined + " " + prompt[si:]).strip()
    return (prompt + ", " + combined).strip()


def _build_prompt(prompt, dialogue, char_desc, language="en"):
    """主 prompt 为已合并的英文视觉提示词；dialogue 为「角色英文名：对白」列表。

    融合规则（用户要求）：不显示角色名，而是找到该角色在 prompt 里的外观描述子句，
    把对白作为 saying: "..." 子句插到该描述之后。同一角色多句对白按顺序一并插入。
    language=en 时把中文对白翻译成英文（需 DEEPSEEK_API_KEY）；language=zh 时保留中文
    原文、不翻译。无 dialogue、无 key（en 时）或翻译失败时仅返回 prompt，不影响生成。
    """
    global _warned_no_key
    prompt = (prompt or "").strip()
    dialogue = dialogue or []
    if not dialogue:
        return prompt
    if language != "zh" and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        if not _warned_no_key:
            print("    提示：未设置 DEEPSEEK_API_KEY，跳过对白融合，仅用 prompt。", flush=True)
            _warned_no_key = True
        return prompt

    by_char = {}   # name -> [saying 子句]
    order = []     # 角色出现顺序
    for line in dialogue:
        name, spoken = _split_dialogue(line)
        if not spoken:
            continue
        if language == "zh":
            spoken_out = spoken
        else:
            try:
                spoken_out = translate_to_english(spoken)
            except Exception as e:
                err_line = " ".join(str(e).split())[:120]
                print(f"    提示：对白翻译失败（{err_line}），跳过该句。", flush=True)
                continue
            if not spoken_out:
                continue
        if name not in by_char:
            by_char[name] = []
            order.append(name)
        by_char[name].append(f'saying: "{spoken_out}"')

    fused = prompt
    for name in order:
        desc = (char_desc.get(name) or "") if name else ""
        fused = _insert_after_clause(fused, desc, by_char[name])
    return fused


def extract_spoken_text(image_prompt, text, prev_text="", char_map=None, prev_char=None,
                        language="en"):
    """结合图像 prompt、中文台词与角色列表，抽取人物对话（英文），去掉旁白/画面描述。

    - 中文文本含成对引号、未闭合左引号，或「上文对话引号未闭合、本段是延续」时 -> 人物对话：
      结合角色列表判断说话者对应哪个角色 id，把对话完整翻译成口语化英文台词。
    - 否则 -> 旁白/画面描述：不输出任何内容（返回空串，由调用方只保留 image_prompt）。
    返回 "Dialogue: <角色id> says \"<英文台词>\""，或旁白时返回 ""。
    """
    image_prompt = (image_prompt or "").strip()
    text = (text or "").strip()
    if not text:
        return ""
    prev_text = (prev_text or "").strip()
    char_lines = "".join(f"- {cid}: {clause}\n" for cid, clause in (char_map or {}).items())
    system = (
        "你是影视台词编辑。根据图像分镜 prompt（英文）、分镜中文文本、上一个分镜的中文文本"
        "（用于判断对话是否延续）与角色列表，判断本分镜是否含【人物对话】。\n"
        "判定为【人物对话】的情况：\n"
        "1. 当前文本含成对的引号（“”「」等）直接引语；\n"
        "2. 当前文本含未闭合的左引号（如“……”却没有对应的右引号），说明对话刚开始、尚未结束；\n"
        "3. 当前文本没有引号、或只有右引号（如……”，），但上一个分镜的对话引号尚未闭合——"
        "说明本段是上一句对话的延续，按对话处理，说话者沿用上一个分镜的说话角色。\n"
        "只有上述都不满足时，本分镜属于【旁白/画面描述】，无人物对话。\n"
        "- 人物对话：结合图像 prompt 与角色列表，判断说话者对应角色列表中的哪个角色 id"
        "（角色 id 必须原样照抄，不要自创或改名），把引号内（或延续的）对话完整翻译成"
        "口语化英文台词，不要删减对话内容。\n"
        "- 旁白/画面描述（无人物对话）：直接输出空行，不要输出任何旁白、叙述或画面描述文字。\n"
        "角色列表（每行一个角色 id 及其英文外观描述）：\n"
        f"{char_lines}"
        "严格按以下格式输出，不要任何解释、多余前缀，也不要再用引号包裹整行：\n"
        "人物对话：Dialogue: <角色id> says \"<英文台词>\"\n"
        "旁白/画面描述：（什么都不写，空行）"
    )
    if language == "zh":
        system += "\n本次目标对白语言为中文：上述英文翻译要求改为保留中文对白原文；Dialogue: <角色id> says 格式不变。"
    prev = prev_text if prev_text else "(无)"
    prev_c = prev_char or "(无)"
    user = (
        f"上一个分镜文本：{prev}\n"
        f"上一个分镜说话角色：{prev_c}\n"
        f"图像 prompt：{image_prompt}\n"
        f"本分镜中文文本：{text}"
    )
    spoken = _dialogue_chat(system, user, temperature=0.4, allow_empty=True)
    # 旁白/画面描述兜底：DeepSeek 有时不返回空串，而是输出「旁白」「旁白：……」等标签，
    # 这里直接判为无对话返回空串（后续 build_segment_prompts 仍会再兜底拦截一次）。
    if _is_narration_label(spoken):
        return ""
    return spoken



def _split_dialogue_by_segments(dialogue, segs):
    """把有序对白列表按子段帧数比例切成若干连续块（每个对白只出现一次，不跨子段重复）。

    segs 为各子段帧数列表。返回与 segs 等长的列表，每项是该子段应携带的对白子列表。
    单子段时全部对白归该子段；无对白时返回若干空列表。
    """
    dialogue = list(dialogue or [])
    if not dialogue:
        return [[] for _ in segs]
    if len(segs) == 1:
        return [dialogue]
    total = sum(segs)
    n = len(dialogue)
    chunks = []
    prev = 0
    acc = 0
    for seg in segs:
        acc += seg
        cur = int(round(n * acc / total))
        cur = max(prev, min(n, cur))
        chunks.append(dialogue[prev:cur])
        prev = cur
    return chunks


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _exists_nonempty(path):
    """文件存在且非空。"""
    return Path(path).is_file() and Path(path).stat().st_size > 0


def _parse_num_arg(value):
    """argparse 类型：把 '3' 或 'scene3' 解析成 int。"""
    m = re.search(r"\d+", str(value))
    if not m:
        raise argparse.ArgumentTypeError(f"无法解析数字: {value!r}")
    return int(m.group())


def _parse_max_frames(value):
    """argparse 类型：解析单段帧数上限，限制在 [9, MAX_FRAMES_LIMIT]。"""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"无法解析帧数: {value!r}")
    if not (9 <= n <= MAX_FRAMES_LIMIT):
        raise argparse.ArgumentTypeError(
            f"单段帧数需在 9~{MAX_FRAMES_LIMIT} 之间（LTX 单段上限），收到 {n}"
        )
    return n


def _entry_key(entry):
    """返回该条目的关键帧下标（frame_image_idx 首元素），用于文件命名与排序。"""
    fi = entry.get("frame_image_idx") or []
    return int(fi[0]) if fi else None


def main():
    global OUT_DIR, WORK_DIR, RESULT_PATH
    parser = argparse.ArgumentParser(description="按 meta.json 的 frame_secene_prompt 生成 LTX-2.5 视频")
    parser.add_argument("meta", nargs="?", default="./work/meta.json",
                        help="meta.json 路径（默认 ./work/meta.json）")
    parser.add_argument("--scene", type=_parse_num_arg, default=None,
                        help="只生成指定场景的所有关键帧段（数字，如 --scene 3）")
    parser.add_argument("--force", action="store_true", help="重新生成已存在的输出")
    parser.add_argument("--max-frames", "--max_frames", type=_parse_max_frames, default=MAX_FRAMES,
                        help=f"单段生成帧数上限，默认 {MAX_FRAMES}，最大 {MAX_FRAMES_LIMIT}（LTX 单段上限）")
    parser.add_argument("--frames-delta", type=int, default=0,
                        help="帧数偏移（单位：8 帧，可正可负），全局平移所有帧数，如 137->145 用 1、137->129 用 -1")
    parser.add_argument("--retries", type=int, default=RETRY_ATTEMPTS,
                        help=f"单段生成失败时帧数 +8 重试的最大总尝试次数，默认 {RETRY_ATTEMPTS}（如 137 失败 -> 145 -> 153 ...）")
    parser.add_argument("--language", choices=["en", "zh"], default="zh",
                        help="目标对白语言：zh=中文（默认，保留原文）；en=英文（翻译成英文）")
    parser.add_argument("--prompt_with_text", "--prompt-with-text", action="store_true", default=True,
                        help="从关键帧段 text 抽取人物对白、过滤旁白，固定音色并分配到子段（默认开启，需 DEEPSEEK_API_KEY；优先于 dialogue）")
    parser.add_argument("--no-prompt-with-text", "--no_prompt_with_text", dest="prompt_with_text",
                        action="store_false", help="关闭文本对白模式，沿用 dialogue 融合")
    parser.add_argument("--key-frame", type=int, help="只生成指定关键帧下标的视频段")
    parser.add_argument("--out-dir", type=Path, help="视频输出目录")
    parser.add_argument("--work-dir", type=Path, help="日志与中间帧目录")
    parser.add_argument("--no-concat", action="store_true", help="仅生成视频段，不自动拼接")
    args = parser.parse_args()
    if args.out_dir:
        OUT_DIR = args.out_dir.resolve()
    if args.work_dir:
        WORK_DIR = args.work_dir.resolve()
    RESULT_PATH = OUT_DIR / "result.mp4"

    meta_path = Path(args.meta).resolve()
    entries, images = load_key_scenes(meta_path)
    char_desc = load_character_desc(meta_path)   # 英文名 -> 外观描述（对白融合定位用）
    char_map, char_voices = {}, {}
    if args.prompt_with_text:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        registry = build_character_registry((data.get("task_data") or {}).get("prompts") or [])
        char_map = {cid: info["clause"] for cid, info in registry.items()}
        char_voices = {cid: info["voice"] for cid, info in registry.items()}
        for name, desc in char_desc.items():
            char_map.setdefault(name, desc)
            char_voices.setdefault(name, _derive_voice(desc))
    base_dir = meta_path.parent
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 按 --scene 过滤（scene_id 为字符串，如 "1".."56"）
    selected = list(entries)
    if args.scene is not None:
        selected = [e for e in entries if int(str(e.get("scene_id"))) == args.scene]
        if not selected:
            raise RuntimeError(f"没有匹配指定 --scene {args.scene} 的关键帧段")
    if args.key_frame is not None:
        selected = [e for e in selected if _entry_key(e) == args.key_frame]
        if not selected:
            raise RuntimeError(f"没有匹配关键帧 {args.key_frame} 的视频段")

    # 统计视频段数（含切段），用于头部汇总
    total_entries = len(selected)
    total_segments = 0
    for e in selected:
        dur = float(e.get("duration") or 0.0)
        total_segments += len(frame_segments(dur, max_frames=args.max_frames, delta=args.frames_delta))

    print("=" * 70)
    print(f"meta.json : {meta_path}")
    if args.scene is not None:
        print(f"筛选条件 : 场景 {args.scene}")
    print(f"关键帧段 : {total_entries}（来自 frame_secene_prompt）")
    print(f"视频段数 : {total_segments}（含切段，单段 <= {args.max_frames} 帧）")
    print(f"输出目录 : {OUT_DIR}")
    print(f"结果视频 : {RESULT_PATH}")
    print(f"工作目录 : {WORKDIR}")
    print("=" * 70)

    all_clips = []   # 按顺序收集每条关键帧段的最终视频，最后拼接为结果视频
    prev_text, prev_char, prev_scene = "", None, None

    for pos, e in enumerate(selected):
        kf_idx = _entry_key(e)
        if kf_idx is None or not (0 <= kf_idx < len(images)):
            raise RuntimeError(f"第 {pos+1} 条 frame_image_idx 非法：{e.get('frame_image_idx')}")

        scene_id = str(e.get("scene_id"))
        label = f"scene{scene_id}/kf{kf_idx}"
        start_image = resolve_project_image(base_dir, images[kf_idx])

        prompt = (e.get("prompt") or "").strip()
        dialogue = e.get("dialogue") or []
        dur = float(e.get("duration") or 0.0)
        segs = frame_segments(dur, max_frames=args.max_frames, delta=args.frames_delta)

        seg_prompts = None
        if args.prompt_with_text:
            if scene_id != prev_scene:
                prev_text, prev_char = "", None
            line_text = (e.get("text") or "").strip()
            spoken = extract_spoken_text(
                prompt, line_text, prev_text, char_map=char_map,
                prev_char=prev_char, language=args.language,
            )
            seg_prompts = build_segment_prompts(
                prompt, spoken, segs, max_frames=args.max_frames, char_voices=char_voices,
            )
            match = re.match(r"(?is)^Dialogue\s*[:：]\s*(.+?)\s+says\b", spoken or "")
            prev_char = match.group(1).strip() if match else None
            prev_text, prev_scene = line_text, scene_id

        # 对白按子段帧数比例切割，避免同一对白在多个子段里重复出现
        dialogue_chunks = _split_dialogue_by_segments(dialogue, segs)

        final_output = OUT_DIR / f"output_kf{kf_idx}.mp4"
        all_clips.append(final_output)

        # 续跑：最终视频已存在且非空，且未指定 --force，则跳过该段
        if (not args.force and final_output.is_file() and final_output.stat().st_size > 0):
            print(f"[{pos+1}/{total_entries}] {label} 已存在，跳过", flush=True)
            continue

        seg_videos = []
        seg_image = start_image
        for k, nf in enumerate(segs):
            if len(segs) == 1:
                seg_output = OUT_DIR / f"output_kf{kf_idx}.pending.mp4"
            else:
                seg_output = OUT_DIR / f"output_kf{kf_idx}_seg{k}.mp4"

            if len(segs) > 1 and _exists_nonempty(seg_output) and not args.force:
                print(f"[{pos+1}/{total_entries}] {label} 段{k+1}/{len(segs)} "
                      f"已存在，跳过（{seg_output.name}）", flush=True)
            else:
                log_path = WORK_DIR / "logs" / f"output_kf{kf_idx}_seg{k}.log"
                print(f"[{pos+1}/{total_entries}] {label} 段{k+1}/{len(segs)} "
                      f"frames={nf} 生成中 ...", flush=True)
                # prompt 为主提示词；该子段只携带属于它自己的对白（en 翻译 / zh 保留原文）
                seg_prompt = (seg_prompts[k] if seg_prompts is not None else
                              _build_prompt(prompt, dialogue_chunks[k], char_desc, args.language))
                _generate_with_retry(seg_prompt, seg_image, nf, seg_output, log_path,
                                     f"{label} 段{k+1}/{len(segs)}", retries=args.retries,
                                     max_frames=args.max_frames)

            seg_videos.append(seg_output)

            # 下一段起始图 = 本段最后一帧（多段首尾相接）
            if k < len(segs) - 1:
                seg_last = WORK_DIR / f"output_kf{kf_idx}_seg{k}_last.png"
                if not _exists_nonempty(seg_last) or args.force:
                    extract_last_frame(seg_output, seg_last)
                seg_image = seg_last

        # 多段合并为最终视频
        if len(segs) > 1:
            if _exists_nonempty(final_output) and not args.force:
                print(f"    合并结果已存在，跳过 -> {final_output.name}", flush=True)
            else:
                print(f"    合并 {len(segs)} 段 -> {final_output.name}", flush=True)
                pending = OUT_DIR / f"output_kf{kf_idx}.pending.mp4"
                concat_videos(seg_videos, pending, keep_audio=True)
                pending.replace(final_output)
        else:
            seg_videos[0].replace(final_output)

        print(f"    完成 {final_output.name}", flush=True)

    # 拼接结果视频（保留 LTX 原生音轨）
    if args.no_concat:
        return
    print("=" * 70)
    print(f"正在拼接结果视频 -> {RESULT_PATH} ...", flush=True)
    concat_videos(all_clips, RESULT_PATH, keep_audio=True)
    print(f"结果视频已生成：{RESULT_PATH}")


if __name__ == "__main__":
    main()
