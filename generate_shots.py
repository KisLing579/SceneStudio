#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 meta.json 逐分镜生成 LTX-2.5 视频（image-to-video，首帧续接）。

规则：
  1. 依次读取 task_data 下的 prompts / images（长度动态读取，不写死 72）。
  2. 每个分镜四个参数：
       PROMPT = image_prompt
       IDX    = shot_idx
       FRAMES = 8 * round(duration * FPS / 8) + 1   （LTX 要求帧数为 8n+1）
       IMAGE  = key_frame_flags[i] 为 True（关键帧）-> 对应 images[i]
                key_frame_flags[i] 为 False          -> 上一个分镜最终视频的最后一帧
                （无 key_frame_flags 时回退：场景内第一个分镜 -> images[i]，
                  同场景后续分镜 -> 上一分镜最终视频的最后一帧）
  3. GPU 单次最多生成 --max-frames 帧（默认 121，上限 480）；换算出的 FRAMES > 该值时，
     切成多段“首尾相接”生成（上一段最后一帧作为下一段起始图），
     最后用 ffmpeg 合并为 output_{IDX}.mp4。
  4. 逐条执行 `uv run python -m ltx_pipelines.distilled`，等待结束再继续。
  5. 分镜视频与结果视频输出到 out/ 目录（不存在则自动创建），全部跑完后拼接为 out/result.mp4，
     默认保留 LTX 原生音轨；加 --tts 时改用 meta.json 的 audios 按顺序拼接合成。

用法：
  python generate_shots.py [meta.json 路径] [--dry-run] [--force]
                           [--scene N] [--shot M] [--tts] [--prompt_with_text]
                           [--max-frames N]

  --dry-run  只打印每个分镜的参数与命令，不真正执行。
  --force    强制重新生成已存在的输出（默认跳过已完成的最终视频，便于续跑）。
  --scene N  只生成第 N 个场景的所有分镜（如 --scene 3）。
  --shot M   只生成指定场景内的第 M 个分镜，需与 --scene 配合（如 --scene 3 --shot 2）。
  --max-frames N  单段生成帧数上限，默认 121，最大 480（LTX 单段上限）。
  --tts      用 audios 配音替换原生音轨（默认保留 LTX 原生音轨；与 --prompt_with_text 互斥）。
  --prompt_with_text  结合 image_prompt 与对应台词，抽取人物对话（去掉旁白/画面描述），并按
                      meta.json 角色固定年龄/音色；能在一段(121帧)内说完则整体放入，说不完则按句
                      分配到各子段，拼接后还原完整台词（需 DEEPSEEK_API_KEY）。

环境变量：
  LTX_WORKDIR  执行 uv run 的工作目录（含 pyproject.toml 与 ltx_pipelines 包，
               默认取本脚本所在目录）。模型目录从 studio_config.json 的 model_directory 读取。
  FFMPEG / FFPROBE  可选，指定 ffmpeg / ffprobe 可执行文件路径。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Windows 控制台默认编码可能是 cp1252/gbk，统一 UTF-8 输出，避免中文报错
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# --------------------------------------------------------------------------- #
# 可配置项
# --------------------------------------------------------------------------- #
FPS = 24                 # LTX 视频帧率
MAX_FRAMES = 121         # 默认单段生成帧数上限（可用 --max-frames 覆盖）
MAX_FRAMES_LIMIT = 480   # LTX 单段帧数上限（--max-frames 不允许超过此值）
SEED = 42                # 固定种子（对应命令里的 --seed）
IMAGE_START = "0"        # --image 后的第一个位置参数
IMAGE_STRENGTH = "0.8"   # --image 后的第二个位置参数（参考实现用 0.9，这里按任务用 0.8）
TIMEOUT = 3600           # 单次生成超时（秒），超时会杀掉子进程
RETRY_ATTEMPTS = 3       # 单段生成失败时，帧数 +8 重试的最大总尝试次数

# Model weights are configured separately from the LTX command working directory.
from model_config import load_model_paths

MODELS = load_model_paths()

SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = os.environ.get("LTX_WORKDIR") or str(SCRIPT_DIR)
OUT_DIR = SCRIPT_DIR / "out"           # 输出目录：分镜视频与结果视频
CLIPS_DIR = OUT_DIR                    # 分镜视频段（最终分镜视频 + 切段中间视频）
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
# meta.json 读取
# --------------------------------------------------------------------------- #
def load_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    task = data.get("task_data") or data
    prompts = task.get("prompts") or []
    images = task.get("images") or []
    durations = task.get("durations") or []
    audios = task.get("audios") or []
    lines = task.get("lines") or []
    key_frame_flags = task.get("key_frame_flags") or []
    if not prompts:
        raise RuntimeError("meta.json 中未找到 prompts 列表")
    if len(images) != len(prompts):
        raise RuntimeError(
            f"images 长度({len(images)}) 与 prompts 长度({len(prompts)}) 不一致"
        )
    if audios and len(audios) != len(prompts):
        raise RuntimeError(
            f"audios 长度({len(audios)}) 与 prompts 长度({len(prompts)}) 不一致"
        )
    if lines and len(lines) != len(prompts):
        raise RuntimeError(
            f"lines 长度({len(lines)}) 与 prompts 长度({len(prompts)}) 不一致"
        )
    if key_frame_flags and len(key_frame_flags) != len(prompts):
        raise RuntimeError(
            f"key_frame_flags 长度({len(key_frame_flags)}) 与 prompts 长度({len(prompts)}) 不一致"
        )
    return prompts, images, durations, audios, lines, key_frame_flags


# --------------------------------------------------------------------------- #
# 帧数换算与切段
# --------------------------------------------------------------------------- #
def frame_segments(duration, fps=FPS, max_frames=MAX_FRAMES, delta=0, avoid=None):
    """由秒数换算成若干段 num_frames 列表，每段均为 8n+1 且 <= max_frames。

    delta 为帧数偏移（单位：8 帧，可正可负），用于全局平移所有帧数。
    avoid 为帧数黑名单（set），命中时向下 -8 绕开（如 137 -> 129），
    只绕开指定值、不影响其它帧数，适合精准避开特定帧数触发的 kernel bug。
    """
    duration = float(duration)
    frames = 8 * (round(duration * fps / 8) + delta) + 1
    frames = max(frames, 1)
    if avoid:
        while frames in avoid and frames >= 17:
            frames -= 8
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
# 命令构建与执行
# --------------------------------------------------------------------------- #
def build_command(prompt, image_path, frames, output_path):
    return [
        "uv", "run", "python", "-m", "ltx_pipelines.distilled",
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
    """生成单个视频段；失败时帧数 +8 重试，最多尝试 retries 次。

    每次失败把帧数 +8（仍满足 8n+1 约束），用于绕开特定帧数触发的 GPU 内存失败，
    但重试帧数不超过 max_frames（单段上限）。重试前删除半成品输出，避免被
    _exists_nonempty 误判为已存在。仍失败则抛出最后一次异常。返回 True 表示成功生成。
    """
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
# 最后一帧抽取（逻辑同 extract_last_frame.py）
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
# 人脸检测（OpenCV 读图 + dlib HOG 检测器，用于非关键帧分镜的续接兜底）
# --------------------------------------------------------------------------- #
_face_detector = None            # dlib HOG 检测器（惰性初始化）
_face_detect_unavailable = False  # cv2/dlib 缺失时置 True，仅提示一次


def _detect_face(image_path):
    """检测图像是否含人脸，返回 True/False；无法判断（缺依赖/读图失败）返回 None。"""
    global _face_detector, _face_detect_unavailable
    if _face_detect_unavailable:
        return None
    try:
        import cv2
        import dlib
    except ImportError:
        _face_detect_unavailable = True
        print("    提示：未安装 cv2/dlib，跳过人脸检测（非关键帧按原续接逻辑处理）。", flush=True)
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    if _face_detector is None:
        _face_detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return len(_face_detector(gray, 1)) > 0


# --------------------------------------------------------------------------- #
# 视频段合并（ffmpeg concat demuxer，默认无损拷贝视频流、丢弃音频；
# keep_audio=True 时连同音轨一起拷贝，用于同一分镜多段首尾相接）
# --------------------------------------------------------------------------- #
def concat_videos(seg_paths, output_path, keep_audio=False):
    output_path = str(output_path)
    list_file = output_path + ".concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in seg_paths:
            # ffmpeg concat 用正斜杠、单引号包裹，避免反斜杠转义问题
            f.write(f"file '{str(p).replace(chr(92), '/')}'\n")
    # 合并同一分镜多段时保留音轨（-c copy 同时拷贝视频与音频），结果拼接仍丢弃音频
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
# 配音拼接与合成（ffmpeg concat demuxer 拼接音频，再 mux 进视频）
# --------------------------------------------------------------------------- #
def concat_audios(audio_paths, output_path):
    """按顺序拼接多段配音为一段 AAC 音频。"""
    output_path = str(output_path)
    list_file = output_path + ".concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in audio_paths:
            f.write(f"file '{str(p).replace(chr(92), '/')}'\n")
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-c:a", "aac", "-b:a", "192k", output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if os.path.exists(list_file):
        os.remove(list_file)
    if r.returncode != 0:
        raise RuntimeError(f"拼接配音失败: {r.stderr.strip()}")
    if not os.path.isfile(output_path):
        raise RuntimeError("拼接配音失败: 未生成输出文件")
    return output_path


def mux_audio(video_path, audio_path, output_path):
    """把配音 mux 进视频（视频流无损拷贝，音频流 AAC）。"""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"合成配音失败: {r.stderr.strip()}")
    if not os.path.isfile(str(output_path)):
        raise RuntimeError("合成配音失败: 未生成输出文件")
    return str(output_path)


# --------------------------------------------------------------------------- #
# DeepSeek 文本处理（中→英翻译 / 台词·旁白抽取）
# --------------------------------------------------------------------------- #
def _deepseek_chat(system_prompt, user_prompt, temperature=0.3, allow_empty=False):
    """调用 DeepSeek chat completions，返回 content；需环境变量 DEEPSEEK_API_KEY。

    allow_empty=True 时，模型返回空内容不报错（用于「无对白」返回空串）。
    """
    import requests  # 惰性导入，仅 --prompt_with_text 时才需要

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("使用 --prompt_with_text 需要设置环境变量 DEEPSEEK_API_KEY")

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

    if not out and not allow_empty:
        raise RuntimeError("DeepSeek 返回结果为空")
    return out


def translate_to_english(text):
    """把中文文本翻译成英文。需要环境变量 DEEPSEEK_API_KEY；失败时抛出 RuntimeError。"""
    text = (text or "").strip()
    if not text:
        return ""
    return _deepseek_chat(
        "你是专业翻译。把下面的中文翻译成英文，只输出英文译文，不要任何解释、前缀或引号。",
        text,
        temperature=0.3,
    )


SPEECH_WPS = 2.2  # 英文口语语速（词/秒），用于估算一句对白能否在一段(121帧)内说完

# 角色音色推导关键词（从英文外观描述中提取年龄/性别/类型，保证同一角色音色恒定）
_AGE_WORDS = ("elderly", "middle-aged", "adult", "young", "teenage", "child", "old")
_FEMALE_WORDS = ("woman", "women", "female", "girl", "mother", "wife", "maiden", "lady")
_CREATURE_WORDS = ("creature", "monster", "beast", "non-human", "golem", "spirit", "dragon", "giant")


def _character_clause(prompts, char_id):
    """取该角色单独出现（或作为首位）分镜的 image_prompt 开头外观描述。"""
    for p in prompts:
        if (p.get("focus_characters") or []) == [char_id]:
            return (p.get("image_prompt") or "").strip()
    for p in prompts:
        fc = p.get("focus_characters") or []
        if fc and fc[0] == char_id:
            return (p.get("image_prompt") or "").strip()
    return ""


def _derive_voice(desc):
    """由角色英文外观描述推导稳定音色（年龄 + 性别/类型），同一角色输出恒定不变。"""
    d = (desc or "").lower()
    if any(re.search(rf"\b{re.escape(w)}\b", d) for w in _CREATURE_WORDS):
        return "deep, rumbling, gravelly non-human voice, slow and heavy, low growl"
    age = "adult"
    # 优先匹配 "a/an <年龄>" 描述，避免 "years old" 里的 "old" 被误判为年龄
    m = re.search(r"\b(?:a|an)\s+(elderly|middle-aged|adult|young|teenage|child|old)\b", d)
    if m:
        age = m.group(1)
    else:
        for w in _AGE_WORDS:
            if re.search(rf"\b{re.escape(w)}\b", d):
                age = w
                break
    if any(re.search(rf"\b{re.escape(w)}\b", d) for w in _FEMALE_WORDS):
        gender = "female"
    else:
        gender = "male"
    return f"{age} {gender} voice, steady and clear, natural conversational tone"


def build_character_registry(prompts):
    """从 meta.json 的 prompts 收集角色，并由其外观描述推导稳定音色。

    返回 {char_id: {"clause": 角色英文外观描述, "voice": 稳定音色描述}}。
    供 --prompt_with_text 判断说话者并固定对白年龄/音色。
    """
    registry = {}
    for p in prompts:
        for cid in (p.get("focus_characters") or []):
            if cid in registry:
                continue
            clause = _character_clause(prompts, cid)[:160]
            registry[cid] = {"clause": clause, "voice": _derive_voice(clause)}
    return registry


_NARRATION_RE = re.compile(
    r"(?i)^\s*(?:[（(【\[(]?\s*(?:旁白|画面描述|叙述|解说|旁述|narration)\s*[）)】\]]?\s*[:：]?\s*)"
)


def _is_narration_label(text):
    """判断 DeepSeek 抽取结果是否为旁白/画面描述标签（而非人物对话）。

    DeepSeek 对旁白分镜本该返回空串，但常返回「旁白」「旁白：……」等标签，
    这里统一识别为旁白，避免标签被当成台词拼进 prompt 再由 LTX 原生音轨读出来。
    """
    return bool(_NARRATION_RE.match(text or ""))


def extract_spoken_text(image_prompt, text, prev_text="", char_map=None, prev_char=None):
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
    prev = prev_text if prev_text else "(无)"
    prev_c = prev_char or "(无)"
    user = (
        f"上一个分镜文本：{prev}\n"
        f"上一个分镜说话角色：{prev_c}\n"
        f"图像 prompt：{image_prompt}\n"
        f"本分镜中文文本：{text}"
    )
    spoken = _deepseek_chat(system, user, temperature=0.4, allow_empty=True)
    # 旁白/画面描述兜底：DeepSeek 有时不返回空串，而是输出「旁白」「旁白：……」等标签，
    # 这里直接判为无对话返回空串（后续 build_segment_prompts 仍会再兜底拦截一次）。
    if _is_narration_label(spoken):
        return ""
    return spoken


def _split_speech_by_frames(spoken, segs):
    """按句把完整对白/旁白切分为 len(segs) 段，每段词数按其帧长比例分配。"""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", spoken) if s.strip()]
    if not sentences:
        return [spoken] + [""] * (len(segs) - 1)
    seg_secs = [(nf - 1) / FPS for nf in segs]
    total_secs = sum(seg_secs) or 1.0
    total_words = sum(len(s.split()) for s in sentences)
    targets = [max(1, round(total_words * sec / total_secs)) for sec in seg_secs]
    targets[-1] += total_words - sum(targets)  # 最后一档吸收舍入误差
    chunks = []
    si = 0
    for tw in targets:
        bucket = []
        cur = 0
        while si < len(sentences) and cur < tw:
            bucket.append(sentences[si])
            cur += len(sentences[si].split())
            si += 1
        chunks.append(" ".join(bucket))
    if si < len(sentences):
        tail = " ".join(sentences[si:])
        chunks[-1] = " ".join(x for x in (chunks[-1], tail) if x)
    return chunks


def build_segment_prompts(image_prompt, spoken, segs, max_frames=MAX_FRAMES, char_voices=None):
    """把完整对白分配到各子段，返回与 segs 等长的逐段 prompt 列表。

    - 一句对白若能在一段（<= max_frames 帧）内说完：整段放入第一个子段，其余子段只保留 image_prompt。
    - 否则按句切分，依各子段帧长比例分配词数，各子段依次朗读，拼接后即完整对白。
    - 每个非空子段统一回填该说话角色的固定音色（char_voices），保证同一角色各分镜音色一致。
    - 旁白/画面描述在 extract 阶段已去除；此处再兜底拦截 "Narration:" 输出。
    """
    image_prompt = (image_prompt or "").strip()
    spoken = (spoken or "").strip()
    n = len(segs)
    if not spoken:
        return [image_prompt] * n

    char_voices = char_voices or {}

    # 旁白兜底：直接丢弃，只保留 image_prompt（正常流程 extract 不会返回旁白）
    if _is_narration_label(spoken):
        return [image_prompt] * n

    # 解析 "Dialogue: <角色id> says \"<英文台词>\""，取出角色 id 与台词，注入该角色固定音色
    m = re.match(r"(?is)^Dialogue\s*[:：]\s*(.+?)\s+says\b(.*)$", spoken)
    if m:
        char_id = m.group(1).strip()
        line = m.group(2).strip().strip('"“”').strip()
        voice = char_voices.get(char_id, "")
    else:
        char_id, line, voice = "", spoken, ""

    body = line
    word_count = len(body.split())
    speech_sec = word_count / SPEECH_WPS if SPEECH_WPS else word_count / 2.2
    one_seg_sec = max_frames / FPS

    chunks = [""] * n
    if n == 1 or speech_sec <= one_seg_sec:
        chunks[0] = body
    else:
        chunks = _split_speech_by_frames(body, segs)

    prefix = f"Dialogue ({voice}): " if voice else "Dialogue: "
    chunks = [prefix + c if c else "" for c in chunks]

    return [f"{image_prompt}\n\n{c}" if c else image_prompt for c in chunks]


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


def _parse_avoid_frames(value):
    """argparse 类型：解析帧数黑名单（英文/中文逗号分隔），如 '137' 或 '137,265'。"""
    out = set()
    for tok in re.split(r"[,，]", str(value or "")):
        tok = tok.strip()
        if not tok:
            continue
        try:
            n = int(tok)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"无法解析帧数: {tok!r}")
        if n < 9:
            raise argparse.ArgumentTypeError(f"帧数需 >= 9，收到 {n}")
        out.add(n)
    return out


def parse_shot_id(shot_id):
    """从 shot_id（如 scene3-shot2）解析出 (scene_num, shot_num)。"""
    parts = str(shot_id or "").split("-")

    def _num(s):
        m = re.search(r"\d+", s or "")
        return int(m.group()) if m else None

    scene_num = _num(parts[0]) if parts else None
    shot_num = _num(parts[1]) if len(parts) > 1 else None
    return scene_num, shot_num


def main():
    parser = argparse.ArgumentParser(description="按 meta.json 逐分镜生成 LTX-2.5 视频")
    parser.add_argument("meta", nargs="?", default="./work/meta.json",
                        help="meta.json 路径（默认 ./work/meta.json）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    parser.add_argument("--force", action="store_true", help="重新生成已存在的输出")
    parser.add_argument("--scene", type=_parse_num_arg, default=None,
                        help="只生成指定场景（数字，如 --scene 3）")
    parser.add_argument("--shot", type=_parse_num_arg, default=None,
                        help="只生成指定场景内的指定分镜（数字，需与 --scene 配合，如 --scene 3 --shot 2）")
    parser.add_argument("--max-frames", type=_parse_max_frames, default=MAX_FRAMES,
                        help=f"单段生成帧数上限，默认 {MAX_FRAMES}，最大 {MAX_FRAMES_LIMIT}（LTX 单段上限）")
    parser.add_argument("--frames-delta", type=int, default=0,
                        help="帧数偏移（单位：8 帧，可正可负），用于全局平移所有帧数，如 137->145 用 1、137->129 用 -1")
    parser.add_argument("--avoid-frames", type=_parse_avoid_frames, default=None,
                        help="帧数黑名单（逗号分隔），命中时向下 -8 绕开，如 137（只绕开该值，不影响其它帧数）")
    parser.add_argument("--retries", type=int, default=RETRY_ATTEMPTS,
                        help=f"单段生成失败时帧数 +8 重试的最大总尝试次数，默认 {RETRY_ATTEMPTS}（如 137 失败 -> 145 -> 153 ...）")
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--tts", action="store_true",
                        help="用 audios 配音替换原生音轨（默认保留 LTX 原生音轨；与 --prompt_with_text 互斥）")
    audio_group.add_argument("--prompt_with_text", action="store_true",
                        help="结合 image_prompt 与对应台词抽取人物对话（去掉旁白/画面描述），并按 meta.json 角色固定年龄/音色；能在一段内说完则整体放入，说不完则按句分配到各子段拼接（与 --tts 互斥）")
    args = parser.parse_args()

    meta_path = Path(args.meta).resolve()
    prompts, images, durations, audios, lines, key_frame_flags = load_meta(meta_path)
    base_dir = meta_path.parent
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 配音路径（与分镜一一对应，相对 meta.json 所在目录解析为绝对路径）
    audio_paths = [(base_dir / a).resolve() for a in audios]

    # 可选：按场景 / 分镜过滤要处理的分镜下标
    indices = list(range(len(prompts)))
    if args.scene is not None or args.shot is not None:
        kept = []
        for idx, p in enumerate(prompts):
            scene_num, shot_num = parse_shot_id(p.get("shot_id"))
            if args.scene is not None and scene_num != args.scene:
                continue
            if args.shot is not None and shot_num != args.shot:
                continue
            kept.append(idx)
        if not kept:
            raise RuntimeError("没有匹配指定 --scene / --shot 的分镜")
        indices = kept

    total_shots = len(indices)
    total_segments = 0
    for idx in indices:
        p = prompts[idx]
        dur = p.get("duration")
        if dur is None and idx < len(durations):
            dur = durations[idx]
        dur = float(dur)
        total_segments += len(frame_segments(dur, max_frames=args.max_frames, delta=args.frames_delta, avoid=args.avoid_frames))

    print(f"meta.json : {meta_path}")
    if args.scene is not None or args.shot is not None:
        cond = []
        if args.scene is not None:
            cond.append(f"场景 {args.scene}")
        if args.shot is not None:
            cond.append(f"分镜 {args.shot}")
        print(f"筛选条件 : {'，'.join(cond)}")
    print(f"分镜总数 : {total_shots}")
    print(f"视频段数 : {total_segments}（含切段，单段 <= {args.max_frames} 帧）")
    print(f"分镜目录 : {CLIPS_DIR}")
    print(f"结果视频 : {RESULT_PATH}")
    print(f"工作目录 : {WORKDIR}")
    print("=" * 70)

    # 角色音色注册表：--prompt_with_text 时由 meta.json 的角色外观描述推导，固定对白年龄/音色
    char_map = {}
    char_voices = {}
    if args.prompt_with_text:
        char_registry = build_character_registry(prompts)
        char_map = {cid: info["clause"] for cid, info in char_registry.items()}
        char_voices = {cid: info["voice"] for cid, info in char_registry.items()}
        if char_voices:
            print("角色音色 :")
            for cid, voice in char_voices.items():
                print(f"    {cid}  ->  {voice}")

    prev_scene_key = None
    prev_cont_image = None   # 上一个分镜最终视频的最后一帧
    prev_text = ""           # 上一个分镜的台词文本，供 --prompt_with_text 判断对话是否延续
    prev_char = None         # 上一个分镜的说话角色 id，供 --prompt_with_text 判断对话是否延续
    all_clips = []           # 按顺序收集每个分镜的最终视频，最后拼接成结果视频

    for pos, idx in enumerate(indices):
        i = idx              # 原始下标，用于 images/durations/audios 对齐
        p = prompts[idx]
        shot_idx = p.get("shot_idx", i)
        shot_id = p.get("shot_id", f"shot{shot_idx}")
        prompt = (p.get("image_prompt") or p.get("text") or "").strip()

        dur = p.get("duration")
        if dur is None and i < len(durations):
            dur = durations[i]
        dur = float(dur)
        segs = frame_segments(dur, max_frames=args.max_frames, delta=args.frames_delta, avoid=args.avoid_frames)

        # 可选：抽取人物对话，并按需分配到各子段（不按 duration 截断，供拼接后还原）
        seg_prompts = [prompt] * len(segs)
        if args.prompt_with_text:
            line_text = ""
            if i < len(lines):
                line_text = (lines[i] or "").strip()
            if not line_text:
                line_text = (p.get("text") or "").strip()
            if line_text:
                if args.dry_run:
                    preview = line_text[:60] + ("..." if len(line_text) > 60 else "")
                    seg_prompts = [
                        f"{prompt}\n\n[待抽取人物对话并分配到各子段: {preview}]"
                        if k == 0 else prompt
                        for k in range(len(segs))
                    ]
                else:
                    spoken = extract_spoken_text(
                        prompt, line_text, prev_text,
                        char_map=char_map, prev_char=prev_char,
                    )
                    seg_prompts = build_segment_prompts(
                        prompt, spoken, segs,
                        max_frames=args.max_frames, char_voices=char_voices,
                    )
                    # 记录本分镜说话角色，供下一分镜判断对话是否延续
                    m = re.match(r"(?is)^Dialogue\s*[:：]\s*(.+?)\s+says\b", spoken or "")
                    prev_char = m.group(1).strip() if m else None
                prev_text = line_text

        # 场景键：优先 scene_idx，其次取 shot_id 横线前半部分
        scene_key = p.get("scene_idx")
        if scene_key is None:
            scene_key = str(shot_id).split("-")[0] if "-" in str(shot_id) else str(shot_id)
        is_first_in_scene = (pos == 0) or (scene_key != prev_scene_key)

        # 关键帧判定：有 key_frame_flags 时按其标记决定起始图（True -> 用 images[i]，
        # False -> 续接上一分镜最后一帧）；无该字段时回退到「场景首镜用 images[i]」的旧逻辑。
        use_flags = len(key_frame_flags) == len(prompts)
        is_keyframe = bool(key_frame_flags[i]) if use_flags else is_first_in_scene

        # 起始图
        if is_keyframe:
            start_image = (base_dir / images[i]).resolve()
        else:
            start_image = Path(prev_cont_image)

        # 非关键帧兜底：续接用的末帧检测不到人脸、且本分镜有焦点角色时，
        # 临时按关键帧处理，改用 images[i] 作为起始图（避免从无人脸的末帧续接）。
        if not is_keyframe and (p.get("focus_characters") or []):
            has_face = _detect_face(start_image)
            if has_face is False:
                start_image = (base_dir / images[i]).resolve()
                print(f"    {shot_id} 末帧无人脸且 focus_characters 非空，"
                      f"临时按关键帧处理 -> {start_image.name}", flush=True)

        final_output = CLIPS_DIR / f"output_{shot_idx}.mp4"
        all_clips.append(final_output)

        # 续跑：最终视频已存在且非空，且未指定 --force，则跳过该分镜
        if (not args.dry_run and not args.force
                and final_output.is_file() and final_output.stat().st_size > 0):
            cont_path = WORK_DIR / f"output_{shot_idx}_last.png"
            if not cont_path.is_file():
                extract_last_frame(final_output, cont_path)
            prev_cont_image = cont_path
            prev_scene_key = scene_key
            print(f"[{pos+1}/{total_shots}] {shot_id} 已存在，跳过", flush=True)
            continue

        seg_videos = []
        seg_image = start_image
        any_generated = False
        for k, nf in enumerate(segs):
            if len(segs) == 1:
                seg_output = final_output
            else:
                seg_output = CLIPS_DIR / f"output_{shot_idx}_seg{k}.mp4"
            seg_exists = _exists_nonempty(seg_output)
            generated = False
            if args.dry_run:
                command = build_command(seg_prompts[k], seg_image, nf, seg_output)
                print(f"[{pos+1}/{total_shots}] {shot_id} 段{k+1}/{len(segs)} "
                      f"frames={nf} start={seg_image.name}")
                print("   " + " ".join(command))
            elif seg_exists and not args.force:
                print(f"[{pos+1}/{total_shots}] {shot_id} 段{k+1}/{len(segs)} "
                      f"已存在，跳过（{seg_output.name}）", flush=True)
            else:
                log_path = WORK_DIR / "logs" / f"output_{shot_idx}_seg{k}.log"
                print(f"[{pos+1}/{total_shots}] {shot_id} 段{k+1}/{len(segs)} "
                      f"frames={nf} 生成中 ...", flush=True)
                _generate_with_retry(seg_prompts[k], seg_image, nf, seg_output, log_path,
                                     f"{shot_id} 段{k+1}/{len(segs)}", retries=args.retries,
                                     max_frames=args.max_frames)
                generated = True
                any_generated = True

            seg_videos.append(seg_output)

            # 下一段起始图 = 本段最后一帧（本段重新生成或帧缺失时才补抽）
            if k < len(segs) - 1:
                seg_last = WORK_DIR / f"output_{shot_idx}_seg{k}_last.png"
                if not args.dry_run and (generated or not _exists_nonempty(seg_last)):
                    extract_last_frame(seg_output, seg_last)
                seg_image = seg_last

        # 多段合并为最终分镜视频
        merged = False
        if len(segs) > 1 and not args.dry_run:
            if _exists_nonempty(final_output) and not args.force:
                print(f"    合并结果已存在，跳过 -> {final_output.name}", flush=True)
            else:
                print(f"    合并 {len(segs)} 段 -> {final_output.name}", flush=True)
                concat_videos(seg_videos, final_output, keep_audio=True)
                merged = True

        # 抽最终视频最后一帧，供同场景下一个分镜续接（最终视频被重建或续接帧缺失时才补抽）
        cont_path = WORK_DIR / f"output_{shot_idx}_last.png"
        prev_cont_image = cont_path
        if not args.dry_run:
            if any_generated or merged or not _exists_nonempty(cont_path):
                extract_last_frame(final_output, cont_path)
            print(f"    完成 {final_output.name}，续接帧 {cont_path.name}", flush=True)

        prev_scene_key = scene_key

    if args.dry_run:
        print("=" * 70)
        print("以上为 --dry-run 计划，未执行任何生成。")
    else:
        print("=" * 70)
        print(f"全部 {total_shots} 个分镜处理完成，分镜目录：{CLIPS_DIR}")

        # 先拼结果视频（保留原生音轨；--tts 时再替换为配音）
        video_only = WORK_DIR / "result_video.mp4"
        print(f"正在拼接结果视频 -> {video_only} ...", flush=True)
        concat_videos(all_clips, video_only, keep_audio=True)

        if args.tts:
            # 只取本次筛选范围内的配音，保证与视频顺序一一对应
            sel_audio = [audio_paths[idx] for idx in indices]
            if sel_audio and len(sel_audio) == len(all_clips):
                audio_concat = WORK_DIR / "result_audio.m4a"
                print(f"正在拼接 {len(sel_audio)} 段配音 -> {audio_concat} ...", flush=True)
                concat_audios(sel_audio, audio_concat)
                print(f"正在合成配音 -> {RESULT_PATH} ...", flush=True)
                mux_audio(video_only, audio_concat, RESULT_PATH)
            else:
                print("--tts 已指定，但未找到与分镜一一对应的配音，输出原生音轨结果。", flush=True)
                os.replace(str(video_only), str(RESULT_PATH))
        else:
            os.replace(str(video_only), str(RESULT_PATH))

        print(f"结果视频已生成：{RESULT_PATH}")


if __name__ == "__main__":
    main()
