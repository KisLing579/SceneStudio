#!/usr/bin/env python3
"""Project-scoped key-frame and video editor. Run: python scene_studio.py."""
import argparse
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps
from werkzeug.exceptions import HTTPException

from prompt_editor import generate_image, _config_size, save_meta
from key_scene_editor import concat_with_fades
from generate_key_scenes import resolve_project_image

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)
app.config.update(PROJECT_ROOT=os.environ.get("PROJECT_ROOT", "/root/project"),
                  MERGE_OFFLINE=False, MAX_CONTENT_LENGTH=4 * 1024 * 1024)
locks = {}
jobs = {}
generation_requests = {}
project_paths = {}
guard = threading.RLock()


def project(project_id):
    with guard:
        registered = project_paths.get(project_id)
    if registered is not None:
        if not (registered / "meta.json").is_file():
            raise ValueError("项目不存在或缺少 meta.json")
        return registered
    if not isinstance(project_id, str) or not re.fullmatch(r"[\w-]+", project_id):
        raise ValueError("project_id 只能包含字母、数字、下划线和连字符")
    root = Path(app.config["PROJECT_ROOT"]).resolve()
    directory = (root / project_id).resolve()
    if not directory.is_relative_to(root) or not (directory / "meta.json").is_file():
        raise ValueError("项目不存在或缺少 meta.json")
    return directory


def read_meta(directory):
    data = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    return data, data.get("task_data", data)


def entries(data):
    return data.get("frame_secene_prompt") or []


def key(entry):
    return int(entry["frame_image_idx"][0])


def image_path(directory, value):
    return resolve_project_image(directory, value)


def video_path(directory, index):
    return directory / "key_scene_out" / f"output_kf{index}.mp4"


def media_url(project_id, kind, index, path=None):
    version = path.stat().st_mtime_ns if path and path.is_file() else 0
    return f"/api/projects/{project_id}/media/{kind}/{index}?v={version}"


def snapshot(project_id, directory):
    data, task = read_meta(directory)
    images = task.get("images") or []
    prompts = task.get("prompts") or []
    groups = []
    for entry in entries(data):
        idx = key(entry)
        if not 0 <= idx < min(len(images), len(prompts)):
            raise ValueError(f"关键帧下标越界：{idx}")
        video = video_path(directory, idx)
        try:
            img = image_path(directory, images[idx])
        except ValueError:
            img = None
        groups.append(dict(index=idx, scene_id=entry.get("scene_id"),
                           image_prompt=prompts[idx].get("image_prompt", ""),
                           video_prompt=entry.get("prompt", ""),
                           image=media_url(project_id, "image", idx, img),
                           generation_started=idx in generation_requests.get(project_id, set()),
                           video=media_url(project_id, "video", idx, video)
                           if video.is_file() and video.stat().st_size else None))
    output = directory / "key_scene_out" / "concat_fade.mp4"
    with guard:
        job = dict(jobs.get(project_id, {"status": "idle"}))
    return dict(ok=True, project_id=project_id, project_directory=str(directory), groups=groups, job=job,
                thumbnails=[media_url(project_id, "thumb", i) for i in range(len(images))],
                result=media_url(project_id, "result", 0, output) if output.is_file() else None)


def save_edits(directory, edits):
    data, task = read_meta(directory)
    by_key = {key(e): e for e in entries(data)}
    prompts = task.get("prompts") or []
    configs = task.get("images_configs") or []
    for edit in edits:
        idx = int(edit["index"])
        if idx not in by_key or not 0 <= idx < len(prompts):
            raise ValueError("无效关键帧")
        for field in ("image_prompt", "video_prompt"):
            if not isinstance(edit.get(field), str) or not edit[field].strip():
                raise ValueError("提示词不能为空")
        prompts[idx]["image_prompt"] = edit["image_prompt"]
        if idx < len(configs) and isinstance(configs[idx], dict):
            configs[idx]["prompt"] = edit["image_prompt"]
        by_key[idx]["prompt"] = edit["video_prompt"]
    save_meta(directory / "meta.json", data)


def run_script(project_id, script, *args):
    command = [sys.executable, "-u", str(HERE / script), *map(str, args)]
    with subprocess.Popen(command, cwd=project(project_id), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace") as process:
        for line in process.stdout:
            with guard:
                jobs[project_id]["log"] = (jobs[project_id]["log"] + line)[-12000:]
        if process.wait():
            raise RuntimeError(f"{script} 执行失败，请查看任务日志")


def start_job(project_id, directory, action, payload):
    with guard:
        lock = locks.setdefault(project_id, threading.Lock())
        if not lock.acquire(blocking=False):
            return jsonify(ok=False, error="该项目有任务正在执行"), 409
    try:
        max_frames = int(payload.get("max_frames", 480))
        delta = int(payload.get("frames_delta", 3))
        prompt_with_text = payload.get("prompt_with_text", True)
        language = payload.get("language", "zh")
        if not isinstance(prompt_with_text, bool):
            raise ValueError("对白选项必须为布尔值")
        if language not in ("zh", "en"):
            raise ValueError("对白语言必须为 zh 或 en")
        if not 9 <= max_frames <= 480:
            raise ValueError("max-frames 必须在 9 到 480 之间")
        data, _ = read_meta(directory)
        by_key = {key(e): e for e in entries(data)}
        idx = int(payload.get("index", -1))
        if action in ("image", "video") and idx not in by_key:
            raise ValueError("无效关键帧")
        if action == "batch" and not by_key:
            raise ValueError("请先创建关键场景帧")
        selected = payload.get("selected") or []
        if action == "concat":
            if not selected or any(i not in by_key for i in selected):
                raise ValueError("请选择有效的场景视频段")
            selected = [i for i in by_key if i in selected]
            if any(not video_path(directory, i).is_file() for i in selected):
                raise ValueError("选中的场景中存在尚未生成的视频段")
        save_edits(directory, payload.get("edits") or [])
        with guard:
            jobs[project_id] = dict(id=uuid.uuid4().hex, status="running", action=action, log="", error="")
            if action in ("batch", "video"):
                generation_requests.setdefault(project_id, set()).update(by_key if action == "batch" else [idx])
    except Exception:
        lock.release()
        raise

    offline = app.config["MERGE_OFFLINE"]

    def work():
        try:
            meta = directory / "meta.json"
            if action == "create":
                # Merge on a sibling staging file; publish only after both scripts succeed.
                staging = directory / f".studio-{uuid.uuid4().hex}.json"
                try:
                    data, _ = read_meta(directory)
                    data.pop("frame_secene_prompt", None)
                    save_meta(staging, data)
                    run_script(project_id, "generate_key_frame_flags.py", staging)
                    run_script(project_id, "merge_frame_scene.py", staging, *(["--offline"] if offline else []))
                    staging.replace(meta)
                finally:
                    staging.unlink(missing_ok=True)
            elif action == "image":
                data, task = read_meta(directory)
                prompt = task["prompts"][idx]["image_prompt"]
                image = generate_image(prompt, directory, _config_size(task.get("images_configs") or [], idx))
                task["images"][idx] = "./" + image.name
                save_meta(meta, data)
            elif action in ("batch", "video"):
                args = [meta, "--max-frames", max_frames, "--frames-delta", delta,
                        "--out-dir", directory / "key_scene_out", "--work-dir", directory / "key_scene_work",
                        "--no-concat", "--force", "--language", language,
                        "--prompt_with_text" if prompt_with_text else "--no-prompt-with-text"]
                if action == "video":
                    args += ["--scene", by_key[idx]["scene_id"], "--key-frame", idx]
                run_script(project_id, "generate_key_scenes.py", *args)
            elif action == "concat":
                # Stage concatenation so browsers never read an incomplete output.
                import tempfile
                import shutil
                out = directory / "key_scene_out"
                with tempfile.TemporaryDirectory(prefix="concat_", dir=out) as temp:
                    temp = Path(temp)
                    names = []
                    for i in selected:
                        src = video_path(directory, i)
                        shutil.copy2(src, temp / src.name)
                        names.append(src.name)
                    result = Path(concat_with_fades(temp, names))
                    result.replace(out / "concat_fade.mp4")
            with guard:
                jobs[project_id]["status"] = "done"
        except Exception as exc:
            with guard:
                jobs[project_id].update(status="error", error=str(exc))
        finally:
            lock.release()

    threading.Thread(target=work, daemon=True).start()
    return jsonify(ok=True), 202


@app.get("/")
def index():
    return send_from_directory(HERE, "scene_studio.html")


@app.get("/api/projects/<project_id>")
def get_project(project_id):
    return jsonify(snapshot(project_id, project(project_id)))


@app.post("/api/load")
def load_project():
    payload = request.get_json() or {}
    value = payload.get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("请输入 Project ID 或绝对路径")
    value = value.strip()
    mode = payload.get("mode", "id")
    if mode == "id":
        directory = project(value)
    elif mode == "path":
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("请输入服务器上的绝对路径")
        path = path.resolve()
        directory = path.parent if path.name == "meta.json" and path.is_file() else path
        if not directory.is_dir() or not (directory / "meta.json").is_file():
            raise ValueError("该路径不是项目目录或其中缺少 meta.json")
    else:
        raise ValueError("未知项目输入模式")
    # Both input modes share a canonical identity, task lock, and media routes.
    token = "path-" + uuid.uuid5(uuid.NAMESPACE_URL, os.path.normcase(str(directory))).hex
    with guard:
        project_paths[token] = directory
    return jsonify(snapshot(token, directory))


@app.post("/api/projects/<project_id>/<action>")
def mutate(project_id, action):
    directory = project(project_id)
    if action not in {"create", "image", "video", "batch", "concat", "save"}:
        return jsonify(ok=False, error="未知操作"), 404
    payload = request.get_json() or {}
    if action != "save":
        return start_job(project_id, directory, action, payload)
    with guard:
        lock = locks.setdefault(project_id, threading.Lock())
        if not lock.acquire(blocking=False):
            return jsonify(ok=False, error="该项目有任务正在执行"), 409
    try:
        save_edits(directory, payload.get("edits") or [])
    finally:
        lock.release()
    return jsonify(ok=True)


@app.get("/api/projects/<project_id>/media/<kind>/<int:idx>")
def media(project_id, kind, idx):
    directory = project(project_id)
    if kind in ("image", "thumb"):
        _, task = read_meta(directory)
        images = task.get("images") or []
        if idx >= len(images):
            raise ValueError("图像下标越界")
        path = image_path(directory, images[idx])
        if kind == "thumb":
            with Image.open(path) as original:
                original.thumbnail((192, 128))
                small = ImageOps.exif_transpose(original).convert("RGB")
                buffer = io.BytesIO()
                small.save(buffer, "JPEG", quality=75)
            buffer.seek(0)
            return send_file(buffer, mimetype="image/jpeg", max_age=0)
    elif kind == "video":
        data, _ = read_meta(directory)
        if idx not in {key(e) for e in entries(data)}:
            raise ValueError("无效视频段")
        path = video_path(directory, idx)
    elif kind == "result":
        path = directory / "key_scene_out" / "concat_fade.mp4"
    else:
        return jsonify(ok=False, error="未知媒体类型"), 404
    return send_file(path, conditional=True, max_age=0)


@app.errorhandler(Exception)
def error(exc):
    status = exc.code if isinstance(exc, HTTPException) else 400 if isinstance(exc, (ValueError, KeyError, TypeError)) else 500
    return jsonify(ok=False, error=str(exc)), status


def main():
    parser = argparse.ArgumentParser(description="关键场景工作台")
    parser.add_argument("--project-root", default=app.config["PROJECT_ROOT"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7200)
    parser.add_argument("--offline", action="store_true", help="融合时跳过 DeepSeek 对白抽取")
    args = parser.parse_args()
    app.config.update(PROJECT_ROOT=args.project_root, MERGE_OFFLINE=args.offline)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
