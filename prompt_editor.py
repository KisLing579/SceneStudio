#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""即梦图像 prompt 编辑与重抽的前端服务。

提供网页：输入 meta.json 路径 -> 展示 images 指向的所有图像，每张图旁边一个可编辑的
prompt 输入框；可「保存」修改后的 prompt 到 meta.json，或「重抽」——用火山引擎(即梦)
按输入框里的 prompt 重新生成图像，并把新图像路径与 prompt 一并写回 meta.json。

重抽实现参考 Mindawaker/app/image_engine/volc_engine.py（VisualService + jimeng t2i）。

用法：
  python prompt_editor.py [--host 127.0.0.1] [--port 7000]

依赖：
  flask、volcengine；重抽需要火山引擎凭证环境变量 VOLC_ACCESS_KEY / VOLC_SECRET_KEY
  （或 IMAGE_MODEL_PASSWORD=AccessKey**SecretKey）。
"""

import base64
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

# Windows 控制台统一 UTF-8 输出，避免中文报错
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, jsonify, request, send_file, send_from_directory

SCRIPT_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)

DEFAULT_SIZE = "1024*1024"
REQ_KEY = os.environ.get("VOLC_REQ_KEY", "jimeng_t2i_v40")


# --------------------------------------------------------------------------- #
# 火山引擎凭证（与 Mindawaker app/configs/credentials.py 的 volc_credentials 一致）
# --------------------------------------------------------------------------- #
def _volc_credentials():
    access_key = os.environ.get("VOLC_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("VOLC_SECRET_KEY", "").strip()
    if access_key and secret_key:
        return access_key, secret_key
    password = os.environ.get("IMAGE_MODEL_PASSWORD", "").strip()
    if password:
        parts = password.split("**")
        if len(parts) == 2 and all(p.strip() for p in parts):
            return access_key or parts[0].strip(), secret_key or parts[1].strip()
        raise RuntimeError(
            "IMAGE_MODEL_PASSWORD 格式应为 AccessKey**SecretKey，或分别设置 VOLC_ACCESS_KEY / VOLC_SECRET_KEY"
        )
    if not access_key:
        raise RuntimeError("缺少火山引擎凭证：请设置环境变量 VOLC_ACCESS_KEY / VOLC_SECRET_KEY")
    raise RuntimeError("缺少火山引擎凭证：请设置环境变量 VOLC_SECRET_KEY")


def _parse_size(size):
    m = re.match(r"^\s*(\d+)\s*[\*xX]\s*(\d+)\s*$", str(size or ""))
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        if 0 < w <= 4096 and 0 < h <= 4096:
            return w, h
    w, h = (int(x) for x in DEFAULT_SIZE.split("*"))
    return w, h


def generate_image(prompt, output_dir, size=DEFAULT_SIZE):
    """用火山引擎即梦 t2i 生成一张图，保存到 output_dir，返回绝对路径。"""
    from volcengine import visual
    from volcengine.visual.VisualService import VisualService

    access_key, secret_key = _volc_credentials()
    visual_service = VisualService()
    visual_service.set_ak(access_key)
    visual_service.set_sk(secret_key)

    width, height = _parse_size(size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"jimeng_img_{int(time.time() * 1000)}_0.png"

    form = {
        "req_key": REQ_KEY,
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": random.randint(0, 2147483647),
    }

    resp = visual_service.cv_process(form)

    img_base64 = resp["data"]["binary_data_base64"][0]

    # 解码
    img_bytes = base64.b64decode(img_base64)

    # 保存为文件
    with open(output_path, "wb") as f:
        f.write(img_bytes)

    print("保存成功：output.png")

    return output_path


# --------------------------------------------------------------------------- #
# meta.json 读写
# --------------------------------------------------------------------------- #
def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _resolve_meta(path):
    path = (path or "").strip()
    if not path:
        raise ValueError("meta.json 路径不能为空")
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        p = p / "meta.json"
    if not p.is_file():
        raise ValueError(f"meta.json 不存在：{p}")
    return str(p)


def _task_data(path):
    data = load_meta(path)
    task = data.get("task_data") or data
    return data, task


def _resolve_image(base_dir, rel):
    # 只取 images 里的文件名，落到 meta.json 所在目录下
    name = Path(str(rel or "").replace("\\", "/")).name
    return (base_dir / name).resolve()


def _image_src(meta_path, index, rel):
    v = Path(str(rel or "")).name or str(index)
    return f"/api/image?meta={quote(meta_path)}&index={index}&v={quote(v)}"


def _config_size(configs, index):
    if index < len(configs) and isinstance(configs[index], dict):
        s = str(configs[index].get("size") or "").strip()
        if s:
            return s
    return DEFAULT_SIZE


def _scene_key(p, i):
    """场景唯一键：优先 scene_idx，其次 scene_id，最后从 shot_id 解析。"""
    k = p.get("scene_idx")
    if k is None:
        k = p.get("scene_id")
    if k is None:
        sid = str(p.get("shot_id") or "")
        k = sid.split("-")[0] if "-" in sid else (sid or i)
    return k


# --------------------------------------------------------------------------- #
# 前端
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(str(SCRIPT_DIR), "prompt_editor.html")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.route("/api/load", methods=["POST"])
def api_load():
    payload = request.get_json(force=True, silent=True) or {}
    meta_path = _resolve_meta(payload.get("path"))
    _, task = _task_data(meta_path)
    prompts = task.get("prompts") or []
    images = task.get("images") or []
    base_dir = Path(meta_path).parent

    flags = task.get("key_frame_flags") or []
    use_flags = len(flags) == len(prompts)

    shots = []
    seen_scenes = set()
    for i, p in enumerate(prompts):
        scene_key = _scene_key(p, i)
        if use_flags:
            # 关键帧标记为 True 的分镜才显示
            if not flags[i]:
                continue
        elif scene_key in seen_scenes:
            # 回退：无 key_frame_flags 时，每场景只显示第一个分镜
            continue
        seen_scenes.add(scene_key)
        rel = images[i] if i < len(images) else ""
        shots.append({
            "index": i,
            "shot_id": p.get("shot_id"),
            "scene_id": p.get("scene_id"),
            "scene_key": str(scene_key),
            "prompt": p.get("image_prompt") or p.get("text") or "",
            "image_path": rel,
            "image_src": _image_src(meta_path, i, rel),
        })
    return jsonify({
        "ok": True,
        "meta_path": meta_path,
        "shots": shots,
        "scene_count": len(shots),
        "total_shots": len(prompts),
    })


@app.route("/api/image")
def api_image():
    meta_path = _resolve_meta(request.args.get("meta"))
    try:
        index = int(request.args.get("index", "0"))
    except (TypeError, ValueError):
        return "bad index", 400
    _, task = _task_data(meta_path)
    images = task.get("images") or []
    if not (0 <= index < len(images)):
        return "bad index", 400
    img_path = _resolve_image(Path(meta_path).parent, images[index])
    if not img_path.is_file():
        return "image not found", 404
    return send_file(str(img_path), mimetype="image/png", max_age=0)


@app.route("/api/save", methods=["POST"])
def api_save():
    payload = request.get_json(force=True, silent=True) or {}
    meta_path = _resolve_meta(payload.get("path"))
    index = int(payload.get("index", -1))
    prompt = payload.get("prompt") or ""
    data, task = _task_data(meta_path)
    prompts = task.get("prompts") or []
    configs = task.get("images_configs") or []
    if not (0 <= index < len(prompts)):
        raise ValueError(f"index 越界：{index}")
    prompts[index]["image_prompt"] = prompt
    if index < len(configs) and isinstance(configs[index], dict):
        configs[index]["prompt"] = prompt
    save_meta(meta_path, data)
    return jsonify({"ok": True, "index": index})


@app.route("/api/save_all", methods=["POST"])
def api_save_all():
    payload = request.get_json(force=True, silent=True) or {}
    meta_path = _resolve_meta(payload.get("path"))
    items = payload.get("items") or []
    data, task = _task_data(meta_path)
    prompts = task.get("prompts") or []
    configs = task.get("images_configs") or []
    saved = 0
    for it in items:
        i = int(it.get("index", -1))
        prompt = it.get("prompt") or ""
        if 0 <= i < len(prompts):
            prompts[i]["image_prompt"] = prompt
            if i < len(configs) and isinstance(configs[i], dict):
                configs[i]["prompt"] = prompt
            saved += 1
    save_meta(meta_path, data)
    return jsonify({"ok": True, "saved": saved})


@app.route("/api/regenerate", methods=["POST"])
def api_regenerate():
    payload = request.get_json(force=True, silent=True) or {}
    meta_path = _resolve_meta(payload.get("path"))
    index = int(payload.get("index", -1))
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt 不能为空")

    data, task = _task_data(meta_path)
    prompts = task.get("prompts") or []
    images = task.get("images") or []
    configs = task.get("images_configs") or []
    if not (0 <= index < len(prompts)):
        raise ValueError(f"index 越界：{index}")
    if index >= len(images):
        raise ValueError("images 列表长度不足")

    base_dir = Path(meta_path).parent
    size = _config_size(configs, index)
    new_abs = generate_image(prompt, base_dir, size=size)
    rel = "./" + new_abs.name

    images[index] = rel
    prompts[index]["image_prompt"] = prompt
    if index < len(configs) and isinstance(configs[index], dict):
        configs[index]["prompt"] = prompt
    save_meta(meta_path, data)

    return jsonify({
        "ok": True,
        "index": index,
        "prompt": prompt,
        "image_path": rel,
        "image_src": _image_src(meta_path, index, rel),
    })


@app.errorhandler(ValueError)
def _handle_value_error(e):
    return jsonify({"ok": False, "error": str(e)}), 400


@app.errorhandler(Exception)
def _handle_generic_error(e):
    return jsonify({"ok": False, "error": str(e)}), 500


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main():
    import argparse
    parser = argparse.ArgumentParser(description="即梦图像 prompt 编辑与重抽前端")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=7000, help="监听端口（默认 7000）")
    args = parser.parse_args()
    print(f"请在浏览器打开 http://{args.host}:{args.port}")
    print("提示：重抽需设置 VOLC_ACCESS_KEY / VOLC_SECRET_KEY（或 IMAGE_MODEL_PASSWORD）。")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
