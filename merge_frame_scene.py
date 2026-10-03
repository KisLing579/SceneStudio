#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_frame_scene.py — 在关键帧（key frame）之间合并分镜提示词，生成视频段级提示。

功能：
  1. 读取 meta.json 的 task_data.prompts 与 task_data.key_frame_flags。key_frame_flags
     为与 prompts 等长的布尔数组（由 generate_key_frame_flags.py 生成）：True 表示该分镜
     使用自己的首图（images[i]）开启新的一段视频，False 表示紧接上一分镜的最后一帧续跑。
  2. 按关键帧切段：每段从某个 True（关键帧）开始，直到下一个 True 之前为止。每段即一个
     连续生成的视频段——一个 scene_id 可能切出多个视频段（有新角色进场即新关键帧），因此
     不再按 scene_id 一条对应一个视频。
  3. 对每一段，把该段内各分镜合并成【一条】段级描述：
       - prompt（英文视觉 prompt）：确定性合并。重复出现的人物外观、物体/道具、场景环境、
         固定风格后缀只出现一次（按子句去重），之后按分镜顺序追加各分镜独有的动作/表情/
         镜头运动/光线/mood 等变化部分。
       - text（中文原文）：按分镜顺序拼接的旁白与人物对话原文（离线参考，不标注说话者）。
       - dialogue（对白）：把整篇文章统一抽取的【人物对白】按关键帧段分散到各段，每行格式
         「角色英文名：对白」。角色英文名取自 character_specs.json 的 name 字段；连续对白
         缺失的说话者由 DeepSeek 结合角色名单推断补齐。--offline 时 dialogue 为空列表。
  4. 生成 frame_secene_prompt 列表（每段一条，含 scene_id / world_scene_id /
     frame_image_idx / text / dialogue / prompt / duration），追加到 meta.json 末尾（顶层键）。
     另生成 frame_secene_characters（角色 id -> 外观/推导音色 + 英文名），供下游保证同一
     角色音色统一、并按英文名找到角色的外观描述（用于对白融合）。

角色音色推导复用 generate_shots.py 的 build_character_registry / _derive_voice，与
--prompt_with_text 保持一致（同一角色恒定音色）。角色英文名/外观来自 character_specs.json
（与 meta.json 所在目录的 artifacts/story 下）。

用法：
  python merge_frame_scene.py [meta.json 路径] [--dry-run] [--offline] [--scene N] [--shot M]

  --dry-run  只打印计划（不调用 DeepSeek、不写文件）。
  --offline  不调用 DeepSeek，无对白抽取（dialogue 为空；prompt 合并不受影响）。
  --scene N  只合并指定场景（数字，如 --scene 3）。
  --shot M   只合并指定场景内的指定分镜（需与 --scene 配合，用于测试）。

环境变量：
  DEEPSEEK_API_KEY  全局对白抽取（说话者识别）所需（默认流程必需；--offline 时不需要）。
  DEEPSEEK_MODEL    可选，默认 deepseek-chat。
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

# Windows 控制台统一 UTF-8 输出，避免中文报错
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# 复用 generate_shots.py 的角色音色推导与 DeepSeek 调用，保证同一角色音色一致
from generate_shots import (
    build_character_registry,
    _deepseek_chat,
    parse_shot_id,
)

# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
_CHAPTER_RE = re.compile(r"^第\s*\d+\s*章[^\n]*\n*")
_STYLE_MARKER = "highly detailed"  # 每个分镜 prompt 末尾的固定风格后缀起点


def _clean_chapter(text):
    """去掉章节标记（如「第 1 章（已批准）」）与首尾空白。"""
    t = (text or "").strip()
    t = _CHAPTER_RE.sub("", t)
    return t.strip()


def _norm_clause(c):
    """子句归一化（小写 + 去掉非字母数字），用于跨分镜去重比较。"""
    return re.sub(r"[^a-z0-9]+", "", (c or "").lower())


def _strip_style(prompt):
    """去掉 prompt 末尾的固定风格后缀（highly detailed ... 之后的部分）。"""
    idx = (prompt or "").lower().find(_STYLE_MARKER)
    if idx != -1:
        return prompt[:idx].rstrip().rstrip(",").strip()
    return (prompt or "").strip()


def _style_suffix(prompts):
    """取第一个带风格后缀的 prompt 里的后缀文本（各分镜一致）。"""
    for p in prompts:
        p = (p or "").strip()
        idx = p.lower().find(_STYLE_MARKER)
        if idx != -1:
            return p[idx:].strip()
    return ""


# --------------------------------------------------------------------------- #
# 合并逻辑
# --------------------------------------------------------------------------- #
def _merge_prompt_deterministic(shots):
    """确定性合并英文视觉 prompt：子句级去重 + 风格后缀只出现一次。

    每个分镜 prompt 按逗号切成子句，跨分镜去重（首次出现的子句保留，后续重复跳过），
    这样人物外观、场景环境、物体描述等重复内容只出现一次，各分镜独有的动作/镜头/光线/
    mood 按顺序追加；末尾固定的「highly detailed ...」风格后缀只保留一份放到最后。
    """
    prompts = [(s.get("image_prompt") or "").strip() for s in shots]
    style = _style_suffix(prompts)

    seen = set()
    shot_deltas = []
    for p in prompts:
        t = _strip_style(p)
        delta = []
        for c in (x.strip() for x in t.split(",") if x.strip()):
            n = _norm_clause(c)
            if n and n not in seen:
                delta.append(c)
                seen.add(n)
        shot_deltas.append(delta)

    body = "\n\n".join(", ".join(d) for d in shot_deltas if d).strip()
    if style:
        body = (body + "\n\n" + style).strip()
    return body


def load_character_list(meta_path):
    """读取 meta.json 所在目录 artifacts/story 下的 character_specs.json。

    返回 {character_id: {name, description}}。name 取 specs 里的英文 name 字段
    （如 "Su Qingyan"），description 取 visual_core（缺失时退化为 description）。
    文件缺失时抛出 RuntimeError。
    """
    char_file = Path(meta_path).parent / "artifacts" / "story" / "character_specs.json"
    if not char_file.is_file():
        raise RuntimeError(f"找不到 character_specs.json：{char_file}")
    specs = json.loads(char_file.read_text(encoding="utf-8"))
    out = {}
    for s in specs:
        cid = s.get("character_id")
        if not cid:
            continue
        out[cid] = {
            "name": (s.get("name") or cid).strip(),
            "description": (s.get("visual_core") or s.get("description") or "").strip(),
        }
    return out


def extract_dialogue_global(prompts, char_list):
    """把整篇文章（按【分镜N】标记）统一交给 DeepSeek，抽取全部人物对白并标注说话者。

    返回 [(shot_idx, 角色英文名, 对白原文), ...]。连续对白缺失的说话者由 DeepSeek 结合
    角色名单（英文名 + 外观）与上下文推断补齐。
    """
    char_lines = "\n".join(
        f"- {info['name']}: {info['description']}"
        for info in char_list.values()
    )
    marked = "\n".join(
        f"【分镜{i}】{_clean_chapter(p.get('text') or '')}"
        for i, p in enumerate(prompts)
    )
    system = (
        "你是影视对白抽取助手。给定一整篇文章（按【分镜N】切分，N 从 0 开始）和一份角色名单，"
        "从全文中抽取所有【人物对白】（角色直接说出口的话，含引号内直接引语），忽略旁白、"
        "画面描写、心理活动、动作描写。要求：\n"
        "1. 每句对白单独一行，格式：\n"
        "   【分镜N】角色英文名：对白原文\n"
        "2. 角色英文名必须从给定角色名单中原样照抄，不得自创、不得翻译成中文；\n"
        "3. 连续对白常缺少说话者，必须结合上下文与角色关系推断出说话者并补上角色名；\n"
        "4. 对白原文照抄文章原文，不要改写、不要翻译成英文；\n"
        "5. 只输出对白行，不要任何解释、不要 markdown、不要多余编号。"
    )
    user = (
        "角色名单（角色英文名 + 外观）：\n"
        f"{char_lines}\n\n"
        "文章（按【分镜N】切分）：\n"
        f"{marked}"
    )
    out = _deepseek_chat(system, user, temperature=0.3)
    return _parse_dialogue_lines(out, len(prompts), char_list)


def _parse_dialogue_lines(out, n_shots, char_list):
    """解析 DeepSeek 输出，每行「【分镜N】角色名：对白」-> (shot_idx, name, dialogue)。"""
    results = []
    for line in (out or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^【\s*分镜\s*(\d+)\s*】\s*(.+)$", line)
        if not m:
            continue
        shot = int(m.group(1))
        if not (0 <= shot < n_shots):
            continue
        rest = m.group(2).strip()
        sep = re.search(r"[:：]\s*", rest)
        if sep:
            name = rest[:sep.start()].strip()
            dialogue = rest[sep.end():].strip()
        else:
            name, dialogue = "", rest
        if not dialogue:
            continue
        name = _resolve_name(name, char_list)
        if name:
            results.append((shot, name, dialogue))
    return results


def _resolve_name(raw, char_list):
    """把 DeepSeek 输出的说话者名规范化为角色名单里的英文名（大小写/片段宽松匹配）。"""
    r = (raw or "").strip()
    if not r:
        return ""
    for info in char_list.values():
        if r.lower() == info["name"].lower():
            return info["name"]
    for info in char_list.values():
        n = info["name"].lower()
        if n in r.lower() or r.lower() in n:
            return info["name"]
    return r


def _merge_text_offline(shots):
    """离线兜底：按顺序拼接台词原文，不标注说话者。"""
    return "\n\n".join(
        c for s in shots if (c := _clean_chapter(s.get("text") or ""))
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _parse_num_arg(value):
    """argparse 类型：把 '3' 或 'scene3' 解析成 int。"""
    m = re.search(r"\d+", str(value))
    if not m:
        raise argparse.ArgumentTypeError(f"无法解析数字: {value!r}")
    return int(m.group())


def _match(p, scene, shot):
    sn, sh = parse_shot_id(p.get("shot_id"))
    if scene is not None and sn != scene:
        return False
    if shot is not None and sh != shot:
        return False
    return True


def _build_segments(prompts, flags):
    """按关键帧切段，返回 (segments, n_segments, seg_of_shot)。

    segments 为 OrderedDict {seg_id: [(i, p), ...]}；seg_of_shot 为与 prompts 等长的
    seg_id 列表（shot_idx -> seg_id）。每个 True 开启新段，False 并入当前段。若 flags
    缺失或长度不符，退回每镜单独成段（安全兜底）。
    """
    if len(flags) != len(prompts):
        flags = [True] * len(prompts)

    seg_of_shot = []
    sid = -1
    for f in flags:
        if f:
            sid += 1
        seg_of_shot.append(sid)

    segments = OrderedDict()
    for i, p in enumerate(prompts):
        segments.setdefault(seg_of_shot[i], []).append((i, p))
    return segments, len(segments), seg_of_shot


def main():
    parser = argparse.ArgumentParser(description="按关键帧(key_frame_flags)切段，合并段内分镜提示词，生成 frame_secene_prompt 追加到 meta.json")
    parser.add_argument("meta", nargs="?", default="./work/meta.json",
                        help="meta.json 路径（默认 ./work/meta.json）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不调用 DeepSeek、不写文件")
    parser.add_argument("--offline", action="store_true", help="不调用 DeepSeek，text 退化为纯拼接、不标注说话者")
    parser.add_argument("--scene", type=_parse_num_arg, default=None,
                        help="只合并指定场景（数字，如 --scene 3）")
    parser.add_argument("--shot", type=_parse_num_arg, default=None,
                        help="只合并指定场景内的指定分镜（需与 --scene 配合，用于测试）")
    args = parser.parse_args()

    meta_path = Path(args.meta).resolve()
    if not meta_path.is_file():
        raise SystemExit(f"找不到 meta.json：{meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task = data.get("task_data") or data
    prompts = task.get("prompts") or []
    if not prompts:
        raise SystemExit("meta.json 中未找到 prompts 列表")
    durations = task.get("durations") or []
    flags = task.get("key_frame_flags") or []

    # 角色音色注册表（复用 generate_shots，保证同一角色音色恒定）
    registry = build_character_registry(prompts)

    # 角色名单（character_specs.json，meta.json 所在目录的 artifacts/story 下）：英文名 + 外观
    char_list = {}
    if not args.offline and not args.dry_run:
        try:
            char_list = load_character_list(meta_path)
        except RuntimeError as e:
            print(f"警告：{e}；对白抽取将不可用")

    # 按关键帧切段（每个 True 开启新段）
    all_segments, total_segments, seg_of_shot = _build_segments(prompts, flags)
    if len(flags) != len(prompts):
        print("警告：key_frame_flags 缺失或长度不符，退回每镜单独成段")

    # 按 --scene / --shot 过滤（分镜级），再按段分组（保留段出现顺序）
    selected = []  # [(seg_id, [(i, p), ...])]
    for seg_id, shots in all_segments.items():
        kept = [(i, p) for i, p in shots if _match(p, args.scene, args.shot)]
        if kept:
            selected.append((seg_id, kept))
    if not selected:
        raise SystemExit("没有匹配指定 --scene / --shot 的分镜")

    need_key = not args.offline and not args.dry_run
    if need_key and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise SystemExit("全局对白抽取需要环境变量 DEEPSEEK_API_KEY；或加 --offline 跳过对白抽取")

    # 全局抽取对白（整篇文章一次交给 DeepSeek），再按 shot_idx -> seg_id 分散到各段
    seg_dialogue = {}
    if need_key and char_list:
        try:
            dialogue_lines = extract_dialogue_global(prompts, char_list)
        except Exception as e:
            print(f"警告：全局对白抽取失败，dialogue 置空：{e}", flush=True)
            dialogue_lines = []
        for shot_idx, name, dl in dialogue_lines:
            sid = seg_of_shot[shot_idx] if shot_idx < len(seg_of_shot) else -1
            if sid >= 0:
                seg_dialogue.setdefault(sid, []).append(f"{name}：{dl}")
        print(f"对白抽取   : 共 {len(dialogue_lines)} 句对白，分布到 {len(seg_dialogue)} 段")

    print("=" * 70)
    print(f"meta.json   : {meta_path}")
    print(f"分镜总数   : {len(prompts)}，关键帧段总数 : {total_segments}，本次合并段 : {len(selected)}")
    if args.dry_run:
        print("模式       : dry-run（不调用 DeepSeek、不写文件）")
    elif args.offline:
        print("模式       : offline（prompt 确定性合并；无对白抽取）")
    else:
        print("模式       : prompt 确定性合并 + 全局对白抽取（说话者识别）")
    if registry:
        print("角色音色   :")
        for cid, info in registry.items():
            nm = char_list.get(cid, {}).get("name", "")
            print(f"    {cid}  ->  {info['voice']}" + (f"（{nm}）" if nm else ""))
    print("=" * 70)

    entries = []   # 本次合并出的新条目
    for pos, (seg_id, shots) in enumerate(selected, 1):
        dur = round(sum(
            float(p.get("duration") or (durations[i] if i < len(durations) else 0.0))
            for i, p in shots
        ), 2)
        frame_idx = [i for i, _ in shots]

        if args.dry_run:
            first = shots[0][1]
            print(f"[{pos}/{len(selected)}] seg={seg_id}  scene_id={first.get('scene_id')}  "
                  f"world={first.get('world_scene_id')}  key_frame={frame_idx[0]}  "
                  f"frames={frame_idx}  dur={dur}s")
            print(f"    首镜 prompt: {(first.get('image_prompt') or '')[:80]}...")
            print(f"    首镜 text  : {_clean_chapter(first.get('text') or '')[:40]}...")
            continue

        print(f"[{pos}/{len(selected)}] 合并 seg={seg_id}（scene_id={shots[0][1].get('scene_id')}，{len(shots)} 分镜）...", flush=True)
        shot_objs = [p for _, p in shots]
        merged_prompt = _merge_prompt_deterministic(shot_objs)
        merged_text = _merge_text_offline(shot_objs)  # 中文原文（离线参考，不标注说话者）

        entries.append({
            "scene_id": str(shots[0][1].get("scene_id")),
            "world_scene_id": shots[0][1].get("world_scene_id"),
            "frame_image_idx": frame_idx,
            "text": merged_text,
            "dialogue": seg_dialogue.get(seg_id, []),
            "prompt": merged_prompt,
            "duration": dur,
        })

    if args.dry_run:
        print("\ndry-run 完成，未写入文件。")
        return

    # 与已有 frame_secene_prompt 合并（部分筛选时保留其它段条目），按关键帧起始下标排序
    existing = data.get("frame_secene_prompt") or []
    if not isinstance(existing, list):
        existing = []
    entry_by_start = {}
    for e in existing:
        if isinstance(e, dict):
            fi = e.get("frame_image_idx") or []
            if fi:
                entry_by_start[fi[0]] = e
    for e in entries:
        entry_by_start[e["frame_image_idx"][0]] = e
    order = sorted(entry_by_start.keys())
    data["frame_secene_prompt"] = [entry_by_start[k] for k in order]

    # 角色音色表（供下游保证同一角色音色统一；附带 character_specs.json 的英文名与外观）
    if registry:
        fc_chars = {}
        for cid, info in registry.items():
            entry = dict(info)
            cl = char_list.get(cid) or {}
            if cl.get("name"):
                entry["name"] = cl["name"]
            if cl.get("description"):
                entry["description"] = cl["description"]
            fc_chars[cid] = entry
        data["frame_secene_characters"] = fc_chars

    # 与现有 meta.json 保持一致：indent=2、ensure_ascii=False、CRLF、无末尾换行
    text = json.dumps(data, ensure_ascii=False, indent=2)
    text = text.replace("\n", "\r\n")
    with open(meta_path, "wb") as f:
        f.write(text.encode("utf-8"))

    print("=" * 70)
    print(f"已写入 frame_secene_prompt（{len(order)} 条段）到 {meta_path}")
    if registry:
        print(f"已写入 frame_secene_characters（{len(registry)} 个角色）")
    print("\n样例（第 1 条）：")
    first = data["frame_secene_prompt"][0]
    print(f"  scene_id={first['scene_id']}  world={first['world_scene_id']}  "
          f"frames={first['frame_image_idx']}  dur={first['duration']}s")
    print(f"  prompt  : {first['prompt'][:160]}...")
    print(f"  text    : {first['text'][:160]}...")
    print(f"  dialogue: {first.get('dialogue', [])}")


if __name__ == "__main__":
    main()
