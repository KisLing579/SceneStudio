#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 meta.json 逐分镜生成 clips 视频段（只生成，不做最终拼接与配音合成）。

分镜视频输出到 clips/ 目录（output_{shot_idx}.mp4），末帧续接图、日志输出到 work/。
最终结果视频用 concat_clips.py 拼接，配音用 concat_audio.py / add_audio.py 处理。

用法：
  python generate_clips.py [meta.json 路径] [--dry-run] [--force]
                           [--scene N] [--shot M]

  --dry-run  只打印每个分镜的参数与命令，不真正执行。
  --force    强制重新生成已存在的输出（默认跳过已完成的视频段，便于续跑）。
  --scene N  只生成第 N 个场景的所有分镜（如 --scene 3）。
  --shot M   只生成指定场景内的第 M 个分镜，需与 --scene 配合（如 --scene 3 --shot 2）。
  --prompt_with_text  把对应 lines 的中文文本翻译成英文，追加到 prompt（需 DEEPSEEK_API_KEY）。
"""

import argparse
from pathlib import Path

from ltx_common import (
    CLIPS_DIR,
    WORK_DIR,
    MAX_FRAMES,
    frame_segments,
    build_command,
    run_generation,
    extract_last_frame,
    concat_videos,
    load_meta,
    translate_to_english,
    _exists_nonempty,
    _parse_num_arg,
    select_indices,
)


def main():
    parser = argparse.ArgumentParser(description="按 meta.json 逐分镜生成 clips 视频段")
    parser.add_argument("meta", nargs="?", default="meta.json",
                        help="meta.json 路径（默认脚本同目录下 meta.json）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    parser.add_argument("--force", action="store_true", help="重新生成已存在的输出")
    parser.add_argument("--scene", type=_parse_num_arg, default=None,
                        help="只生成指定场景（数字，如 --scene 3）")
    parser.add_argument("--shot", type=_parse_num_arg, default=None,
                        help="只生成指定场景内的指定分镜（数字，需与 --scene 配合，如 --scene 3 --shot 2）")
    parser.add_argument("--prompt_with_text", action="store_true",
                        help="把对应 lines 的中文文本翻译成英文，追加到每个分镜的 prompt 后面")
    args = parser.parse_args()

    meta_path = Path(args.meta).resolve()
    prompts, images, durations, _, lines = load_meta(meta_path)
    base_dir = meta_path.parent
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    indices = select_indices(prompts, args.scene, args.shot)

    total_shots = len(indices)
    total_segments = 0
    for idx in indices:
        p = prompts[idx]
        dur = p.get("duration")
        if dur is None and idx < len(durations):
            dur = durations[idx]
        dur = float(dur)
        total_segments += len(frame_segments(dur))

    print(f"meta.json : {meta_path}")
    if args.scene is not None or args.shot is not None:
        cond = []
        if args.scene is not None:
            cond.append(f"场景 {args.scene}")
        if args.shot is not None:
            cond.append(f"分镜 {args.shot}")
        print(f"筛选条件 : {'，'.join(cond)}")
    print(f"分镜总数 : {total_shots}")
    print(f"视频段数 : {total_segments}（含切段，单段 <= {MAX_FRAMES} 帧）")
    print(f"分镜目录 : {CLIPS_DIR}")
    print(f"工作目录 : {WORK_DIR}")
    print("=" * 70)

    prev_scene_key = None
    prev_cont_image = None   # 上一个分镜最终视频的最后一帧

    for pos, idx in enumerate(indices):
        i = idx              # 原始下标，用于 images/durations 对齐
        p = prompts[idx]
        shot_idx = p.get("shot_idx", i)
        shot_id = p.get("shot_id", f"shot{shot_idx}")
        prompt = (p.get("image_prompt") or p.get("text") or "").strip()

        # 可选：把对应 lines 的中文文本翻译成英文，追加到 prompt
        if args.prompt_with_text:
            line_text = ""
            if i < len(lines):
                line_text = (lines[i] or "").strip()
            if not line_text:
                line_text = (p.get("text") or "").strip()
            if line_text:
                if args.dry_run:
                    preview = line_text[:60] + ("..." if len(line_text) > 60 else "")
                    prompt = f"{prompt}\n\n[待翻译并追加: {preview}]"
                else:
                    en = translate_to_english(line_text)
                    prompt = f"{prompt}\n\n{en}"

        dur = p.get("duration")
        if dur is None and i < len(durations):
            dur = durations[i]
        dur = float(dur)
        segs = frame_segments(dur)

        # 场景键：优先 scene_idx，其次取 shot_id 横线前半部分
        scene_key = p.get("scene_idx")
        if scene_key is None:
            scene_key = str(shot_id).split("-")[0] if "-" in str(shot_id) else str(shot_id)
        is_first_in_scene = (pos == 0) or (scene_key != prev_scene_key)

        # 起始图
        if is_first_in_scene:
            start_image = (base_dir / images[i]).resolve()
        else:
            start_image = Path(prev_cont_image)

        final_output = CLIPS_DIR / f"output_{shot_idx}.mp4"

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
            command = build_command(prompt, seg_image, nf, seg_output)

            if args.dry_run:
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
                run_generation(command, log_path)
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
        print("下一步：python concat_clips.py [meta.json 路径] 拼接结果视频。")


if __name__ == "__main__":
    main()
