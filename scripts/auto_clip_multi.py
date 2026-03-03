#!/usr/bin/env python3
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.services.clippers import MultiClipRequest, MultiClipperError, auto_clip_multi  # noqa: E402


def _parse_engines(raw: str):
    text = (raw or "").strip()
    if not text:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-clip videos with multiple engines and best-pick fusion"
    )
    parser.add_argument("--url", default="", help="YouTube URL")
    parser.add_argument("--source-video", default="", help="Local source video path")
    parser.add_argument("--clip-count", type=int, default=3, help="Number of best clips")
    parser.add_argument("--min-duration", type=float, default=20.0, help="Minimum clip duration")
    parser.add_argument("--max-duration", type=float, default=50.0, help="Maximum clip duration")
    parser.add_argument("--language", default="", help="Transcript language code, e.g. en")
    parser.add_argument("--llm-enhancement", action="store_true", help="Enable LLM ranking boost")
    parser.add_argument(
        "--engines",
        default="",
        help="Comma-separated engines (native,auto_editor,pyscenedetect,samuraigpt_plugin)",
    )
    parser.add_argument("--output-dir", default="", help="Output task directory")

    args = parser.parse_args()

    if not args.url and not args.source_video:
        print("ERROR: --url or --source-video is required", file=sys.stderr)
        return 2

    request = MultiClipRequest(
        youtube_url=args.url,
        source_video=args.source_video,
        clip_count=args.clip_count,
        min_clip_duration=args.min_duration,
        max_clip_duration=args.max_duration,
        language=args.language,
        llm_enhancement=True if args.llm_enhancement else None,
        output_dir=args.output_dir,
        selected_engines=_parse_engines(args.engines),
    )

    try:
        manifest = auto_clip_multi(request)
    except MultiClipperError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 3

    print(f"Task: {manifest.get('task_id')}")
    print(f"Workdir: {manifest.get('workdir')}")
    print(f"Selected engines: {', '.join(manifest.get('selected_engines', []))}")
    print(f"Best clips: {len(manifest.get('best', []))}")
    for clip in manifest.get("best", []):
        print(
            f"- best-{clip.get('index'):02d}: {clip.get('file')} "
            f"({clip.get('start')}s -> {clip.get('end')}s, engine={clip.get('engine')}, score={clip.get('score')})"
        )

    warnings = manifest.get("warnings", [])
    if warnings:
        print("\\nWarnings:")
        for w in warnings:
            print(f"- {w}")

    print("\\nManifest JSON:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
