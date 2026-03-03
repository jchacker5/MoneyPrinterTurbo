#!/usr/bin/env python3
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.services import clipping  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-clip YouTube videos into vertical short-form clips"
    )
    parser.add_argument("--url", required=True, help="YouTube URL")
    parser.add_argument("--clip-count", type=int, default=None, help="Number of clips")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Minimum clip duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum clip duration in seconds",
    )
    parser.add_argument("--language", default="", help="Transcript language code, e.g. en")
    parser.add_argument(
        "--llm-enhancement",
        action="store_true",
        help="Enable LLM enhancement for ranking",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory (defaults to storage/tasks/yt-clips-<uuid>)",
    )

    args = parser.parse_args()

    try:
        manifest = clipping.auto_clip_youtube(
            youtube_url=args.url,
            clip_count=args.clip_count,
            min_clip_duration=args.min_duration,
            max_clip_duration=args.max_duration,
            language=args.language,
            llm_enhancement=args.llm_enhancement,
            output_dir=args.output_dir,
        )
    except clipping.ClippingError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 3

    clips = manifest.get("clips", [])
    print(f"Generated {len(clips)} clips")
    print(f"Workdir: {manifest.get('workdir')}")
    for clip in clips:
        print(
            f"- clip-{clip.get('index'):02d}: {clip.get('file')} "
            f"({clip.get('start')}s -> {clip.get('end')}s, score={clip.get('score')})"
        )
    print("\nManifest JSON:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
