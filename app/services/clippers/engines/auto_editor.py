import json
import os
import re
import tempfile

from app.services.clippers.base import ClipperEngine
from app.services.clippers.helpers import resolve_binary, run_cmd
from app.services.clippers.models import ClipCandidate, EngineResult, MultiClipContext


class AutoEditorEngine(ClipperEngine):
    name = "auto_editor"

    def is_available(self) -> tuple[bool, str]:
        try:
            resolve_binary("", "auto-editor", required=True)
            return True, ""
        except Exception as e:
            return False, str(e)

    def _parse_json_segments(self, payload: dict):
        segments = []

        def walk(node):
            if isinstance(node, dict):
                if "start" in node and "end" in node:
                    try:
                        start = float(node["start"])
                        end = float(node["end"])
                        if end > start:
                            segments.append((start, end))
                    except Exception:
                        pass
                for val in node.values():
                    walk(val)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        return segments

    def _parse_preview_segments(self, text: str):
        segments = []
        for line in text.splitlines():
            nums = re.findall(r"(\d+(?:\.\d+)?)", line)
            if len(nums) >= 2:
                try:
                    start = float(nums[0])
                    end = float(nums[1])
                    if end > start:
                        segments.append((start, end))
                except Exception:
                    continue
        return segments

    def collect(self, ctx: MultiClipContext) -> EngineResult:
        result = EngineResult(engine=self.name)
        available, reason = self.is_available()
        if not available:
            result.warnings.append(reason)
            return result

        auto_editor_bin = resolve_binary("", "auto-editor", required=True)
        segments = []
        with tempfile.TemporaryDirectory(prefix="auto-editor-") as tmp:
            json_path = os.path.join(tmp, "timeline.json")
            cmd = [
                auto_editor_bin,
                ctx.source_video,
                "--edit",
                "audio",
                "--silent-speed",
                "99999",
                "--video-speed",
                "1",
                "--margin",
                "0.2sec",
                "--export",
                "json",
                "-o",
                json_path,
                "--no-open",
            ]
            try:
                run_cmd(cmd)
                if os.path.isfile(json_path):
                    with open(json_path, "r", encoding="utf-8") as fp:
                        segments = self._parse_json_segments(json.load(fp))
            except Exception as e:
                result.warnings.append(f"auto-editor json export failed: {e}")

            if not segments:
                try:
                    preview = run_cmd([auto_editor_bin, ctx.source_video, "--preview", "--no-open"])
                    segments = self._parse_preview_segments(preview)
                except Exception as e:
                    result.warnings.append(f"auto-editor preview failed: {e}")

        for start, end in segments:
            duration = end - start
            if duration < ctx.min_clip_duration or duration > ctx.max_clip_duration:
                continue
            score = 1.2 + max(0.0, 1.0 - abs(duration - 28.0) / 25.0)
            result.candidates.append(
                ClipCandidate(
                    engine=self.name,
                    start=start,
                    end=end,
                    score=score,
                    text="auto-editor detected segment",
                )
            )

        if not result.candidates and not result.warnings:
            result.warnings.append("auto-editor produced no valid segments")
        return result
