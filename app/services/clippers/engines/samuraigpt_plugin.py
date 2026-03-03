import json
import os
import subprocess

from app.config import config
from app.services.clippers.base import ClipperEngine
from app.services.clippers.models import ClipCandidate, EngineResult, MultiClipContext


class SamurAIGPTPluginEngine(ClipperEngine):
    name = "samuraigpt_plugin"

    def _runner_path(self) -> str:
        cfg = config.multi_clipper or {}
        return str(cfg.get("plugin_runner", "")).strip()

    def is_available(self) -> tuple[bool, str]:
        runner = self._runner_path()
        if not runner:
            return False, "multi_clipper.plugin_runner not configured"
        if not os.path.exists(runner):
            return False, f"plugin runner not found: {runner}"
        return True, ""

    def collect(self, ctx: MultiClipContext) -> EngineResult:
        result = EngineResult(engine=self.name)
        available, reason = self.is_available()
        if not available:
            result.warnings.append(reason)
            return result

        runner = self._runner_path()
        output_manifest = os.path.join(ctx.clips_root, self.name, "plugin_manifest.json")
        os.makedirs(os.path.dirname(output_manifest), exist_ok=True)

        cmd = [
            runner,
            "--source-video",
            ctx.source_video,
            "--output-manifest",
            output_manifest,
            "--clip-count",
            str(ctx.clip_count),
            "--min-duration",
            f"{ctx.min_clip_duration}",
            "--max-duration",
            f"{ctx.max_clip_duration}",
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except Exception as e:
            result.warnings.append(f"Plugin runner failed to execute: {e}")
            return result

        payload = None
        if os.path.isfile(output_manifest):
            try:
                with open(output_manifest, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except Exception as e:
                result.warnings.append(f"Failed to parse plugin manifest file: {e}")

        if payload is None:
            stdout = (proc.stdout or "").strip()
            if stdout:
                try:
                    payload = json.loads(stdout)
                except Exception:
                    pass

        if payload is None:
            details = (proc.stderr or "").strip() or "empty plugin response"
            result.warnings.append(f"Plugin backend produced no usable output: {details}")
            return result

        clips = payload.get("clips", payload if isinstance(payload, list) else [])
        if not isinstance(clips, list):
            result.warnings.append("Plugin output is not a clip list")
            return result

        for item in clips:
            try:
                start = float(item.get("start", 0.0))
                end = float(item.get("end", 0.0))
                if end <= start:
                    continue
                duration = end - start
                if duration < ctx.min_clip_duration or duration > ctx.max_clip_duration:
                    continue
                score = float(item.get("score", 1.0))
                text = str(item.get("text", "plugin clip"))
                result.candidates.append(
                    ClipCandidate(
                        engine=self.name,
                        start=start,
                        end=end,
                        score=score,
                        text=text,
                        metadata={"raw": item},
                    )
                )
            except Exception:
                continue

        if not result.candidates and not result.warnings:
            result.warnings.append("Plugin backend returned zero valid clips")
        return result
