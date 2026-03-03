from app.services.clippers.base import ClipperEngine
from app.services.clippers.models import ClipCandidate, EngineResult, MultiClipContext


class PySceneDetectEngine(ClipperEngine):
    name = "pyscenedetect"

    def is_available(self) -> tuple[bool, str]:
        try:
            import scenedetect  # noqa: F401

            return True, ""
        except Exception as e:
            return False, f"PySceneDetect unavailable: {e}"

    def collect(self, ctx: MultiClipContext) -> EngineResult:
        result = EngineResult(engine=self.name)
        available, reason = self.is_available()
        if not available:
            result.warnings.append(reason)
            return result

        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector

            video = open_video(ctx.source_video)
            manager = SceneManager()
            manager.add_detector(ContentDetector(threshold=27.0))
            manager.detect_scenes(video, show_progress=False)
            scenes = manager.get_scene_list()
        except Exception as e:
            result.errors.append(f"PySceneDetect detection failed: {e}")
            return result

        for idx, scene in enumerate(scenes):
            start_tc, end_tc = scene
            start = start_tc.get_seconds()
            end = end_tc.get_seconds()
            duration = end - start
            if duration < ctx.min_clip_duration or duration > ctx.max_clip_duration:
                continue

            # Reward scenes near ideal short-form length and early momentum.
            score = 1.0 + max(0.0, 1.2 - abs(duration - 25.0) / 22.0)
            score += max(0.0, 0.3 - idx * 0.01)
            result.candidates.append(
                ClipCandidate(
                    engine=self.name,
                    start=start,
                    end=end,
                    score=score,
                    text=f"Scene {idx + 1}",
                )
            )

        if not result.candidates and not result.errors:
            result.warnings.append("PySceneDetect found no scenes matching duration constraints")
        return result
