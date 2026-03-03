from typing import Dict, List

from app.services.clippers.base import ClipperEngine
from app.services.clippers.helpers import llm_boost, text_score, transcribe_video
from app.services.clippers.models import ClipCandidate, EngineResult, MultiClipContext


class NativeHeuristicEngine(ClipperEngine):
    name = "native"

    def is_available(self) -> tuple[bool, str]:
        has_faster = True
        has_whisper = True
        try:
            import faster_whisper  # noqa: F401
        except Exception:
            has_faster = False
        try:
            import whisper  # noqa: F401
        except Exception:
            has_whisper = False

        if has_faster or has_whisper:
            return True, ""
        return False, "Missing transcription dependencies: faster-whisper or openai-whisper"

    def _build_candidates(
        self,
        transcript: List[Dict],
        min_clip_duration: float,
        max_clip_duration: float,
    ) -> List[Dict]:
        candidates: List[Dict] = []
        limit = 400

        for i in range(len(transcript)):
            start = transcript[i]["start"]
            parts: List[str] = []
            for j in range(i, len(transcript)):
                seg = transcript[j]
                parts.append(seg["text"])
                end = seg["end"]
                duration = end - start
                if duration > max_clip_duration:
                    break
                if duration < min_clip_duration:
                    continue

                text = " ".join(parts).strip()
                score = text_score(text)
                score += max(0.0, 1.5 - abs(duration - 30.0) / 20.0)
                score += min(1.0, len(text) / 220.0)
                candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "text": text,
                        "score": score,
                    }
                )
                if len(candidates) >= limit:
                    return candidates
        return candidates

    def collect(self, ctx: MultiClipContext) -> EngineResult:
        result = EngineResult(engine=self.name)
        transcript, backend = transcribe_video(ctx.source_video, language=ctx.language)
        if not transcript:
            result.errors.append("Transcription returned no segments")
            return result

        raw_candidates = self._build_candidates(
            transcript=transcript,
            min_clip_duration=ctx.min_clip_duration,
            max_clip_duration=ctx.max_clip_duration,
        )
        if not raw_candidates:
            result.warnings.append("No native heuristic candidates found")
            return result

        boosted = llm_boost(raw_candidates, enabled=ctx.llm_enhancement)
        result.candidates = [
            ClipCandidate(
                engine=self.name,
                start=float(c["start"]),
                end=float(c["end"]),
                score=float(c["score"]),
                text=c.get("text", ""),
                metadata={"transcript_backend": backend},
            )
            for c in boosted
            if float(c["end"]) > float(c["start"])
        ]
        return result
