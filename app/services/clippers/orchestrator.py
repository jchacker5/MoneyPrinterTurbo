import json
import os
from typing import Dict, List

from loguru import logger

from app.config import config
from app.services.clippers.base import ClipperEngine
from app.services.clippers.engines.auto_editor import AutoEditorEngine
from app.services.clippers.engines.native import NativeHeuristicEngine
from app.services.clippers.engines.pyscenedetect import PySceneDetectEngine
from app.services.clippers.engines.samuraigpt_plugin import SamurAIGPTPluginEngine
from app.services.clippers.helpers import (
    MultiClipperError,
    cut_vertical_clip,
    download_youtube_video,
    pick_top_non_overlapping,
    resolve_dependencies,
    to_float,
    to_int,
)
from app.services.clippers.models import ClipCandidate, EngineResult, MultiClipContext, MultiClipRequest
from app.utils import utils


ENGINE_REGISTRY: Dict[str, ClipperEngine] = {
    NativeHeuristicEngine.name: NativeHeuristicEngine(),
    AutoEditorEngine.name: AutoEditorEngine(),
    PySceneDetectEngine.name: PySceneDetectEngine(),
    SamurAIGPTPluginEngine.name: SamurAIGPTPluginEngine(),
}


def _default_engines() -> List[str]:
    cfg = config.multi_clipper or {}
    engines = cfg.get("enabled_engines") or []
    if isinstance(engines, list) and engines:
        return [str(e).strip() for e in engines if str(e).strip()]
    return [
        NativeHeuristicEngine.name,
        AutoEditorEngine.name,
        PySceneDetectEngine.name,
        SamurAIGPTPluginEngine.name,
    ]


def _strategy() -> str:
    cfg = config.multi_clipper or {}
    return str(cfg.get("selection_strategy", "weighted_sum")).strip().lower()


def _engine_weights() -> Dict[str, float]:
    cfg = config.multi_clipper or {}
    w = cfg.get("weights") or {}
    if isinstance(w, dict):
        return {str(k): to_float(v, 1.0) for k, v in w.items()}
    return {}


def _apply_global_scoring(engine_results: List[EngineResult]) -> List[ClipCandidate]:
    strategy = _strategy()
    weights = _engine_weights()

    fused: List[ClipCandidate] = []
    for result in engine_results:
        ranked = sorted(result.candidates, key=lambda c: c.score, reverse=True)
        for rank, cand in enumerate(ranked, start=1):
            weight = weights.get(result.engine, 1.0)
            merged = ClipCandidate(
                engine=cand.engine,
                start=cand.start,
                end=cand.end,
                score=cand.score,
                text=cand.text,
                metadata=dict(cand.metadata),
            )
            if strategy == "rank_fusion":
                merged.score = weight * (1.0 / (rank + 4.0))
            else:
                merged.score = cand.score * weight
            merged.metadata["raw_score"] = cand.score
            merged.metadata["weight"] = weight
            merged.metadata["strategy"] = strategy
            fused.append(merged)

    return fused


def _render_candidates(
    source_video: str,
    candidates: List[ClipCandidate],
    output_dir: str,
    ffmpeg_bin: str,
    name_prefix: str,
) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    rendered = []
    for idx, cand in enumerate(candidates, start=1):
        duration = max(0.5, cand.duration)
        out_file = os.path.join(output_dir, f"{name_prefix}-{idx:02d}.mp4")
        cut_vertical_clip(
            source_video=source_video,
            start=cand.start,
            duration=duration,
            output_file=out_file,
            ffmpeg_bin=ffmpeg_bin,
        )
        rendered.append(
            {
                "index": idx,
                "engine": cand.engine,
                "start": round(cand.start, 2),
                "end": round(cand.end, 2),
                "duration": round(duration, 2),
                "score": round(cand.score, 4),
                "text": cand.text,
                "file": out_file,
                "metadata": cand.metadata,
            }
        )
    return rendered


def auto_clip_multi(req: MultiClipRequest) -> Dict:
    if not req.youtube_url and not req.source_video:
        raise MultiClipperError("youtube_url or source_video is required")

    if req.youtube_url and ("youtube.com" not in req.youtube_url and "youtu.be" not in req.youtube_url):
        raise MultiClipperError("youtube_url must be a valid YouTube URL")

    yc_cfg = config.youtube_clipper or {}
    req.clip_count = to_int(req.clip_count if req.clip_count is not None else yc_cfg.get("clip_count", 3), 3)
    req.min_clip_duration = to_float(
        req.min_clip_duration if req.min_clip_duration is not None else yc_cfg.get("min_clip_duration", 20),
        20.0,
    )
    req.max_clip_duration = to_float(
        req.max_clip_duration if req.max_clip_duration is not None else yc_cfg.get("max_clip_duration", 50),
        50.0,
    )
    if req.clip_count < 1:
        raise MultiClipperError("clip_count must be >= 1")
    if req.min_clip_duration <= 0 or req.max_clip_duration <= 0:
        raise MultiClipperError("Clip durations must be > 0")
    if req.max_clip_duration < req.min_clip_duration:
        raise MultiClipperError("max_clip_duration must be >= min_clip_duration")

    llm_enhancement = bool(yc_cfg.get("enable_llm_enhancement", False))
    if req.llm_enhancement is not None:
        llm_enhancement = bool(req.llm_enhancement)

    task_id = req.task_id.strip() if req.task_id else f"yt-multi-{utils.get_uuid(remove_hyphen=True)}"
    workdir = req.output_dir.strip() if req.output_dir else utils.task_dir(task_id)
    clips_root = os.path.join(workdir, "clips")
    best_root = os.path.join(workdir, "best")
    os.makedirs(clips_root, exist_ok=True)
    os.makedirs(best_root, exist_ok=True)

    yt_dlp_bin, ffmpeg_bin = resolve_dependencies(require_ytdlp=bool(req.youtube_url))

    source_video = req.source_video
    if req.youtube_url:
        source_video = download_youtube_video(req.youtube_url, workdir, yt_dlp_bin)
    if not source_video or not os.path.isfile(source_video):
        raise MultiClipperError(f"Source video not found: {source_video}")

    selected_engines = req.selected_engines or _default_engines()
    selected_engines = [e for e in selected_engines if e in ENGINE_REGISTRY]
    if not selected_engines:
        raise MultiClipperError("No valid clipping engine selected")

    ctx = MultiClipContext(
        source_video=source_video,
        task_id=task_id,
        workdir=workdir,
        clips_root=clips_root,
        best_root=best_root,
        ffmpeg_bin=ffmpeg_bin,
        min_clip_duration=req.min_clip_duration,
        max_clip_duration=req.max_clip_duration,
        clip_count=req.clip_count,
        language=req.language or "",
        llm_enhancement=llm_enhancement,
    )

    engine_results: List[EngineResult] = []
    per_engine_rendered: Dict[str, List[Dict]] = {}
    warnings: List[str] = []

    for engine_name in selected_engines:
        engine = ENGINE_REGISTRY[engine_name]
        available, reason = engine.is_available()
        if not available:
            warnings.append(f"[{engine_name}] skipped: {reason}")
            engine_results.append(EngineResult(engine=engine_name, warnings=[reason]))
            continue

        result = engine.collect(ctx)
        engine_results.append(result)
        warnings.extend([f"[{engine_name}] {w}" for w in result.warnings])
        warnings.extend([f"[{engine_name}] ERROR: {e}" for e in result.errors])

        picked = pick_top_non_overlapping(result.candidates, clip_count=req.clip_count)
        if not picked:
            per_engine_rendered[engine_name] = []
            continue

        engine_out = os.path.join(clips_root, engine_name)
        try:
            per_engine_rendered[engine_name] = _render_candidates(
                source_video=source_video,
                candidates=picked,
                output_dir=engine_out,
                ffmpeg_bin=ffmpeg_bin,
                name_prefix="clip",
            )
        except Exception as e:
            warnings.append(f"[{engine_name}] render failed: {e}")
            per_engine_rendered[engine_name] = []

    fused = _apply_global_scoring(engine_results)
    best_candidates = pick_top_non_overlapping(fused, clip_count=req.clip_count)
    if not best_candidates:
        raise MultiClipperError(
            "No clips produced by selected engines. Check dependencies or relax duration constraints."
        )

    best_rendered = _render_candidates(
        source_video=source_video,
        candidates=best_candidates,
        output_dir=best_root,
        ffmpeg_bin=ffmpeg_bin,
        name_prefix="best",
    )

    manifest = {
        "task_id": task_id,
        "youtube_url": req.youtube_url,
        "source_video": source_video,
        "selected_engines": selected_engines,
        "selection_strategy": _strategy(),
        "engine_weights": _engine_weights(),
        "clips": per_engine_rendered,
        "best": best_rendered,
        "warnings": warnings,
        "workdir": workdir,
    }

    manifest_file = os.path.join(workdir, "multi-clip-manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(manifest, ensure_ascii=False, indent=2))

    logger.success(
        f"Multi-engine clipping complete: {len(best_rendered)} best clips generated in {workdir}"
    )
    return manifest
