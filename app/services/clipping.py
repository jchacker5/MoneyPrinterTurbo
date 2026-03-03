import json
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from loguru import logger

from app.config import config
from app.services import llm
from app.utils import utils


class ClippingError(Exception):
    pass


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _resolve_binary(config_value: str, fallback_name: str) -> str:
    if config_value:
        if os.path.isfile(config_value):
            return config_value
        raise ClippingError(
            f"Configured binary does not exist: {config_value}. Please fix your config."
        )
    found = shutil.which(fallback_name)
    if found:
        return found
    raise ClippingError(
        f"Required dependency '{fallback_name}' is not installed or not in PATH."
    )


def _run(cmd: List[str], error_hint: str = "") -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (proc.stdout or "").strip()
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        details = stderr or stdout or str(e)
        if error_hint:
            raise ClippingError(f"{error_hint}\nCommand failed: {' '.join(cmd)}\n{details}")
        raise ClippingError(f"Command failed: {' '.join(cmd)}\n{details}")


def _dependency_check() -> Tuple[str, str]:
    clip_cfg = config.youtube_clipper or {}
    yt_dlp_bin = _resolve_binary(clip_cfg.get("yt_dlp_path", ""), "yt-dlp")
    ffmpeg_bin = _resolve_binary(config.app.get("ffmpeg_path", ""), "ffmpeg")

    # We rely on one local transcription backend.
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
    if not has_faster and not has_whisper:
        raise ClippingError(
            "Missing transcription dependencies. Install 'faster-whisper' or 'openai-whisper'."
        )

    return yt_dlp_bin, ffmpeg_bin


def _download_youtube_video(youtube_url: str, output_dir: str, yt_dlp_bin: str) -> str:
    output_template = os.path.join(output_dir, "source.%(ext)s")
    cmd = [
        yt_dlp_bin,
        "--no-playlist",
        "-f",
        "mp4/bestvideo+bestaudio/best",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        youtube_url,
    ]
    _run(cmd, "Failed to download YouTube video with yt-dlp.")

    candidates = [
        os.path.join(output_dir, n)
        for n in os.listdir(output_dir)
        if n.startswith("source.") and os.path.isfile(os.path.join(output_dir, n))
    ]
    if not candidates:
        raise ClippingError("yt-dlp finished but no video file was produced.")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _transcribe_with_faster_whisper(video_path: str, language: str) -> Tuple[List[Dict], str]:
    from faster_whisper import WhisperModel

    clip_cfg = config.youtube_clipper or {}
    whisper_cfg = config.whisper or {}
    model_size = clip_cfg.get("faster_whisper_model_size", whisper_cfg.get("model_size", "base"))
    device = str(clip_cfg.get("faster_whisper_device", whisper_cfg.get("device", "cpu"))).lower()
    compute_type = clip_cfg.get(
        "faster_whisper_compute_type",
        whisper_cfg.get("compute_type", "int8"),
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(
        video_path,
        language=language or None,
        vad_filter=True,
    )

    data = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        data.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return data, "faster-whisper"


def _transcribe_with_whisper(video_path: str, language: str) -> Tuple[List[Dict], str]:
    import whisper

    clip_cfg = config.youtube_clipper or {}
    whisper_cfg = config.whisper or {}
    model_name = clip_cfg.get("fallback_whisper_model_size", whisper_cfg.get("model_size", "base"))
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path, language=language or None)
    segments = result.get("segments", [])
    data = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        data.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text,
            }
        )
    return data, "whisper"


def _transcribe_video(video_path: str, language: str) -> Tuple[List[Dict], str]:
    try:
        return _transcribe_with_faster_whisper(video_path, language=language)
    except Exception as e:
        logger.warning(f"faster-whisper unavailable or failed, fallback to whisper: {e}")
    try:
        return _transcribe_with_whisper(video_path, language=language)
    except Exception as e:
        raise ClippingError(f"Transcription failed with both faster-whisper and whisper: {e}")


def _text_score(text: str) -> float:
    clip_cfg = config.youtube_clipper or {}
    bonus_keywords = clip_cfg.get(
        "viral_keywords",
        ["secret", "mistake", "truth", "hack", "shocking", "warning", "money", "why"],
    )
    low = text.lower()
    score = 1.0

    for kw in bonus_keywords:
        if kw and kw.lower() in low:
            score += 1.0
    if re.search(r"\d", text):
        score += 0.6
    if "?" in text:
        score += 0.8
    if "!" in text:
        score += 0.5
    if re.search(r"\$(\d+|\s*\d+)|\d+%", text):
        score += 0.8

    words = [w for w in re.split(r"\s+", text.strip()) if w]
    wc = len(words)
    if 10 <= wc <= 80:
        score += 0.8
    elif wc < 5:
        score -= 0.5

    return score


def _build_candidates(
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
            score = _text_score(text)
            # lightweight prior toward ~30s clips and concise text density
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


def _llm_boost(candidates: List[Dict], enabled: bool) -> List[Dict]:
    if not enabled or not candidates:
        return candidates

    top_n = 15
    probe = candidates[:top_n]
    prompt_payload = [
        {
            "idx": i,
            "start": round(c["start"], 2),
            "end": round(c["end"], 2),
            "duration": round(c["duration"], 2),
            "text": c["text"][:450],
        }
        for i, c in enumerate(probe)
    ]
    prompt = (
        "Rank these potential short-video clips for virality. "
        "Return ONLY JSON array items as {\"idx\": number, \"boost\": 0..3, \"reason\": \"...\"}.\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False)}"
    )
    try:
        llm_result = llm._generate_response(prompt)  # reuse configured provider
        match = re.search(r"\[[\s\S]*\]", llm_result or "")
        if not match:
            logger.warning("LLM enhancement returned non-JSON response, skipping.")
            return candidates
        items = json.loads(match.group(0))
        weight = _to_float((config.youtube_clipper or {}).get("llm_enhance_weight", 1.0), 1.0)
        for item in items:
            idx = item.get("idx")
            boost = _to_float(item.get("boost", 0.0), 0.0)
            if isinstance(idx, int) and 0 <= idx < len(probe):
                probe[idx]["score"] += max(0.0, min(3.0, boost)) * weight
    except Exception as e:
        logger.warning(f"LLM enhancement failed, continuing with heuristic scores: {e}")
    return candidates


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _pick_top_non_overlapping(candidates: List[Dict], clip_count: int) -> List[Dict]:
    selected: List[Dict] = []
    for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
        if all(_overlap(cand["start"], cand["end"], s["start"], s["end"]) < 2.0 for s in selected):
            selected.append(cand)
        if len(selected) >= clip_count:
            break
    return selected


def _cut_vertical_clip(
    source_video: str,
    start: float,
    duration: float,
    output_file: str,
    ffmpeg_bin: str,
) -> None:
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        source_video,
        "-t",
        f"{duration:.3f}",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        output_file,
    ]
    _run(cmd, "Failed to cut clip with ffmpeg.")


def auto_clip_youtube(
    youtube_url: str,
    clip_count: Optional[int] = None,
    min_clip_duration: Optional[float] = None,
    max_clip_duration: Optional[float] = None,
    language: str = "",
    llm_enhancement: Optional[bool] = None,
    output_dir: str = "",
) -> Dict:
    if not youtube_url:
        raise ClippingError("youtube_url is required.")
    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        raise ClippingError("youtube_url must be a valid YouTube URL.")

    cfg = config.youtube_clipper or {}
    clip_count = _to_int(clip_count if clip_count is not None else cfg.get("clip_count", 3), 3)
    min_clip_duration = _to_float(
        min_clip_duration if min_clip_duration is not None else cfg.get("min_clip_duration", 20),
        20.0,
    )
    max_clip_duration = _to_float(
        max_clip_duration if max_clip_duration is not None else cfg.get("max_clip_duration", 50),
        50.0,
    )
    if clip_count < 1:
        raise ClippingError("clip_count must be >= 1.")
    if min_clip_duration <= 0 or max_clip_duration <= 0:
        raise ClippingError("Clip durations must be > 0.")
    if max_clip_duration < min_clip_duration:
        raise ClippingError("max_clip_duration must be >= min_clip_duration.")

    if llm_enhancement is None:
        llm_enhancement = bool(cfg.get("enable_llm_enhancement", False))

    yt_dlp_bin, ffmpeg_bin = _dependency_check()

    task_name = f"yt-clips-{utils.get_uuid(remove_hyphen=True)}"
    workdir = output_dir.strip() if output_dir else utils.task_dir(task_name)
    os.makedirs(workdir, exist_ok=True)

    source_video = _download_youtube_video(youtube_url, workdir, yt_dlp_bin)
    transcript, transcript_backend = _transcribe_video(source_video, language=language)
    if not transcript:
        raise ClippingError("Transcription returned no segments; cannot score moments.")

    candidates = _build_candidates(
        transcript=transcript,
        min_clip_duration=min_clip_duration,
        max_clip_duration=max_clip_duration,
    )
    if not candidates:
        raise ClippingError(
            "No candidate clips detected. Try reducing min_clip_duration or using another video."
        )

    candidates = _llm_boost(candidates, enabled=llm_enhancement)
    selected = _pick_top_non_overlapping(candidates, clip_count=clip_count)
    if not selected:
        raise ClippingError("No non-overlapping clips could be selected.")

    created = []
    for idx, seg in enumerate(selected, start=1):
        duration = max(0.5, seg["end"] - seg["start"])
        out_file = os.path.join(workdir, f"clip-{idx:02d}.mp4")
        _cut_vertical_clip(
            source_video=source_video,
            start=seg["start"],
            duration=duration,
            output_file=out_file,
            ffmpeg_bin=ffmpeg_bin,
        )
        created.append(
            {
                "index": idx,
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "duration": round(duration, 2),
                "score": round(seg["score"], 4),
                "text": seg["text"],
                "file": out_file,
            }
        )

    manifest = {
        "youtube_url": youtube_url,
        "source_video": source_video,
        "transcript_backend": transcript_backend,
        "transcript_segments": len(transcript),
        "clips": created,
        "workdir": workdir,
    }
    with open(os.path.join(workdir, "manifest.json"), "w", encoding="utf-8") as fp:
        fp.write(json.dumps(manifest, ensure_ascii=False, indent=2))

    logger.success(
        f"YouTube clipping complete: {len(created)} clips generated in {workdir}"
    )
    return manifest
