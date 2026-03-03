import json
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from loguru import logger

from app.config import config


class MultiClipperError(Exception):
    pass


def to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def resolve_binary(config_value: str, fallback_name: str, required: bool = True) -> str:
    if config_value:
        if os.path.isfile(config_value):
            return config_value
        raise MultiClipperError(
            f"Configured binary does not exist: {config_value}. Please fix your config."
        )
    found = shutil.which(fallback_name)
    if found:
        return found
    if required:
        raise MultiClipperError(
            f"Required dependency '{fallback_name}' is not installed or not in PATH."
        )
    return ""


def run_cmd(cmd: List[str], error_hint: str = "") -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (proc.stdout or "").strip()
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        details = stderr or stdout or str(e)
        if error_hint:
            raise MultiClipperError(f"{error_hint}\nCommand failed: {' '.join(cmd)}\n{details}")
        raise MultiClipperError(f"Command failed: {' '.join(cmd)}\n{details}")


def resolve_dependencies(require_ytdlp: bool = False) -> tuple[str, str]:
    clip_cfg = config.youtube_clipper or {}
    ytdlp_path = ""
    if require_ytdlp:
        ytdlp_path = resolve_binary(clip_cfg.get("yt_dlp_path", ""), "yt-dlp", required=True)
    ffmpeg_bin = resolve_binary(config.app.get("ffmpeg_path", ""), "ffmpeg", required=True)
    return ytdlp_path, ffmpeg_bin


def download_youtube_video(youtube_url: str, output_dir: str, yt_dlp_bin: str) -> str:
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
    run_cmd(cmd, "Failed to download YouTube video with yt-dlp.")

    candidates = [
        os.path.join(output_dir, n)
        for n in os.listdir(output_dir)
        if n.startswith("source.") and os.path.isfile(os.path.join(output_dir, n))
    ]
    if not candidates:
        raise MultiClipperError("yt-dlp finished but no video file was produced.")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def transcribe_with_faster_whisper(video_path: str, language: str) -> Tuple[List[Dict], str]:
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


def transcribe_with_whisper(video_path: str, language: str) -> Tuple[List[Dict], str]:
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


def transcribe_video(video_path: str, language: str) -> Tuple[List[Dict], str]:
    try:
        return transcribe_with_faster_whisper(video_path=video_path, language=language)
    except Exception as e:
        logger.warning(f"faster-whisper unavailable or failed, fallback to whisper: {e}")
    try:
        return transcribe_with_whisper(video_path=video_path, language=language)
    except Exception as e:
        raise MultiClipperError(f"Transcription failed with both faster-whisper and whisper: {e}")


def text_score(text: str) -> float:
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


def llm_boost(candidates: List[Dict], enabled: bool) -> List[Dict]:
    if not enabled or not candidates:
        return candidates

    try:
        from app.services import llm
    except Exception as e:
        logger.warning(f"LLM module unavailable, skipping enhancement: {e}")
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
        llm_result = llm._generate_response(prompt)
        match = re.search(r"\[[\s\S]*\]", llm_result or "")
        if not match:
            logger.warning("LLM enhancement returned non-JSON response, skipping.")
            return candidates
        items = json.loads(match.group(0))
        yc_cfg = config.youtube_clipper or {}
        weight = to_float(yc_cfg.get("llm_enhance_weight", 1.0), 1.0)
        for item in items:
            idx = item.get("idx")
            boost = to_float(item.get("boost", 0.0), 0.0)
            if isinstance(idx, int) and 0 <= idx < len(probe):
                probe[idx]["score"] += max(0.0, min(3.0, boost)) * weight
    except Exception as e:
        logger.warning(f"LLM enhancement failed, continuing with heuristic scores: {e}")
    return candidates


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def pick_top_non_overlapping(candidates, clip_count: int, min_gap: float = 2.0):
    selected = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        if all(overlap(cand.start, cand.end, s.start, s.end) < min_gap for s in selected):
            selected.append(cand)
        if len(selected) >= clip_count:
            break
    return selected


def cut_vertical_clip(
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
    run_cmd(cmd, "Failed to cut clip with ffmpeg.")


def ffprobe_duration(video_path: str) -> float:
    ffprobe = resolve_binary(config.app.get("ffprobe_path", ""), "ffprobe", required=True)
    out = run_cmd(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        "Failed to probe video duration.",
    )
    return to_float(out.strip(), 0.0)
