from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClipCandidate:
    engine: str
    start: float
    end: float
    score: float
    text: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))


@dataclass
class EngineResult:
    engine: str
    candidates: List[ClipCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class MultiClipRequest:
    youtube_url: str = ""
    source_video: str = ""
    clip_count: int = 3
    min_clip_duration: float = 20.0
    max_clip_duration: float = 50.0
    language: str = ""
    output_dir: str = ""
    task_id: str = ""
    llm_enhancement: Optional[bool] = None
    selected_engines: Optional[List[str]] = None


@dataclass
class MultiClipContext:
    source_video: str
    task_id: str
    workdir: str
    clips_root: str
    best_root: str
    ffmpeg_bin: str
    min_clip_duration: float
    max_clip_duration: float
    clip_count: int
    language: str = ""
    llm_enhancement: bool = False
