from abc import ABC, abstractmethod

from app.services.clippers.models import EngineResult, MultiClipContext


class ClipperEngine(ABC):
    name = "base"

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def collect(self, ctx: MultiClipContext) -> EngineResult:
        raise NotImplementedError
