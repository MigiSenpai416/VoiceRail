"""Progress reporting shared by long-running training stages."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class StopRequested(Exception):
    """Raised inside a stage when the user asked the job to stop."""


@dataclass
class Reporter:
    """Callbacks a stage uses to report progress and honour stop requests."""

    on_progress: Callable[[float, str], None] = lambda fraction, message: None
    stop_requested: Callable[[], bool] = lambda: False
    on_metrics: Callable[[dict], None] = lambda metrics: None
    _last_message: str = field(default="", repr=False)

    def progress(self, fraction: float, message: str) -> None:
        self.on_progress(max(0.0, min(1.0, fraction)), message)
        if message != self._last_message:
            log.info(message)
            self._last_message = message

    def check_stop(self) -> None:
        if self.stop_requested():
            raise StopRequested

    def metrics(self, values: dict) -> None:
        self.on_metrics(values)
