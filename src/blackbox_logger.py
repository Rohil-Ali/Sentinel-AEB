"""
blackbox_logger.py – Black Box Logger for the AEB system.

Records every AEB decision to a timestamped CSV file for post-run
evaluation, debugging, and evidence of system behaviour.

Each row captures a snapshot of the system state at that frame:
timestamp, speed, AEB state, brake command, danger score, closest
object info, FPS, latency, and collision flag.

Usage:
    logger = BlackBoxLogger()
    logger.start()                  # creates a new CSV file
    logger.log(speed_mph, decision, fps, latency_ms, collision)
    logger.stop()                   # flushes and closes the file
"""

from __future__ import annotations

import os
import csv
import time
from datetime import datetime
from typing import Optional

from aeb_controller import AEBDecision


# CSV column headers
_COLUMNS = [
    "timestamp",
    "elapsed_s",
    "speed_mph",
    "aeb_state",
    "brake_cmd",
    "danger_score",
    "closest_class",
    "closest_conf",
    "closest_closeness",
    "reason",
    "fps",
    "latency_ms",
    "collision",
]


class BlackBoxLogger:
    """
    Writes AEB system state to a CSV file every frame.

    Files are saved to a 'logs/' directory with a timestamped filename so each run produces a separate log.
    """

    def __init__(self, log_dir: str = "logs"):
        self._log_dir = log_dir
        self._file = None
        self._writer: Optional[csv.writer] = None
        self._start_time: float = 0.0
        self._enabled = True
        self._row_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    @property
    def is_logging(self) -> bool:
        return self._file is not None and not self._file.closed

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def filepath(self) -> Optional[str]:
        if self._file and not self._file.closed:
            return self._file.name
        return None

    def start(self) -> str:
        os.makedirs(self._log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"aeb_log_{timestamp}.csv"
        filepath = os.path.join(self._log_dir, filename)

        self._file = open(filepath, mode="w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_COLUMNS)
        self._start_time = time.time()
        self._row_count = 0

        print(f"BlackBox logging to: {filepath}")
        return filepath

    def log(self, speed_mph: float, decision: AEBDecision, fps: float = 0.0, latency_ms: float = 0.0, collision: bool = False) -> None:

        if not self._enabled or not self._writer or not self._file:
            return

        now = time.time()
        elapsed = now - self._start_time

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"{elapsed:.3f}",
            f"{speed_mph:.2f}",
            decision.state.value,
            f"{decision.brake:.3f}",
            f"{decision.danger_score:.3f}",
            decision.closest_class or "",
            f"{decision.closest_conf:.3f}" if decision.closest_conf is not None else "",
            f"{decision.closest_closeness:.3f}" if decision.closest_closeness is not None else "",
            decision.reason,
            f"{fps:.1f}",
            f"{latency_ms:.1f}",
            "TRUE" if collision else "",
        ]

        self._writer.writerow(row)
        self._row_count += 1

        if self._row_count % 50 == 0:
            self._file.flush()

    def stop(self) -> None:
        if self._file and not self._file.closed:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            print(f"BlackBox log closed ({self._row_count} rows)")

        self._writer = None
        self._file = None