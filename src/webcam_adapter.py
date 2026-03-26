"""
webcam_adapter.py – Webcam input adapter for the AEB system.

Implements the same interface as CarlaAdapter so the GUI can use
either adapter interchangeably without knowing which one is active.

Interface methods:
    start()                     → connect to the input source
    stop()                      → release the input source
    get_frame()                 → return latest BGR frame (or None)
    get_state()                 → return VehicleState (or None)
    drive(target_speed_mph, aeb_brake) → no-op for webcam
    set_rain(intensity)         → no-op for webcam
    spawn_pedestrian_ahead()    → no-op for webcam
    spawn_static_vehicle_ahead()→ no-op for webcam
    clear_spawned_objects()     → no-op for webcam
"""

from __future__ import annotations
import sys
from typing import Optional
import cv2
import numpy as np

from carla_adapter import VehicleState


class WebcamAdapter:
    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720):
        self._device_index = device_index
        self._width = width
        self._height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame: Optional[np.ndarray] = None
        self._running = False

    # ---------- lifecycle ----------
    def start(self) -> None:
        if self._running:
            return

        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self._device_index)

        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Cannot open webcam (device {self._device_index})"
            )

        
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        ok, _ = self._cap.read()
        if not ok:
            self._cap.release()
            self._cap = None
            raise RuntimeError(
                f"Webcam opened but cannot read frames (device {self._device_index})"
            )

        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._last_frame = None

    # ---------- frame + telemetry ----------
    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        if not self._running or not self._cap:
            return None

        ok, frame = self._cap.read()
        if ok:
            self._last_frame = frame
            return frame
        return self._last_frame

    def get_state(self) -> Optional[VehicleState]:
        return None

    # ---------- control (no-ops) ----------
    def apply_control(self, throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0) -> None:
        pass

    def drive(self, target_speed_mph: float, aeb_brake: float = 0.0) -> None:
        pass

    def brake_full(self) -> None:
        pass

    # ---------- environment (no-ops) ----------
    def set_rain(self, intensity: float) -> None:
        pass

    def set_fog(self, intensity: float) -> None:
        pass

    def spawn_pedestrian_ahead(self, meters: Optional[float] = None) -> None:
        pass

    def spawn_static_vehicle_ahead(self, meters: Optional[float] = None) -> None:
        pass

    def clear_spawned_objects(self) -> None:
        pass