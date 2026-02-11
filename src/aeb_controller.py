# aeb_controller.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict

import time
import math

from detector import Detection


class AEBState(str, Enum):
    SCANNING = "SCANNING"
    BRAKING = "BRAKING"


class BrakeCommand(str, Enum):
    NO_ACTION = "NO_ACTION"
    SOFT_BRAKE = "SOFT_BRAKE"
    HARD_BRAKE = "HARD_BRAKE"


@dataclass
class AEBDecision:
    state: AEBState
    command: BrakeCommand
    brake: float                # 0..1
    risk: float                 # 0..1
    closeness_now: float        # 0..1
    approach_score: float       # 0..1
    target: Optional[Detection]
    reason: str                 # short debug string


@dataclass
class VehicleKinematics:
    speed_mps: float


class AEBController:
    """
    Decision logic for vision-based AEB.

    Inputs each tick:
      - detections: list[Detection] from detector.predict(frame)
      - frame_shape: (h, w) to compute bbox area_norm
      - vehicle: speed (m/s)

    Outputs:
      - AEBDecision (risk, brake level, state, target)

    Design choices:
      - Target tracking: simple nearest-centre matching (+ prefer same class)
      - approach_score: derived from bbox area growth over a rolling window
      - debounce: require risk condition for N consecutive frames
      - cooldown: after braking triggers, ignore new triggers for a short period
    """

    def __init__(
        self,
        clossness_weights: float = 0.65,
        approach_weights: float = 0.35,

        # Approach score settings
        history_len: int = 6,            
        growth_ref: float = 0.0035,      

        soft_threshold: float = 0.45,
        hard_threshold: float = 0.65,

        debounce_frames: int = 3,
        cooldown_s: float = 1.0,

        match_max_dist_norm: float = 0.12,

        # Brake strengths
        soft_brake: float = 0.35,
        hard_brake: float = 1.0,

        # Speed influence
        speed_ref_mps: float = 13.4,     # ~30 mph
        speed_risk_gain: float = 0.25,   # risk multiplier range (0..+)
    ):
        self.clossness_weights = float(clossness_weights)
        self.approach_weights = float(approach_weights)

        self.history_len = int(max(2, history_len))
        self.growth_ref = float(max(1e-6, growth_ref))

        self.soft_threshold = float(soft_threshold)
        self.hard_threshold = float(hard_threshold)

        self.debounce_frames = int(max(1, debounce_frames))
        self.cooldown_s = float(max(0.0, cooldown_s))

        self.match_max_dist_norm = float(max(0.01, match_max_dist_norm))

        self.soft_brake = float(max(0.0, min(1.0, soft_brake)))
        self.hard_brake = float(max(0.0, min(1.0, hard_brake)))

        self.speed_ref_mps = float(max(1e-3, speed_ref_mps))
        self.speed_risk_gain = float(max(0.0, speed_risk_gain))

        # Internal state
        self.state: AEBState = AEBState.SCANNING
        self._last_trigger_time: float = 0.0

        self._target_cls_id: Optional[int] = None
        self._target_center: Optional[Tuple[float, float]] = None
        self._area_history: List[float] = []  # area_norm history

        self._danger_streak: int = 0

    # ---------- public API ----------

    def reset(self) -> None:
        self.state = AEBState.SCANNING
        self._last_trigger_time = 0.0
        self._target_cls_id = None
        self._target_center = None
        self._area_history.clear()
        self._danger_streak = 0

    def update(self, detections: List[Detection], frame_shape: Tuple[int, int, int], vehicle: VehicleKinematics, now_s: Optional[float] = None) -> AEBDecision:
        if now_s is None:
            now_s = time.time()

        # Cooldown handling
        if self._in_cooldown(now_s):
            self.state = AEBState.BRAKING
            return AEBDecision(
                state=self.state,
                command=BrakeCommand.NO_ACTION,  # controller says "no new trigger", main may keep brake applied separately
                brake=0.0,
                risk=0.0,
                closeness_now=0.0,
                approach_score=0.0,
                target=None,
                reason="cooldown_active",
            )

        target = self._select_target(detections, frame_shape)
        if target is None:
            self._clear_target()
            self._danger_streak = 0
            self.state = AEBState.SCANNING
            return AEBDecision(
                state=self.state,
                command=BrakeCommand.NO_ACTION,
                brake=0.0,
                risk=0.0,
                closeness_now=0.0,
                approach_score=0.0,
                target=None,
                reason="no_target",
            )

        # Update tracking + history
        area_norm = self._bbox_area_norm(target.xyxy, frame_shape)
        self._update_target_tracking(target, frame_shape, area_norm)

        approach_score = self._compute_approach_score()
        closeness_now = float(max(0.0, min(1.0, target.closeness)))

        # Combine into base risk
        risk_base = (self.clossness_weights * closeness_now) + (self.approach_weights * approach_score)
        risk_base = float(max(0.0, min(1.0, risk_base)))

        # Speed scaling 
        risk = self._apply_speed_scaling(risk_base, vehicle.speed_mps)

        # Debounce logic
        danger_now = risk >= self.soft_threshold
        if danger_now:
            self._danger_streak += 1
        else:
            self._danger_streak = 0

        if self._danger_streak < self.debounce_frames:
            self.state = AEBState.SCANNING
            return AEBDecision(
                state=self.state,
                command=BrakeCommand.NO_ACTION,
                brake=0.0,
                risk=risk,
                closeness_now=closeness_now,
                approach_score=approach_score,
                target=target,
                reason=f"debounce_{self._danger_streak}/{self.debounce_frames}",
            )

        # Decide brake severity
        if risk >= self.hard_threshold:
            self._trigger(now_s)
            self.state = AEBState.BRAKING
            return AEBDecision(
                state=self.state,
                command=BrakeCommand.HARD_BRAKE,
                brake=self.hard_brake,
                risk=risk,
                closeness_now=closeness_now,
                approach_score=approach_score,
                target=target,
                reason="hard_threshold",
            )

        # Soft braking
        self._trigger(now_s)
        self.state = AEBState.BRAKING
        return AEBDecision(
            state=self.state,
            command=BrakeCommand.SOFT_BRAKE,
            brake=self.soft_brake,
            risk=risk,
            closeness_now=closeness_now,
            approach_score=approach_score,
            target=target,
            reason="soft_threshold",
        )

    # ---------- helper functions ----------

    def _in_cooldown(self, now_s: float) -> bool:
        if self._last_trigger_time <= 0.0:
            return False
        return (now_s - self._last_trigger_time) < self.cooldown_s

    def _trigger(self, now_s: float) -> None:
        self._last_trigger_time = now_s
        self._danger_streak = 0  

    def _clear_target(self) -> None:
        self._target_cls_id = None
        self._target_center = None
        self._area_history.clear()

    @staticmethod
    def _bbox_center(xyxy: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _select_target(self, detections: List[Detection], frame_shape: Tuple[int, int, int]) -> Optional[Detection]:
        if not detections:
            return None

        # choose detection closest to previous target centre
        if self._target_center is not None:
            prev_cx, prev_cy = self._target_center
            h, w = frame_shape[0], frame_shape[1]
            diag = math.sqrt((w * w) + (h * h))

            best: Optional[Detection] = None
            best_score = float("inf")

            for d in detections:
                cx, cy = self._bbox_center(d.xyxy)
                dist = math.hypot(cx - prev_cx, cy - prev_cy) / max(1e-6, diag)

                # Hard reject if too far away (likely different object)
                if dist > self.match_max_dist_norm:
                    continue

                # Prefer same class by giving it a bonus (lower score is better)
                class_penalty = 0.0 if (self._target_cls_id is None or d.cls_id == self._target_cls_id) else 0.05
                score = dist + class_penalty

                if score < best_score:
                    best_score = score
                    best = d

            if best is not None:
                return best

        # Fallback: pick most urgent by (closeness, conf)
        return max(detections, key=lambda d: (d.closeness, d.conf))

    def _update_target_tracking(self, target: Detection, frame_shape: Tuple[int, int, int], area_norm: float) -> None:
        if self._target_cls_id is not None and target.cls_id != self._target_cls_id:
            self._area_history.clear()

        self._target_cls_id = target.cls_id
        self._target_center = self._bbox_center(target.xyxy)

        self._area_history.append(float(area_norm))
        if len(self._area_history) > self.history_len:
            self._area_history.pop(0)

    @staticmethod
    def _bbox_area_norm(xyxy: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> float:
        h, w = frame_shape[0], frame_shape[1]
        x1, y1, x2, y2 = xyxy
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        return (bw * bh) / max(1.0, (w * h))

    def _compute_approach_score(self) -> float:
        """
        Compute approach score from area history.
        Uses area_now - area_Nframes_ago.
        """
        if len(self._area_history) < 2:
            return 0.0

        # Compare current area to earliest in window
        area_now = self._area_history[-1]
        area_then = self._area_history[0]
        frames = max(1, len(self._area_history) - 1)

        growth_per_frame = (area_now - area_then) / frames
        if growth_per_frame <= 0:
            return 0.0

        score = growth_per_frame / self.growth_ref
        return float(max(0.0, min(1.0, score)))

    def _apply_speed_scaling(self, risk_base: float, speed_mps: float) -> float:
        """
        Increase risk slightly with speed.
        Example: at speed_ref -> multiplier ~ 1 + speed_risk_gain
        """
        speed_ratio = float(max(0.0, speed_mps) / self.speed_ref_mps)
        speed_ratio = min(speed_ratio, 2.0)  # cap effect
        multiplier = 1.0 + (self.speed_risk_gain * speed_ratio)
        risk = risk_base * multiplier
        return float(max(0.0, min(1.0, risk)))
