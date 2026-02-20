from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from detector import Detection


class AEBState(str, Enum):
    SCANNING = "SCANNING"
    BRAKING = "BRAKING"


@dataclass
class AEBDecision:
    state: AEBState
    brake: float          
    reason: str            
    danger_score: float    
    closest_class: Optional[str] = None
    closest_conf: Optional[float] = None
    closest_closeness: Optional[float] = None


@dataclass
class AEBConfig:
    soft_brake_threshold: float = 0.25   
    hard_brake_threshold: float = 0.40 

    debounce_frames: int = 1
    cooldown_frames: int = 20

    enabled: bool = True

    speed_weight: float = 0.30
    speed_mph_for_max: float = 35.0

    min_brake_during_event: float = 0.65


class AEBController:
    """
    AEB decision logic.

    Inputs:
      - detections: List[Detection] (already filtered, with closeness proxy)
      - speed_mph: float (from adapter)

    Output:
      - AEBDecision (state + brake value)
    """

    def __init__(self, config: Optional[AEBConfig] = None):
        self.cfg = config or AEBConfig()
        self.state: AEBState = AEBState.SCANNING

        self._danger_streak = 0
        self._cooldown_left = 0
        self._last_closest: Optional[Detection] = None

        self._in_braking_event = False

    def set_enabled(self, enabled: bool) -> None:
        self.cfg.enabled = bool(enabled)
        if not self.cfg.enabled:
            self.reset_states() # Reset to a clean state when turned off

    def update(self, detections: List[Detection], speed_mph: float) -> AEBDecision:
        if not self.cfg.enabled:
            return AEBDecision(
                state=AEBState.SCANNING,
                brake=0.0,
                reason="System disabled",
                danger_score=0.0,
            )

        speed_mph = float(max(0.0, speed_mph))

        closest = detections[0] if detections else None
        self._last_closest = closest
        closeness = closest.closeness if closest else 0.0

        speed_norm = min(1.0, speed_mph / max(1e-6, self.cfg.speed_mph_for_max))

        # calculate danger score
        danger_score = closeness * (1.0 + speed_norm * self.cfg.speed_weight)
        danger_score = float(max(0.0, min(1.0, danger_score)))        

        # ------ Hard brake - no debounce  ------
        # effective_hard_threshold = self.cfg.hard_brake_threshold
        # if speed_mph >= 30.0:
        #     effective_hard_threshold = 0.35  # be more aggressive at higher speeds
        effective_hard_threshold = self.cfg.hard_brake_threshold - (0.12 * speed_norm)
        effective_hard_threshold = float(max(0.20, min(self.cfg.hard_brake_threshold, effective_hard_threshold)))

        
        # ----- hard brake -----
        if danger_score >= effective_hard_threshold:
            self.state = AEBState.BRAKING
            self._trigger_braking()
            return self._decision(
                brake=1.0,
                danger_score=danger_score,
                reason=f"Hard Brake (threshold={effective_hard_threshold:.2f})",
                closest=closest,
            )


        # ------ Cooldown ------
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

            # Decide brake strength based on current danger
            brake, reason = self._get_brake_score(danger_score, effective_hard_threshold)

            # keeps braking until danger leaves
            if self._in_braking_event:
                brake = max(brake, self.cfg.min_brake_during_event)  # ensure we keep braking during cooldown

            if speed_mph < 5.0 and danger_score < 0.15:
                self._in_braking_event = False  # reset event if we're basically stopped and clear
                brake = 0.0
                self._cooldown_left = 0 # exit cooldown early if we're safe and slow

            self.state = AEBState.BRAKING
            return self._decision(
                brake=brake,
                danger_score=danger_score,
                reason=f"Cooldown ({self._cooldown_left}f, brake={brake:.2f}, reason={reason})",
                closest=closest,
            )

        # ----- Debounce ------
        if danger_score >= self.cfg.soft_brake_threshold:
            self._danger_streak += 1
        else:
            self._danger_streak = 0

        # Trigger braking only if danger has persisted enough frames
        if self._danger_streak >= self.cfg.debounce_frames:
            brake, reason = self._get_brake_score(danger_score, effective_hard_threshold)
            self.state = AEBState.BRAKING
            self._trigger_braking()

            return self._decision(
                brake=brake,
                danger_score=danger_score,
                reason=f"Danger persisted {self.cfg.debounce_frames} frames, reason={reason}",
                closest=closest,
            )

        # Otherwise scanning
        self.state = AEBState.SCANNING
        return self._decision(
            brake=0.0,
            danger_score=danger_score,
            reason=f"Scanning (streak {self._danger_streak}/{self.cfg.debounce_frames})",
            closest=closest,
        )

 
    
    # --------- helper ----------

    def reset_states(self) -> None:
        self.state = AEBState.SCANNING
        self._danger_streak = 0
        self._cooldown_left = 0
        self._last_closest = None
        self._in_braking_event = False


    def _trigger_braking(self):
        self._cooldown_left = self.cfg.cooldown_frames
        self._danger_streak = 0
        self._in_braking_event = True

    def _get_brake_score(self, danger_score: float, hard_threshold: float) -> Tuple[float, str]:
        """
        Map danger_score to brake intensity.
        """
        if danger_score >= hard_threshold:
            return (1.0, "Hard Brake")
        
        if danger_score >= self.cfg.soft_brake_threshold:
            # Scale from 0.6 to 1.0
            t = (danger_score - self.cfg.soft_brake_threshold) / max(
                1e-6, (hard_threshold - self.cfg.soft_brake_threshold)
            )
            return (0.6 + 0.4 * t, "Soft Brake")
        
        return (0.0, "No Brake")
        

        

    def _decision(self, brake: float, danger_score: float, reason: str, closest: Optional[Detection]) -> AEBDecision:
        return AEBDecision(
            state=self.state,
            brake=float(max(0.0, min(1.0, brake))),
            reason=reason,
            danger_score=float(max(0.0, min(1.0, danger_score))),
            closest_class=closest.cls_name if closest else None,
            closest_conf=closest.conf if closest else None,
            closest_closeness=closest.closeness if closest else None,
        )
