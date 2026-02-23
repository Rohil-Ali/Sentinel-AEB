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

    debounce_frames: int = 3
    release_debounce_frames: int = 5

    enabled: bool = True
    speed_weight: float = 0.45
    speed_fear_cap_mph: float = 35.0

    min_brake_during_event: float = 0.6


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
        self._safe_streak = 0
        self._trigger_thresh = 0.0

        self._last_closest: Optional[Detection] = None

        
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

        speed_norm = min(1.0, speed_mph / max(1e-6, self.cfg.speed_fear_cap_mph))

        # calculate danger score
        danger_score = closeness * (1.0 + speed_norm * self.cfg.speed_weight)
        danger_score = float(max(0.0, min(1.0, danger_score)))        

        effective_hard_threshold = self.cfg.hard_brake_threshold - (0.15 * speed_norm)
        effective_hard_threshold = float(max(0.20, min(self.cfg.hard_brake_threshold, effective_hard_threshold)))

        
        # ----- hard brake -----
        if danger_score >= effective_hard_threshold:
            self.state = AEBState.BRAKING
            self._trigger_braking(effective_hard_threshold)
            return self._decision(
                brake=1.0,
                danger_score=danger_score,
                reason=f"Hard Brake (triggered at threshold={self._trigger_thresh:.2f})",
                closest=closest,
            )
        
        # ------ hold brake until danger clears ------
        if self.state == AEBState.BRAKING:
            if danger_score < (self.cfg.soft_brake_threshold - 0.10):
                self._safe_streak += 1
                self._danger_streak = 0

                # release brake if safe
                if self._safe_streak >= self.cfg.release_debounce_frames:
                    self.state = AEBState.SCANNING
                    self._safe_streak = 0
                    return self._decision(
                        brake=0.0,
                        danger_score=danger_score,
                        reason=f"Brake released (safe for {self.cfg.release_debounce_frames}f)",
                        closest=closest,
                    )
                
                # in release debounce - still breaking at the minimum brake
                return self._decision(
                    brake=self.cfg.min_brake_during_event,
                    danger_score=danger_score,
                    reason=f"Braking (safe streak {self._safe_streak}/{self.cfg.release_debounce_frames})",
                    closest=closest,
                )
            else:
                # danger still present 
                self._safe_streak = 0
                brake, reason = self._get_brake_score(danger_score, effective_hard_threshold)

                if brake < self.cfg.min_brake_during_event:
                    brake = self.cfg.min_brake_during_event
                    reason += " (maintaining minimum brake during event)"

                return self._decision(
                    brake=brake,
                    danger_score=danger_score,
                    reason=f"Braking ({reason})",
                    closest=closest,
                )

        # ----- Debounce for danger scanning ------
        if danger_score >= self.cfg.soft_brake_threshold:
            self._danger_streak += 1
            self._safe_streak = 0
        else:
            self._danger_streak = 0

        # Trigger braking only if danger has persisted enough frames
        if self._danger_streak >= self.cfg.debounce_frames:
            brake, reason = self._get_brake_score(danger_score, effective_hard_threshold)
            self.state = AEBState.BRAKING
            self._trigger_braking(effective_hard_threshold)

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
        self._safe_streak = 0
        self._last_closest = None

    def _trigger_braking(self, threshold: float):
        self._danger_streak = 0
        self._safe_streak = 0
        self._trigger_thresh = threshold

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
            return (self.cfg.min_brake_during_event + 0.5 * t, "Soft Brake")
        
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
