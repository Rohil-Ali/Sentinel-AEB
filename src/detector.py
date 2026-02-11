# detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict

import numpy as np
import cv2
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    closeness: float                 # 0..1 proxy score (bigger = closer/more urgent)


class YOLODetector:
    """
    Production YOLO wrapper.

    Responsibilities:
      - Load model once
      - Run inference on frames
      - Filter detections by class + confidence
      - Provide a "closeness proxy" for downstream AEB logic
    """

    def __init__(
            self, 
            weights_path: str = "yolov8n.pt", 
            conf_thresh: float = 0.25, 
            iou_thresh: float = 0.45, 
            target_classes: Optional[Iterable[str]] = None, 
            device: Optional[str] = None,
            use_corridor: bool = True,
            corridor_x_min: float = 0.30,
            corridor_x_max: float = 0.70,
            min_bottom_y: float = 0.45,
            ):
        
        self.model = YOLO(weights_path)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)

        self.names: Dict[int, str] = self.model.names  

        if target_classes is None:
            target_classes = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]

        self.target_classes = set(target_classes)
        self.target_class_ids = {cid for cid, name in self.names.items() if name in self.target_classes}

        self.device = device

        self.use_corridor = bool(use_corridor)
        self.corridor_x_min = float(corridor_x_min)
        self.corridor_x_max = float(corridor_x_max)
        self.min_bottom_y = float(min_bottom_y)

    def set_thresholds(self, conf: float, iou: float) -> None:
        self.conf_thresh = float(conf)
        self.iou_thresh = float(iou)

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame and return filtered detections.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        
        results = self.model.predict(source=frame_bgr, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False, device=self.device)

        r0 = results[0]
        detections: List[Detection] = []

        if r0.boxes is None or len(r0.boxes) == 0:
            return detections

        boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)

        for xyxy_f, conf, cls_id in zip(boxes_xyxy, confs, clss):
            if cls_id not in self.target_class_ids:
                continue

            cls_name = self.names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = [int(v) for v in xyxy_f.tolist()]
            xyxy = (x1, y1, x2, y2)

            if self.use_corridor and not self._in_forward_corridor(xyxy, frame_bgr.shape):
                continue

            closeness = self._closeness_proxy(xyxy, frame_bgr.shape)

            detections.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=cls_name,
                    conf=float(conf),
                    xyxy=xyxy,
                    closeness=closeness,
                )
            )

        detections.sort(key=lambda d: (d.closeness, d.conf), reverse=True)
        return detections

    # ---------- static utils / helper functions ----------
    @staticmethod
    def _closeness_proxy(xyxy: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> float:
        """
        Heuristic closeness proxy in [0..1].

        Intuition:
          - Bigger bbox area => closer
          - Lower bbox bottom (closer to bottom of frame) => closer

        This is NOT real distance; it's a proxy useful for triggering logic.
        """
        h, w = frame_shape[0], frame_shape[1]
        x1, y1, x2, y2 = xyxy

        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)

        area_norm = (bw * bh) / max(1, (w * h))
        bottom_norm = y2 / max(1, h)
        score = (0.65 * area_norm) + (0.35 * bottom_norm)

        return float(max(0.0, min(1.0, score)))
    
    def _in_forward_corridor(self, xyxy: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> bool:
        '''
        Check if the detection is within the defined forward corridor.
        '''
        h, w = frame_shape[0], frame_shape[1]
        x1, y1, x2, y2 = xyxy

        x_center = (x1 + x2) / 2.0
        x_center_norm = x_center / max(1, w)
        bottom_norm = y2 / max(1, h)

        in_x = self.corridor_x_min <= x_center_norm <= self.corridor_x_max
        in_y = bottom_norm >= self.min_bottom_y

        return in_x and in_y
    
    def draw_corridor(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Debug helper: draw the forward corridor ROI on the frame.
        """
        out = frame_bgr.copy()
        h, w = out.shape[0], out.shape[1]

        x_min = int(self.corridor_x_min * w)
        x_max = int(self.corridor_x_max * w)
        y_min = int(self.min_bottom_y * h)

        cv2.rectangle(out, (x_min, y_min), (x_max, h - 1), (255, 255, 255), 2)
        cv2.putText(out, "FORWARD CORRIDOR", (x_min, max(20, y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    @staticmethod
    def draw_detections(frame_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Returns a copy of the frame with drawn boxes + labels (for debug/testing).
        """
        out = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d.xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d.cls_name} {d.conf:.2f} close:{d.closeness:.2f}"
            cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out

