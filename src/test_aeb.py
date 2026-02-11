# run_aeb_carla_demo.py
"""
CARLA AEB controller demo:
- Uses CarlaAdapter to get frames + speed
- Uses YOLODetector to get detections
- Uses AEBController to decide braking
- Overlays debug info so you can tune thresholds

Controls:
  Q = quit
  C = toggle forward corridor filter
  B = toggle AEB enabled
  R/F = increase/decrease conf threshold
  T/G = increase/decrease soft threshold
  Y/H = increase/decrease hard threshold
  U/J = increase/decrease growth_ref
  1/2 = set brake mode (1 = soft only, 2 = soft+hard)
"""

from __future__ import annotations

import time
import cv2

from carla_adapter import CarlaAdapter
from detector import YOLODetector
from aeb_controller import AEBController, VehicleKinematics, BrakeCommand


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def main():
    # --- Adapter ---
    adapter = CarlaAdapter(autopilot=True, image_width=960, image_height=540)
    adapter.start()

    # --- Detector ---
    detector = YOLODetector(
        weights_path="yolov8n.pt",
        conf_thresh=0.25,
        iou_thresh=0.45,
        use_corridor=True,
        corridor_x_min=0.30,
        corridor_x_max=0.70,
        min_bottom_y=0.45,
    )

    # --- Controller ---
    controller = AEBController(
        debounce_frames=3,
        cooldown_s=0.8,
        soft_threshold=0.45,
        hard_threshold=0.65,
        growth_ref=0.01,          # start conservative; tune with U/J
        speed_risk_gain=0.25,
        soft_brake=0.35,
        hard_brake=1.0,
    )

    aeb_enabled = True
    corridor_enabled = detector.use_corridor
    allow_hard_brake = True

    last_print = 0.0
    fps_counter = 0
    fps_start = time.perf_counter()
    fps = 0.0

    print("✅ AEB demo running. Press Q to quit.")

    try:
        while True:
            frame = adapter.get_frame()
            if frame is None:
                continue

            # FPS calc
            fps_counter += 1
            dt = time.perf_counter() - fps_start
            if dt >= 1.0:
                fps = fps_counter / dt
                fps_counter = 0
                fps_start = time.perf_counter()

            # Vehicle state
            state = adapter.get_state()
            speed_mps = state.speed_mps if state else 0.0
            kin = VehicleKinematics(speed_mps=speed_mps)

            # Detection + decision
            detector.use_corridor = corridor_enabled
            detections = detector.predict(frame)
            decision = controller.update(detections, frame.shape, kin)

            # Apply braking
            applied_brake = 0.0
            if aeb_enabled:
                if decision.command == BrakeCommand.SOFT_BRAKE:
                    applied_brake = decision.brake
                    adapter.apply_control(throttle=0.0, brake=applied_brake)
                elif decision.command == BrakeCommand.HARD_BRAKE and allow_hard_brake:
                    applied_brake = decision.brake
                    adapter.apply_control(throttle=0.0, brake=applied_brake)

            # Visualize
            vis = frame
            vis = detector.draw_corridor(vis) if corridor_enabled else vis
            vis = detector.draw_detections(vis, detections)

            # HUD text
            h, w = vis.shape[:2]
            lines = [
                f"AEB: {'ON' if aeb_enabled else 'OFF'}   Corridor: {'ON' if corridor_enabled else 'OFF'}   FPS: {fps:.1f}",
                f"Speed: {speed_mps*2.23693629:.1f} mph   CtrlBrake: {state.brake:.2f}" if state else "Speed: n/a",
                f"Decision: {decision.state} / {decision.command}   AppliedBrake: {applied_brake:.2f}",
                f"Risk: {decision.risk:.2f}  CloseNow: {decision.closeness_now:.2f}  Approach: {decision.approach_score:.2f}",
                f"conf={detector.conf_thresh:.2f}  soft={controller.soft_threshold:.2f}  hard={controller.hard_threshold:.2f}  growth_ref={controller.growth_ref:.4f}",
                f"Reason: {decision.reason}",
            ]

            y = 25
            for line in lines:
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 24

            cv2.imshow("Sentinel AEB - CARLA Demo (tune live)", vis)

            # Optional console status every ~1s
            now = time.time()
            if now - last_print > 1.0:
                last_print = now
                top = detections[0].cls_name if detections else "none"
                print(f"Top={top:>10}  speed={speed_mps*2.2369:5.1f}mph  risk={decision.risk:.2f}  cmd={decision.command}")

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

            elif key in (ord("b"), ord("B")):
                aeb_enabled = not aeb_enabled

            elif key in (ord("c"), ord("C")):
                corridor_enabled = not corridor_enabled

            # conf threshold
            elif key in (ord("r"), ord("R")):
                detector.conf_thresh = clamp(detector.conf_thresh + 0.05, 0.05, 0.95)
            elif key in (ord("f"), ord("F")):
                detector.conf_thresh = clamp(detector.conf_thresh - 0.05, 0.05, 0.95)

            # soft/hard thresholds
            elif key in (ord("t"), ord("T")):
                controller.soft_threshold = clamp(controller.soft_threshold + 0.02, 0.10, 0.95)
            elif key in (ord("g"), ord("G")):
                controller.soft_threshold = clamp(controller.soft_threshold - 0.02, 0.10, 0.95)

            elif key in (ord("y"), ord("Y")):
                controller.hard_threshold = clamp(controller.hard_threshold + 0.02, 0.10, 0.99)
            elif key in (ord("h"), ord("H")):
                controller.hard_threshold = clamp(controller.hard_threshold - 0.02, 0.10, 0.99)

            # growth_ref tuning
            elif key in (ord("u"), ord("U")):
                controller.growth_ref = clamp(controller.growth_ref + 0.002, 0.001, 0.05)
            elif key in (ord("j"), ord("J")):
                controller.growth_ref = clamp(controller.growth_ref - 0.002, 0.001, 0.05)

            # brake mode
            elif key == ord("1"):
                allow_hard_brake = False
            elif key == ord("2"):
                allow_hard_brake = True

    finally:
        adapter.stop()
        cv2.destroyAllWindows()
        print("🧹 Cleaned up.")


if __name__ == "__main__":
    main()
