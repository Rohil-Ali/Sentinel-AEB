from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import customtkinter as ctk
from PIL import Image

from carla_adapter import CarlaAdapter
from detector import YOLODetector
from aeb_controller import AEBController, AEBState


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


@dataclass
class RuntimeSnapshot:
    frame_bgr: Optional[any] = None
    fps: float = 0.0
    latency_ms: float = 0.0
    speed_mph: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    danger_score: float = 0.0
    status_text: str = "SCANNING"
    closest_class: str = "None"
    closest_conf: float = 0.0
    closest_closeness: float = 0.0
    reason: str = ""
    system_enabled: bool = True
    source_name: str = "CARLA"


class SentinelAEBApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Sentinel AEB")
        self.geometry("1400x860")
        self.minsize(1200, 760)

        # Backend modules
        self.adapter = CarlaAdapter(autopilot=False)
        self.detector = YOLODetector()
        self.controller = AEBController()

        # Runtime state
        self.snapshot = RuntimeSnapshot()
        self.snapshot_lock = threading.Lock()

        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.latest_ctk_image = None

        # Engineer mode
        self.engineer_unlocked = False

        self._build_layout()
        self._start_system()

        self.after(50, self._refresh_ui)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --------------------------------------------------
    # UI LAYOUT
    # --------------------------------------------------

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left side
        self.left_frame = ctk.CTkFrame(self, corner_radius=18)
        self.left_frame.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)

        self.title_label = ctk.CTkLabel(
            self.left_frame,
            text="Sentinel AEB",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self.video_label = ctk.CTkLabel(
            self.left_frame,
            text="Waiting for video feed...",
            width=960,
            height=540,
        )
        self.video_label.grid(row=1, column=0, padx=16, pady=8, sticky="nsew")

        self.status_frame = ctk.CTkFrame(self.left_frame, corner_radius=16)
        self.status_frame.grid(row=2, column=0, padx=16, pady=(8, 16), sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Status: SCANNING",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.status_label.grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")

        self.reason_label = ctk.CTkLabel(
            self.status_frame,
            text="Reason: N/A",
            wraplength=850,
            justify="left",
        )
        self.reason_label.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="w")

        # Right side
        self.right_frame = ctk.CTkFrame(self, corner_radius=18)
        self.right_frame.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self.right_frame, corner_radius=16)
        self.tabview.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")
        self.tabview.add("Driver View")
        self.tabview.add("Engineer Mode")

        self.driver_tab = self.tabview.tab("Driver View")
        self.engineer_tab = self.tabview.tab("Engineer Mode")

        self._build_driver_tab()
        self._build_engineer_tab()

    def _build_driver_tab(self) -> None:
        self.driver_tab.grid_columnconfigure(0, weight=1)

        self.system_toggle = ctk.CTkSwitch(
            self.driver_tab,
            text="AEB System Enabled",
            command=self._on_system_toggle,
        )
        self.system_toggle.select()
        self.system_toggle.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self.speed_label = ctk.CTkLabel(self.driver_tab, text="Speed: 0.0 mph")
        self.speed_label.grid(row=1, column=0, padx=16, pady=8, sticky="w")

        self.closest_label = ctk.CTkLabel(
            self.driver_tab,
            text="Closest Object: None",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.closest_label.grid(row=2, column=0, padx=16, pady=8, sticky="w")

        self.brake_label = ctk.CTkLabel(self.driver_tab, text="Brake Force")
        self.brake_label.grid(row=3, column=0, padx=16, pady=(12, 4), sticky="w")

        self.brake_bar = ctk.CTkProgressBar(self.driver_tab, height=18)
        self.brake_bar.grid(row=4, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.brake_bar.set(0.0)

        self.throttle_label = ctk.CTkLabel(self.driver_tab, text="Throttle")
        self.throttle_label.grid(row=5, column=0, padx=16, pady=(4, 4), sticky="w")

        self.throttle_bar = ctk.CTkProgressBar(self.driver_tab, height=18)
        self.throttle_bar.grid(row=6, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.throttle_bar.set(0.0)

        self.proximity_label = ctk.CTkLabel(self.driver_tab, text="Proximity / Danger")
        self.proximity_label.grid(row=7, column=0, padx=16, pady=(4, 4), sticky="w")

        self.proximity_bar = ctk.CTkProgressBar(self.driver_tab, height=18)
        self.proximity_bar.grid(row=8, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.proximity_bar.set(0.0)

        self.source_label = ctk.CTkLabel(self.driver_tab, text="Input Source: CARLA")
        self.source_label.grid(row=9, column=0, padx=16, pady=8, sticky="w")

    def _build_engineer_tab(self) -> None:
        self.engineer_tab.grid_columnconfigure(0, weight=1)

        self.unlock_frame = ctk.CTkFrame(self.engineer_tab, corner_radius=16)
        self.unlock_frame.grid(row=0, column=0, padx=16, pady=16, sticky="ew")
        self.unlock_frame.grid_columnconfigure(0, weight=1)

        self.password_label = ctk.CTkLabel(self.unlock_frame, text="Engineer Mode Password")
        self.password_label.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        self.password_entry = ctk.CTkEntry(self.unlock_frame, show="*")
        self.password_entry.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")

        self.unlock_button = ctk.CTkButton(self.unlock_frame, text="Unlock", command=self._unlock_engineer_mode)
        self.unlock_button.grid(row=1, column=1, padx=12, pady=(0, 12))

        self.unlock_status = ctk.CTkLabel(self.unlock_frame, text="Locked")
        self.unlock_status.grid(row=2, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="w")

        self.controls_frame = ctk.CTkFrame(self.engineer_tab, corner_radius=16)
        self.controls_frame.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        self.rain_title = ctk.CTkLabel(self.controls_frame, text="Rain Intensity")
        self.rain_title.grid(row=0, column=0, padx=12, pady=(16, 4), sticky="w")

        self.rain_slider = ctk.CTkSlider(self.controls_frame, from_=0, to=100, command=self._on_rain_change)
        self.rain_slider.grid(row=1, column=0, padx=12, pady=(0, 4), sticky="ew")
        self.rain_slider.set(0)

        self.rain_value = ctk.CTkLabel(self.controls_frame, text="0")
        self.rain_value.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="w")

        self.source_title = ctk.CTkLabel(self.controls_frame, text="Input Source")
        self.source_title.grid(row=3, column=0, padx=12, pady=(8, 4), sticky="w")

        self.source_menu = ctk.CTkOptionMenu(
            self.controls_frame,
            values=["CARLA", "Webcam (Not Implemented)"],
            command=self._on_source_change,
        )
        self.source_menu.grid(row=4, column=0, padx=12, pady=(0, 12), sticky="ew")
        self.source_menu.set("CARLA")

        self.conf_title = ctk.CTkLabel(self.controls_frame, text="AI Confidence Threshold")
        self.conf_title.grid(row=5, column=0, padx=12, pady=(8, 4), sticky="w")

        self.conf_slider = ctk.CTkSlider(
            self.controls_frame,
            from_=0.10,
            to=0.90,
            command=self._on_confidence_change,
        )
        self.conf_slider.grid(row=6, column=0, padx=12, pady=(0, 4), sticky="ew")
        self.conf_slider.set(self.detector.conf_thresh)

        self.conf_value = ctk.CTkLabel(self.controls_frame, text=f"{self.detector.conf_thresh:.2f}")
        self.conf_value.grid(row=7, column=0, padx=12, pady=(0, 12), sticky="w")

        self.health_title = ctk.CTkLabel(
            self.controls_frame,
            text="System Health",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.health_title.grid(row=8, column=0, padx=12, pady=(16, 8), sticky="w")

        self.fps_label = ctk.CTkLabel(self.controls_frame, text="FPS: 0.0")
        self.fps_label.grid(row=9, column=0, padx=12, pady=4, sticky="w")

        self.latency_label = ctk.CTkLabel(self.controls_frame, text="Latency: 0.0 ms")
        self.latency_label.grid(row=10, column=0, padx=12, pady=(4, 16), sticky="w")

        self._set_engineer_controls_enabled(False)

    # --------------------------------------------------
    # BACKEND
    # --------------------------------------------------

    def _start_system(self) -> None:
        self.adapter.start()
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self) -> None:
        frame_counter = 0
        fps_start = time.time()

        while self.running:
            loop_start = time.perf_counter()

            try:
                frame = self.adapter.get_frame()
                vehicle_state = self.adapter.get_state()

                if frame is None:
                    time.sleep(0.01)
                    continue

                speed_mph = vehicle_state.speed_mph if vehicle_state else 0.0
                throttle = vehicle_state.throttle if vehicle_state else 0.0
                brake = vehicle_state.brake if vehicle_state else 0.0

                detections = self.detector.predict(frame)
                decision = self.controller.update(detections, speed_mph=speed_mph)

                # Draw overlays
                vis = frame
                if self.detector.use_corridor:
                    vis = self.detector.draw_corridor(vis)
                vis = self.detector.draw_detections(vis, detections)

                # Apply braking logic
                if decision.brake > 0.0 and self.controller.cfg.enabled:
                    self.adapter.apply_control(throttle=0.0, brake=decision.brake, steer=0.0)
                    applied_brake = decision.brake
                    applied_throttle = 0.0
                else:
                    applied_brake = brake
                    applied_throttle = throttle

                # FPS
                frame_counter += 1
                now = time.time()
                elapsed = now - fps_start
                current_fps = self.snapshot.fps

                if elapsed >= 1.0:
                    current_fps = frame_counter / elapsed
                    frame_counter = 0
                    fps_start = now

                latency_ms = (time.perf_counter() - loop_start) * 1000.0

                with self.snapshot_lock:
                    self.snapshot.frame_bgr = vis
                    self.snapshot.fps = current_fps
                    self.snapshot.latency_ms = latency_ms
                    self.snapshot.speed_mph = speed_mph
                    self.snapshot.throttle = applied_throttle
                    self.snapshot.brake = applied_brake
                    self.snapshot.danger_score = decision.danger_score
                    self.snapshot.status_text = decision.state.value
                    self.snapshot.closest_class = decision.closest_class or "None"
                    self.snapshot.closest_conf = decision.closest_conf or 0.0
                    self.snapshot.closest_closeness = decision.closest_closeness or 0.0
                    self.snapshot.reason = decision.reason
                    self.snapshot.system_enabled = self.controller.cfg.enabled
                    self.snapshot.source_name = "CARLA"

            except Exception as exc:
                with self.snapshot_lock:
                    self.snapshot.reason = f"Runtime error: {exc}"

            time.sleep(0.01)

    # --------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------

    def _on_system_toggle(self) -> None:
        self.controller.set_enabled(bool(self.system_toggle.get()))

    def _unlock_engineer_mode(self) -> None:
        if self.password_entry.get() == "admin":
            self.engineer_unlocked = True
            self.unlock_status.configure(text="Unlocked")
            self._set_engineer_controls_enabled(True)
        else:
            self.unlock_status.configure(text="Incorrect password")

    def _set_engineer_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.rain_slider.configure(state=state)
        self.source_menu.configure(state=state)
        self.conf_slider.configure(state=state)

    def _on_rain_change(self, value: float) -> None:
        self.rain_value.configure(text=f"{value:.0f}")
        self.adapter.set_rain(float(value))

    def _on_source_change(self, value: str) -> None:
        # Placeholder for future webcam support
        self.source_label.configure(text=f"Input Source: {value}")

    def _on_confidence_change(self, value: float) -> None:
        self.conf_value.configure(text=f"{value:.2f}")
        self.detector.set_thresholds(conf=float(value), iou=self.detector.iou_thresh)

    # --------------------------------------------------
    # UI UPDATE
    # --------------------------------------------------

    def _refresh_ui(self) -> None:
        with self.snapshot_lock:
            snap = RuntimeSnapshot(
                frame_bgr=self.snapshot.frame_bgr.copy() if self.snapshot.frame_bgr is not None else None,
                fps=self.snapshot.fps,
                latency_ms=self.snapshot.latency_ms,
                speed_mph=self.snapshot.speed_mph,
                throttle=self.snapshot.throttle,
                brake=self.snapshot.brake,
                danger_score=self.snapshot.danger_score,
                status_text=self.snapshot.status_text,
                closest_class=self.snapshot.closest_class,
                closest_conf=self.snapshot.closest_conf,
                closest_closeness=self.snapshot.closest_closeness,
                reason=self.snapshot.reason,
                system_enabled=self.snapshot.system_enabled,
                source_name=self.snapshot.source_name,
            )

        if snap.frame_bgr is not None:
            rgb = cv2.cvtColor(snap.frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(960, 540))
            self.latest_ctk_image = ctk_img
            self.video_label.configure(image=ctk_img, text="")

        self.status_label.configure(text=f"Status: {snap.status_text}")
        self.reason_label.configure(text=f"Reason: {snap.reason}")

        self.speed_label.configure(text=f"Speed: {snap.speed_mph:.1f} mph")
        self.closest_label.configure(
            text=f"Closest Object: {snap.closest_class} | conf={snap.closest_conf:.2f} | close={snap.closest_closeness:.2f}"
        )

        self.brake_bar.set(max(0.0, min(1.0, snap.brake)))
        self.throttle_bar.set(max(0.0, min(1.0, snap.throttle)))
        self.proximity_bar.set(max(0.0, min(1.0, snap.danger_score)))

        self.source_label.configure(text=f"Input Source: {snap.source_name}")
        self.fps_label.configure(text=f"FPS: {snap.fps:.1f}")
        self.latency_label.configure(text=f"Latency: {snap.latency_ms:.1f} ms")

        self.after(50, self._refresh_ui)

    # --------------------------------------------------
    # CLOSE
    # --------------------------------------------------

    def _on_close(self) -> None:
        self.running = False
        try:
            self.adapter.stop()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = SentinelAEBApp()
    app.mainloop()