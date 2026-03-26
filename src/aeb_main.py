"""
aeb_main.py – AEB System GUI (CustomTkinter)

interface for the Autonomous Emergency Braking system.
Integrates: CarlaAdapter | YOLODetector | AEBController

Features:
  - Live video feed (CARLA or webcam)
  - System status indicator (SCANNING / BRAKING)
  - Brake & throttle gauges
  - Proximity danger bar
  - Speed display
  - FPS / latency monitor
  - Engineer Mode (password-protected) with:
      • Rain intensity slider
      • Input source selector
      • Confidence threshold slider
      • AEB enable/disable toggle
"""

from __future__ import annotations

import threading
import time
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image, ImageTk
import hashlib

import customtkinter as ctk

# -------- project modules --------
from carla_adapter import CarlaAdapter, VehicleState
from webcam_adapter import WebcamAdapter
from detector import YOLODetector, Detection
from aeb_controller import AEBController, AEBConfig, AEBState, AEBDecision


# --------- color palette ---------
_BG_DARK      = "#0D0D0D"
_BG_PANEL     = "#161619"
_BG_CARD      = "#1C1C21"
_BG_INPUT     = "#26262E"
_ACCENT_BLUE  = "#3B82F6"
_ACCENT_CYAN  = "#22D3EE"
_STATUS_GREEN = "#22C55E"
_STATUS_RED   = "#EF4444"
_STATUS_AMBER = "#F59E0B"
_TEXT_PRIMARY  = "#F1F1F4"
_TEXT_DIM      = "#71717A"
_BORDER_SUBTLE = "#2A2A32"

_ENGINEER_PASSWORD_HASH = hashlib.sha256("1234".encode()).hexdigest()
_GUI_REFRESH_MS = 30 # ~33 fps GUI updates


# ---------- helper widgets -----------
class VerticalGauge(ctk.CTkFrame):
    def __init__(
        self,
        master,
        width: int = 36,
        height: int = 140,
        label: str = "",
        fill_colour: str = _ACCENT_BLUE,
        **kw,
    ):
        super().__init__(master, fg_color=_BG_CARD, width=width, height=height, **kw)
        self.pack_propagate(False)
        self._fill = fill_colour
        self._value = 0.0

        self._lbl_top = ctk.CTkLabel(self, text=label, font=("Consolas", 9, "bold"), text_color=_TEXT_DIM, height=16)
        self._lbl_top.pack(pady=(2, 2))

        self._track = ctk.CTkFrame(self, fg_color=_BG_INPUT, corner_radius=3, border_width=1, border_color=_BORDER_SUBTLE)
        self._track.pack(fill="both", expand=True, padx=6)
        self._track.pack_propagate(False)

        self._spacer = ctk.CTkFrame(self._track, fg_color=_BG_INPUT, height=1)
        self._spacer.pack(fill="x", expand=True)

        self._fill_bar = ctk.CTkFrame(self._track, fg_color=fill_colour, corner_radius=2, height=0)
        self._fill_bar.pack(fill="x", side="bottom")

        self._lbl_pct = ctk.CTkLabel(self, text="0%", font=("Consolas", 9), text_color=_TEXT_DIM, height=16)
        self._lbl_pct.pack(pady=(2, 2))

    def set_value(self, v: float):
        self._value = max(0.0, min(1.0, v))
        self._lbl_pct.configure(text=f"{int(self._value * 100)}%")
        track_h = self._track.winfo_height()
        if track_h > 1:
            fill_h = max(0, int(track_h * self._value))
            self._fill_bar.configure(height=fill_h)

class HorizontalBar(ctk.CTkFrame):
    def __init__(self, master, width: int = 260, height: int = 22, **kw):
        super().__init__(
            master, 
            fg_color=_BG_INPUT, 
            width=width, 
            height=height,
            corner_radius=4, 
            border_width=1,
            border_color=_BORDER_SUBTLE, 
            **kw)
        
        self.pack_propagate(False)
        self._value = 0.0

        self._fill_bar = ctk.CTkFrame(self, fg_color=_STATUS_GREEN, corner_radius=3, width=0)
        self._fill_bar.pack(fill="y", side="left")

    def set_value(self, v: float):
        self._value = max(0.0, min(1.0, v))
        total_w = self.winfo_width()
        if total_w > 1:
            fill_w = max(0, int(total_w * self._value))
            colour = self._colour_for_value(self._value)
            self._fill_bar.configure(width=fill_w, fg_color=colour)

    @staticmethod
    def _colour_for_value(v: float) -> str:
        if v < 0.35:
            return _STATUS_GREEN
        if v < 0.55:
            return _STATUS_AMBER
        return _STATUS_RED

class StatusDot(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, fg_color=_BG_CARD, **kw)

        self._dot = ctk.CTkLabel(self, text="●", width=16, font=("Consolas", 14), text_color=_STATUS_GREEN)
        self._dot.pack(side="left", padx=(0, 6))

        self._label = ctk.CTkLabel(self, text="SCANNING", font=("Consolas", 15, "bold"), text_color=_STATUS_GREEN)
        self._label.pack(side="left")

    def set_state(self, state: AEBState):
        if state == AEBState.BRAKING:
            self._label.configure(text="BRAKING", text_color=_STATUS_RED)
            self._dot.configure(text_color=_STATUS_RED)
        else:
            self._label.configure(text="SCANNING", text_color=_STATUS_GREEN)
            self._dot.configure(text_color=_STATUS_GREEN)


# ---------- main application ----------
class AEBApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # window setup
        self.title("AEB System")
        self.geometry("1100x740")
        self.minsize(960, 660)
        self.configure(fg_color=_BG_DARK)
        ctk.set_appearance_mode("dark")

        # shared state (worker: producer, GUI: consumer)
        self._lock = threading.Lock()
        self._display_frame: Optional[np.ndarray] = None
        self._decision = AEBDecision(
            state=AEBState.SCANNING, 
            brake=0.0,
            reason="Initialising…", 
            danger_score=0.0,
        )
        self._vehicle_state: Optional[VehicleState] = None
        self._fps: float = 0.0
        self._latency_ms: float = 0.0
        self._detections: List[Detection] = []

        # modules
        self._adapter = None   # CarlaAdapter | WebcamAdapter
        self._detector = YOLODetector()
        self._controller = AEBController()

        # flags 
        self._source = "carla"  # "carla" | "webcam" #SHOULDNT BE HERE
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._engineer_unlocked = False

        # cruise control state 
        self._map_name: str = "Town04" #SHOULDNT BE HERE

        # build UI 
        self._build_ui()

        # begin GUI polling 
        self.after(_GUI_REFRESH_MS, self._gui_tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):

        # top bar 
        top = ctk.CTkFrame(self, fg_color=_BG_PANEL, height=44, corner_radius=0)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="  ⬢  AEB  SYSTEM", font=("Consolas", 16, "bold"), text_color=_ACCENT_CYAN).pack(side="left", padx=12)

        self._lbl_fps = ctk.CTkLabel(top, text="FPS  —    Latency  — ms", font=("Consolas", 11), text_color=_TEXT_DIM)
        self._lbl_fps.pack(side="right", padx=16)

        self._btn_start = ctk.CTkButton(
            top, text="▶  START", width=100, height=30,
            fg_color=_STATUS_GREEN, hover_color="#16A34A",
            text_color="#000", font=("Consolas", 12, "bold"),
            command=self._toggle_system,
        )
        self._btn_start.pack(side="right", padx=(0, 8))

        # main body
        body = ctk.CTkFrame(self, fg_color=_BG_DARK)
        body.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        # left: video feed
        vid_frame = ctk.CTkFrame(body, fg_color=_BG_PANEL, corner_radius=10)
        vid_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self._video_label = ctk.CTkLabel(vid_frame, text="", fg_color=_BG_DARK, corner_radius=6)
        self._video_label.pack(fill="both", expand=True, padx=6, pady=6)
        self._set_placeholder_frame()

        # right: instrument cluster
        right = ctk.CTkFrame(body, fg_color=_BG_PANEL, width=280, corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.pack_propagate(False)
        self._build_instrument_panel(right)

        # driving controls strip
        self._build_driving_controls()

        # bottom: engineer drawer
        self._build_engineer_bar()

    # --------- instrument panel (right side) ------------
    def _build_instrument_panel(self, parent):

        inner = ctk.CTkFrame(parent, fg_color=_BG_PANEL)
        inner.pack(fill="both", expand=True, padx=10, pady=10)

        # status
        sec_status = ctk.CTkFrame(inner, fg_color=_BG_CARD, corner_radius=8)
        sec_status.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(sec_status, text="SYSTEM STATUS", font=("Consolas", 10), text_color=_TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 2))    
        self._status_dot = StatusDot(sec_status)
        self._status_dot.pack(anchor="w", padx=12, pady=(0, 4))
        self._lbl_reason = ctk.CTkLabel(sec_status, text="—",font=("Consolas", 10), text_color=_TEXT_DIM, wraplength=240, justify="left")
        self._lbl_reason.pack(anchor="w", padx=12, pady=(0, 8))

        # speed
        sec_speed = ctk.CTkFrame(inner, fg_color=_BG_CARD, corner_radius=8)
        sec_speed.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(sec_speed, text="SPEED", font=("Consolas", 10), text_color=_TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 0))
        self._lbl_speed = ctk.CTkLabel(sec_speed, text="0", font=("Consolas", 36, "bold"), text_color=_TEXT_PRIMARY)
        self._lbl_speed.pack(anchor="w", padx=12, pady=(0, 0))
        ctk.CTkLabel(sec_speed, text="MPH", font=("Consolas", 10), text_color=_TEXT_DIM).pack(anchor="w", padx=14, pady=(0, 8))

        # proximity bar
        sec_prox = ctk.CTkFrame(inner, fg_color=_BG_CARD, corner_radius=8)
        sec_prox.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(sec_prox, text="PROXIMITY", font=("Consolas", 10), text_color=_TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 4))
        self._prox_bar = HorizontalBar(sec_prox, width=240, height=22)
        self._prox_bar.pack(padx=12, pady=(0, 8))

        # gauges row
        sec_gauges = ctk.CTkFrame(inner, fg_color=_BG_CARD, corner_radius=8)
        sec_gauges.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(sec_gauges, text="PEDALS", font=("Consolas", 10), text_color=_TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 4))

        gauge_row = ctk.CTkFrame(sec_gauges, fg_color=_BG_CARD)
        gauge_row.pack(padx=12, pady=(0, 10))

        self._gauge_brake = VerticalGauge(gauge_row, label="BRK", fill_colour=_STATUS_RED, height=120)
        self._gauge_brake.pack(side="left", padx=(0, 16))

        self._gauge_throttle = VerticalGauge(gauge_row, label="THR", fill_colour=_STATUS_GREEN, height=120)
        self._gauge_throttle.pack(side="left", padx=(0, 16))

        # detection info 
        sec_det = ctk.CTkFrame(inner, fg_color=_BG_CARD, corner_radius=8)
        sec_det.pack(fill="x", pady=(0, 0))
        ctk.CTkLabel(sec_det, text="CLOSEST OBJECT", font=("Consolas", 10), text_color=_TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 2))
        self._lbl_closest = ctk.CTkLabel(sec_det, text="—", font=("Consolas", 12), text_color=_TEXT_PRIMARY)
        self._lbl_closest.pack(anchor="w", padx=12, pady=(0, 8))

    # ------------ driving controls strip ---------------
    def _build_driving_controls(self):
        strip = ctk.CTkFrame(self, fg_color=_BG_PANEL, height=56, corner_radius=0)
        strip.pack(fill="x", padx=8, pady=(0, 4))
        strip.pack_propagate(False)

        inner = ctk.CTkFrame(strip, fg_color=_BG_PANEL)
        inner.pack(fill="both", expand=True, padx=8, pady=6)

        # cruise speed section
        ctk.CTkLabel(inner, text="CRUISE", font=("Consolas", 9, "bold"), text_color=_TEXT_DIM,).pack(side="left", padx=(4, 8))

        self._lbl_target = ctk.CTkLabel(inner, text="15 mph", font=("Consolas", 13, "bold"), text_color=_ACCENT_CYAN, width=70)
        self._lbl_target.pack(side="left", padx=(0, 8))

        for spd in [10, 15, 20, 25, 30, 40, 50, 60, 70, 80]:
            btn = ctk.CTkButton(
                inner, text=f"{spd}", width=36, height=28,
                fg_color=_BG_INPUT, hover_color="#333340",
                text_color=_TEXT_PRIMARY, font=("Consolas", 11),
                command=lambda s=spd: self._set_target_speed(s),
            )
            btn.pack(side="left", padx=2)

        # separator
        sep = ctk.CTkFrame(inner, fg_color=_BORDER_SUBTLE, width=2)
        sep.pack(side="left", fill="y", padx=12, pady=4)

        # spawn section
        ctk.CTkLabel(inner, text="SPAWN",font=("Consolas", 9, "bold"), text_color=_TEXT_DIM).pack(side="left", padx=(0, 8))

        self._btn_spawn_ped = ctk.CTkButton(
            inner, text="🚶 Pedestrian", width=110, height=28,
            fg_color=_STATUS_AMBER, hover_color="#D97706",
            text_color="#000", font=("Consolas", 11, "bold"),
            command=self._spawn_pedestrian,
        )
        self._btn_spawn_ped.pack(side="left", padx=4)

        self._btn_spawn_veh = ctk.CTkButton(
            inner, text="🚗 Vehicle", width=100, height=28,
            fg_color=_ACCENT_BLUE, hover_color="#2563EB",
            text_color="#FFF", font=("Consolas", 11, "bold"),
            command=self._spawn_vehicle,
        )
        self._btn_spawn_veh.pack(side="left", padx=4)

        self._btn_clear = ctk.CTkButton(
            inner, text="✕ Clear", width=80, height=28,
            fg_color=_BG_INPUT, hover_color="#333340",
            text_color=_STATUS_RED, font=("Consolas", 11, "bold"),
            command=self._clear_spawned,
        )
        self._btn_clear.pack(side="left", padx=4)

    # ------------ engineer bar + drawer ------------
    def _build_engineer_bar(self):

        # toggle bar
        bar = ctk.CTkFrame(self, fg_color=_BG_PANEL, height=38, corner_radius=0)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._btn_engineer = ctk.CTkButton(
            bar, text="⚙  Engineer Mode", width=160, height=28,
            fg_color=_BG_INPUT, hover_color="#333340",
            text_color=_TEXT_DIM, font=("Consolas", 11),
            command=self._toggle_engineer,
        )
        self._btn_engineer.pack(side="left", padx=10, pady=5)

        self._lbl_eng_status = ctk.CTkLabel(
            bar, text="", font=("Consolas", 10), text_color=_TEXT_DIM,
        )
        self._lbl_eng_status.pack(side="left", padx=4)

        # drawer (hidden by default)
        self._eng_drawer = ctk.CTkFrame(self, fg_color=_BG_PANEL,
                                        height=0, corner_radius=0)
        self._eng_drawer.pack(fill="x", side="bottom", before=bar)
        self._eng_drawer.pack_forget()
        self._eng_visible = False

        self._build_engineer_controls(self._eng_drawer)

    def _build_engineer_controls(self, parent):
        inner = ctk.CTkFrame(parent, fg_color=_BG_PANEL)
        inner.pack(fill="x", padx=16, pady=10)

        # row of controls
        row = ctk.CTkFrame(inner, fg_color=_BG_PANEL)
        row.pack(fill="x")

        # source selector 
        grp1 = ctk.CTkFrame(row, fg_color=_BG_CARD, corner_radius=8)
        grp1.pack(side="left", padx=(0, 12), pady=4, fill="y")

        ctk.CTkLabel(grp1, text="INPUT SOURCE", font=("Consolas", 9), text_color=_TEXT_DIM).pack(padx=12, pady=(8, 4))

        self._src_var = ctk.StringVar(value="carla")
        src_frame = ctk.CTkFrame(grp1, fg_color=_BG_CARD)
        src_frame.pack(padx=12, pady=(0, 10))
        ctk.CTkRadioButton(
            src_frame, text="CARLA", variable=self._src_var, value="carla",
            font=("Consolas", 11), text_color=_TEXT_PRIMARY,
            fg_color=_ACCENT_BLUE, hover_color=_ACCENT_BLUE,
            command=self._on_source_change,
        ).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(
            src_frame, text="Webcam", variable=self._src_var, value="webcam",
            font=("Consolas", 11), text_color=_TEXT_PRIMARY,
            fg_color=_ACCENT_BLUE, hover_color=_ACCENT_BLUE,
            command=self._on_source_change,
        ).pack(anchor="w", pady=2)

        # confidence threshold 
        grp2 = ctk.CTkFrame(row, fg_color=_BG_CARD, corner_radius=8)
        grp2.pack(side="left", padx=(0, 12), pady=4, fill="y")

        ctk.CTkLabel(grp2, text="CONFIDENCE", font=("Consolas", 9), text_color=_TEXT_DIM).pack(padx=12, pady=(8, 4))
        self._lbl_conf = ctk.CTkLabel(grp2, text="0.45",
                                      font=("Consolas", 12, "bold"),
                                      text_color=_ACCENT_CYAN)
        self._lbl_conf.pack(padx=12)

        self._slider_conf = ctk.CTkSlider(
            grp2, from_=0.1, to=0.9, number_of_steps=80, width=160,
            fg_color=_BG_INPUT, progress_color=_ACCENT_BLUE,
            button_color=_ACCENT_CYAN, button_hover_color="#67E8F9",
            command=self._on_conf_change,
        )
        self._slider_conf.set(0.45)
        self._slider_conf.pack(padx=12, pady=(4, 12))

        # rain intensity 
        grp3 = ctk.CTkFrame(row, fg_color=_BG_CARD, corner_radius=8)
        grp3.pack(side="left", padx=(0, 12), pady=4, fill="y")

        ctk.CTkLabel(grp3, text="RAIN INTENSITY", font=("Consolas", 9), text_color=_TEXT_DIM).pack(padx=12, pady=(8, 4))
        self._lbl_rain = ctk.CTkLabel(grp3, text="0 %",
                                      font=("Consolas", 12, "bold"),
                                      text_color=_ACCENT_CYAN)
        self._lbl_rain.pack(padx=12)

        self._slider_rain = ctk.CTkSlider(
            grp3, from_=0, to=100, number_of_steps=100, width=160,
            fg_color=_BG_INPUT, progress_color=_ACCENT_BLUE,
            button_color=_ACCENT_CYAN, button_hover_color="#67E8F9",
            command=self._on_rain_change,
        )
        self._slider_rain.set(0)
        self._slider_rain.pack(padx=12, pady=(4, 12))

        # fog intensity
        grp3b = ctk.CTkFrame(row, fg_color=_BG_CARD, corner_radius=8)
        grp3b.pack(side="left", padx=(0, 12), pady=4, fill="y")

        ctk.CTkLabel(grp3b, text="FOG INTENSITY", font=("Consolas", 9), text_color=_TEXT_DIM).pack(padx=12, pady=(8, 4))
        self._lbl_fog = ctk.CTkLabel(grp3b, text="0 %",
                                     font=("Consolas", 12, "bold"),
                                     text_color=_ACCENT_CYAN)
        self._lbl_fog.pack(padx=12)

        self._slider_fog = ctk.CTkSlider(
            grp3b, from_=0, to=100, number_of_steps=100, width=160,
            fg_color=_BG_INPUT, progress_color=_ACCENT_BLUE,
            button_color=_ACCENT_CYAN, button_hover_color="#67E8F9",
            command=self._on_fog_change,
        )
        self._slider_fog.set(0)
        self._slider_fog.pack(padx=12, pady=(4, 12))

        # AEB toggle
        grp4 = ctk.CTkFrame(row, fg_color=_BG_CARD, corner_radius=8)
        grp4.pack(side="left", padx=(0, 12), pady=4, fill="y")

        ctk.CTkLabel(grp4, text="AEB ENABLED", font=("Consolas", 9), text_color=_TEXT_DIM).pack(padx=12, pady=(8, 4))

        self._aeb_switch = ctk.CTkSwitch(
            grp4, text="", width=48,
            fg_color=_BG_INPUT, progress_color=_STATUS_GREEN,
            button_color=_TEXT_PRIMARY, button_hover_color="#D4D4D8",
            command=self._on_aeb_toggle,
        )
        self._aeb_switch.select() 
        self._aeb_switch.pack(padx=12, pady=(4, 12))

        # map selector
        grp5 = ctk.CTkFrame(row, fg_color=_BG_CARD, corner_radius=8)
        grp5.pack(side="left", padx=(0, 12), pady=4, fill="y")

        ctk.CTkLabel(grp5, text="CARLA MAP",font=("Consolas", 9), text_color=_TEXT_DIM).pack(padx=12, pady=(8, 4))
        self._map_var = ctk.StringVar(value="Town04")

        self._map_dropdown = ctk.CTkOptionMenu(
            grp5, variable=self._map_var,
            values=["Town04", "Town10HD_Opt"],
            width=140, height=28,
            fg_color=_BG_INPUT, button_color=_ACCENT_BLUE,
            button_hover_color="#2563EB",
            text_color=_TEXT_PRIMARY, font=("Consolas", 11),
            dropdown_fg_color=_BG_CARD,
            dropdown_text_color=_TEXT_PRIMARY,
            dropdown_hover_color=_ACCENT_BLUE,
            command=self._on_map_change,
        )
        self._map_dropdown.pack(padx=12, pady=(4, 12))

    # ---------- engineer mode gating -------------
    def _toggle_engineer(self):
        if self._eng_visible:
            self._eng_drawer.pack_forget()
            self._eng_visible = False
            self._lbl_eng_status.configure(text="")
            return

        if not self._engineer_unlocked:
            self._prompt_password()
            return

        self._show_engineer()

    def _prompt_password(self):
        dialog = ctk.CTkInputDialog(
            title="Engineer Mode",
            text="Enter engineer password:",
        )
        entered = dialog.get_input()
        entered_hash = hashlib.sha256(entered.encode()).hexdigest()
        if entered_hash == _ENGINEER_PASSWORD_HASH:
            self._engineer_unlocked = True
            self._show_engineer()
        else:
            self._lbl_eng_status.configure(
                text="  Access denied", text_color=_STATUS_RED,
            )

    def _show_engineer(self):
        self._eng_drawer.pack(fill="x", side="bottom",
                              before=self._btn_engineer.master)
        self._eng_visible = True
        self._lbl_eng_status.configure(
            text="  ✓ Unlocked", text_color=_STATUS_GREEN,
        )

    # --------------- callbacks from engineer controls ------------------
    def _on_source_change(self):
        new_src = self._src_var.get()
        if new_src == self._source:
            return

        was_running = self._running

        def _switch():
            if was_running:
                self._running = False
                if self._worker_thread and self._worker_thread.is_alive():
                    self._worker_thread.join(timeout=2.0)
                if self._adapter:
                    try:
                        self._adapter.stop()
                    except Exception:
                        pass
                    self._adapter = None

            self._source = new_src

            if was_running:
                self.after(0, self._start_system)

        threading.Thread(target=_switch, daemon=True).start()

    def _on_conf_change(self, val):
        val = round(val, 2)
        self._lbl_conf.configure(text=f"{val:.2f}")
        self._detector.set_thresholds(conf=val, iou=self._detector.iou_thresh)

    def _on_rain_change(self, val):
        val = int(val)
        self._lbl_rain.configure(text=f"{val} %")
        if self._adapter:
            try:
                self._adapter.set_rain(val)
            except Exception:
                pass

    def _on_fog_change(self, val):
        val = int(val)
        self._lbl_fog.configure(text=f"{val} %")
        if self._adapter:
            try:
                self._adapter.set_fog(val)
            except Exception:
                pass

    def _on_map_change(self, map_name: str):
        if map_name == self._map_name:
            return

        was_running = self._running

        def _switch():
            if was_running:
                self._running = False
                if self._worker_thread and self._worker_thread.is_alive():
                    self._worker_thread.join(timeout=2.0)
                if self._adapter:
                    try:
                        self._adapter.stop()
                    except Exception:
                        pass
                    self._adapter = None

            self._map_name = map_name

            if was_running:
                self.after(0, self._start_system)

        threading.Thread(target=_switch, daemon=True).start()

    def _on_aeb_toggle(self):
        enabled = bool(self._aeb_switch.get())
        self._controller.set_enabled(enabled)

    # --------------- driving control callbacks ------------------
    def _set_target_speed(self, mph: float):
        self._target_speed_mph = float(mph)
        self._lbl_target.configure(text=f"{int(mph)} mph")

    def _spawn_pedestrian(self):
        if not self._adapter:
            return
        try:
            self._adapter.spawn_pedestrian_ahead()
        except Exception as e:
            print(f"Spawn pedestrian failed: {e}")

    def _spawn_vehicle(self):
        if not self._adapter:
            return
        try:
            self._adapter.spawn_static_vehicle_ahead()
        except Exception as e:
            print(f"Spawn vehicle failed: {e}")

    def _clear_spawned(self):
        if not self._adapter:
            return
        try:
            self._adapter.clear_spawned_objects()
        except Exception as e:
            print(f"Clear spawned failed: {e}")

    # ---------- start / stop system -----------
    def _toggle_system(self):
        if self._running:
            self._stop_system()
        else:
            self._start_system()

    def _start_system(self):
        if self._running:
            return

        self._btn_start.configure(text="…", state="disabled")

        def _connect():
            try:
                if self._source == "carla":
                    adapter = CarlaAdapter(autopilot=False, map_name=self._map_name)
                else:
                    adapter = WebcamAdapter()
                adapter.start()
            except Exception as e:
                self.after(0, lambda: self._show_error(f"Failed to start: {e}"))
                self.after(0, lambda: self._btn_start.configure(
                    text="▶  START", state="normal",
                    fg_color=_STATUS_GREEN, hover_color="#16A34A"))
                return

            # success — update state and start worker
            self._adapter = adapter
            self._running = True
            self._controller.reset_states()

            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

            # update GUI on main thread
            self.after(0, lambda: self._btn_start.configure(
                text="■  STOP", state="normal",
                fg_color=_STATUS_RED, hover_color="#DC2626"))

        threading.Thread(target=_connect, daemon=True).start()

    def _stop_system(self):
        self._running = False

        self._btn_start.configure(text="…", state="disabled")

        def _disconnect():
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)

            if self._adapter:
                try:
                    self._adapter.stop()
                except Exception:
                    pass
                self._adapter = None

            # update GUI on main thread
            self.after(0, lambda: self._btn_start.configure(
                text="▶  START", state="normal",
                fg_color=_STATUS_GREEN, hover_color="#16A34A"))

        threading.Thread(target=_disconnect, daemon=True).start()

    # --------- worker thread ----------
    def _worker_loop(self):
        fps_counter = _FPSCounter()
        frame_idx = 0
        last_detections: List[Detection] = []
        detect_every_n = 2

        while self._running:
            t0 = time.perf_counter()

            # 1. grab frame 
            frame = self._grab_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            frame_idx += 1

            # 2. detect 
            if frame_idx % detect_every_n == 0:
                detections = self._detector.predict(frame)
                last_detections = detections
            else:
                detections = last_detections

            # 3. vehicle state 
            v_state: Optional[VehicleState] = None
            speed_mph = 0.0
            if self._adapter:
                v_state = self._adapter.get_state()
                if v_state:
                    speed_mph = v_state.speed_mph

            # 4. AEB decision
            decision = self._controller.update(detections, speed_mph)

            # 5. drive (adapter handles cruise + AEB)
            if self._adapter:
                aeb_brake = decision.brake if decision.state == AEBState.BRAKING else 0.0
                try:
                    self._adapter.drive(aeb_brake=aeb_brake)
                except Exception:
                    pass

            # 6. draw overlays on frame
            display = self._detector.draw_corridor(frame)
            display = self._detector.draw_detections(display, detections)

            # 7. timing
            latency = (time.perf_counter() - t0) * 1000
            fps_counter.tick()

            # 8. push to shared state
            with self._lock:
                self._display_frame = display
                self._decision = decision
                self._vehicle_state = v_state
                self._fps = fps_counter.fps
                self._latency_ms = latency
                self._detections = detections

    def _grab_frame(self) -> Optional[np.ndarray]:
        if self._adapter:
            return self._adapter.get_frame()
        return None

    # ----------- GUI tick (runs on main thread) ------------
    def _gui_tick(self):
        with self._lock:
            frame = self._display_frame
            decision = self._decision
            v_state = self._vehicle_state
            fps = self._fps
            lat = self._latency_ms
            dets = self._detections

        # frame
        if frame is not None:
            self._render_frame(frame)

        # status
        self._status_dot.set_state(decision.state)
        self._lbl_reason.configure(text=decision.reason)

        # speed + throttle
        if v_state:
            self._lbl_speed.configure(text=f"{v_state.speed_mph:.0f}")
            self._gauge_throttle.set_value(v_state.throttle)
        else:
            self._lbl_speed.configure(text="—")
            self._gauge_throttle.set_value(0.0)

        # brake 
        self._gauge_brake.set_value(decision.brake)

        # proximity
        prox = decision.closest_closeness if decision.closest_closeness else 0.0
        self._prox_bar.set_value(prox)

        # closest object
        if decision.closest_class:
            self._lbl_closest.configure(text=f"{decision.closest_class}  " f"conf={decision.closest_conf:.2f}  " f"close={decision.closest_closeness:.2f}",
            )
        else:
            self._lbl_closest.configure(text="—")

        # FPS / latency
        self._lbl_fps.configure(
            text=f"FPS  {fps:.0f}    Latency  {lat:.0f} ms",
        )

        self.after(_GUI_REFRESH_MS, self._gui_tick)

    # ----------- frame rendering -------------
    def _render_frame(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        lw = self._video_label.winfo_width()
        lh = self._video_label.winfo_height()
        if lw > 1 and lh > 1:
            pil.thumbnail((lw, lh), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=pil)
        self._video_label.configure(image=imgtk, text="")
        self._video_label._imgtk = imgtk  

    def _set_placeholder_frame(self):
        self._video_label.configure(
            text="NO  FEED\n\nPress  ▶ START  to begin",
            font=("Consolas", 16), text_color=_TEXT_DIM,
        )

    # --------- helpers -----------
    def _show_error(self, msg: str):
        dialog = ctk.CTkInputDialog(title="Error", text=msg)
        dialog.get_input()  

    def _on_close(self):
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        if self._adapter:
            try:
                self._adapter.stop()
            except Exception:
                pass
            self._adapter = None

        self.destroy()


# --------- Tiny FPS helper ---------
class _FPSCounter:
    def __init__(self, window: int = 30):
        self._times: list[float] = []
        self._window = window
        self.fps = 0.0

    def tick(self):
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times = self._times[-self._window:]
        if len(self._times) >= 2:
            dt = self._times[-1] - self._times[0]
            self.fps = (len(self._times) - 1) / max(1e-9, dt)

# -------- Entry point --------
if __name__ == "__main__":
    app = AEBApp()
    app.mainloop()