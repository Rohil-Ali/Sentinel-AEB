"""
carla_adapter.py – CARLA simulator adapter for the AEB system.

Handles everything to do with CARLA - connecting to the sim, spawning
the car and camera, grabbing frames, reading speed/brake data, and
driving the car with cruise control + lane following + AEB override.

Also has some functions for spawning vehicles and pedestrians ahead
of the car for testing AEB scenarios.

Interface methods:
    start()                      → connect to CARLA, spawn vehicle and camera
    stop()                       → destroy everything and disconnect
    get_frame()                  → grab latest camera frame as BGR
    get_state()                  → get current speed, throttle, brake
    get_collision()              → True if collision occurred since last check
    drive(target_speed_mph, aeb_brake) → cruise at target speed, AEB overrides if active
    apply_control(throttle, brake, steer) → send raw control to the car
    brake_full()                 → full emergency brake
    set_rain(intensity)          → change rain in the sim
    set_fog(intensity)           → change fog in the sim
    spawn_static_vehicle_ahead() → drop a parked car ahead for testing
    spawn_pedestrian_ahead()     → drop a pedestrian ahead for testing
    clear_spawned_objects()      → remove all spawned test objects
"""

from __future__ import annotations

import time
import math
import queue
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

import carla


@dataclass
class VehicleState:
    speed_mps: float
    speed_mph: float
    brake: float
    throttle: float 
   
class CarlaAdapter:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        image_width: int = 1280,
        image_height: int = 720,
        fov: int = 90,
        vehicle_filter: str = "vehicle.tesla.model3",
        autopilot: bool = False,
        map_name: Optional[str] = None,
        spawn_index: int = 50,
        backtrack_meters: float = 850.0,
    ):
        
        self.host = host
        self.port = port
        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov
        self.vehicle_filter = vehicle_filter
        self.autopilot = autopilot
        self.map_name = map_name
        self.spawn_index = spawn_index
        self.backtrack_meters = backtrack_meters

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None
        self._collision_sensor: Optional[carla.Sensor] = None
        self._collision_flag: bool = False
        self._img_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0

        self._running = False

        self.spawned_objects: List[carla.Actor] = []  
        

    # ---------- lifecycle ----------
    def start(self):
        if self._running: 
            return
        
        blueprint_library = self._connect() 
        self._spawn_vehicle(blueprint_library)
        self._spawn_camera(blueprint_library)
        self._spawn_collision_sensor(blueprint_library)

        def _on_image(image: carla.Image) -> None:
            try: 
                while self._img_queue.qsize() > 0:
                    _=self._img_queue.get_nowait()
            except queue.Empty:
                pass
            self._img_queue.put(image)   
        self.camera.listen(_on_image)
        self._running = True

    def stop(self):
        self._running = False
        
        self.clear_spawned_objects()
        self._safe_destroy(self._collision_sensor)
        self._safe_destroy(self.camera)
        self._safe_destroy(self.vehicle)

        self._collision_sensor = None
        self._collision_flag = False
        self.camera = None
        self.vehicle = None 
        self.world = None
        self.client = None


    # ---------- frame + telemetry ----------
    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        if not self._running:
            return None

        try:
            image = self._img_queue.get(timeout=timeout)
            frame = self._carla_image_to_bgr(image)
            self._last_frame = frame
            self._last_frame_ts = time.time()
            return frame
        except queue.Empty:
            return self._last_frame

    def get_state(self) -> Optional[VehicleState]:
        if not self.vehicle:
            return None

        vel = self.vehicle.get_velocity()
        speed_mps = float((vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5)
        speed_mph = speed_mps * 2.23693629

        ctrl = self.vehicle.get_control()
        return VehicleState(
            speed_mps=speed_mps,
            speed_mph=speed_mph,
            throttle=float(ctrl.throttle),
            brake=float(ctrl.brake),
        )

    def get_collision(self) -> bool:
        if self._collision_flag:
            self._collision_flag = False
            return True
        return False
    
    
    # ---------- control ----------
    def apply_control(self, throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0) -> None:
        if not self.vehicle:
            return

        throttle = float(max(0.0, min(1.0, throttle)))
        brake = float(max(0.0, min(1.0, brake)))
        steer = float(max(-1.0, min(1.0, steer)))

        self.vehicle.set_autopilot(False)

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

    def brake_full(self) -> None:
        self.apply_control(throttle=0.0, brake=1.0, steer=0.0)

    def drive(self, target_speed_mph: float, aeb_brake: float = 0.0) -> None:
        """
        Drives the car at the target speed while following the lane.
        If AEB is triggered, it overrides the cruise control and brakes.
        """
        if not self.vehicle or not self.world:
            return

        state = self.get_state()
        if state is None:
            return

        steer = self._compute_lane_follow_steer()
        cruise_throttle, cruise_brake = self._compute_cruise_control(state.speed_mph, target_speed_mph)

        # AEB override
        if aeb_brake > 0.0:
            self.apply_control(throttle=0.0, brake=aeb_brake, steer=0.0)
        else:
            self.apply_control(throttle=cruise_throttle, brake=cruise_brake, steer=steer)

    
    # ---------- environment ----------
    def set_rain(self, intensity: float) -> None:
      
        if not self.world:
            return

        intensity = float(max(0.0, min(100.0, intensity)))
        weather = self.world.get_weather()
        weather.precipitation = intensity
        weather.precipitation_deposits = intensity  
        self.world.set_weather(weather)

    def set_fog(self, intensity: float) -> None:
        if not self.world:
            return

        intensity = float(max(0.0, min(100.0, intensity)))
        weather = self.world.get_weather()
        weather.fog_density = intensity
        weather.fog_distance = max(0.0, 100.0 - intensity)
        self.world.set_weather(weather)

    def set_time(self, preset: str) -> None:
            if not self.world:
                return
    
            angles = {
                "Day": 60.0,
                "Dusk": 5.0,
                "Night": -80.0,
            }
    
            angle = angles.get(preset, 60.0)
            weather = self.world.get_weather()
            weather.sun_altitude_angle = angle
            self.world.set_weather(weather)

    # ---------- driving helpers (internal) ----------
    def _compute_lane_follow_steer(self, lookahead: float = 8.0) -> float:
        """
        Steers the car to stay in the center of the lane using waypoints ahead.
        """
        if not self.vehicle or not self.world:
            return 0.0

        world_map = self.world.get_map()
        vehicle_transform = self.vehicle.get_transform()
        wp = world_map.get_waypoint(
            vehicle_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        # adjust lookahead based on speed (look further at higher speeds)
        vel = self.vehicle.get_velocity()
        speed_mps = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        dynamic_lookahead = max(lookahead, speed_mps * 0.8)

        nxt = wp.next(dynamic_lookahead)
        if not nxt:
            return 0.0
        target_wp = nxt[0]

        # vector from vehicle to target waypoint
        dx = target_wp.transform.location.x - vehicle_transform.location.x
        dy = target_wp.transform.location.y - vehicle_transform.location.y

        # vehicle forward direction
        yaw_rad = math.radians(vehicle_transform.rotation.yaw)
        fwd_x = math.cos(yaw_rad)
        fwd_y = math.sin(yaw_rad)

        # cross product to tell where to steer (- for left, + for right)
        cross = fwd_x * dy - fwd_y * dx
        # dot product for distance
        dot = fwd_x * dx + fwd_y * dy
        if dot < 0.1:
            dot = 0.1

        steer = math.atan2(cross, dot)
        steer *= 1.8

        return float(max(-1.0, min(1.0, steer)))

    @staticmethod
    def _compute_cruise_control(current_mph: float,target_mph: float) -> tuple:
        error = target_mph - current_mph

        kp_throttle = 0.25
        kp_brake = 0.05

        throttle = max(0.0, min(1.0, kp_throttle * error))
        brake = max(0.0, min(1.0, kp_brake * (-error)))

        if abs(error) < 0.5:
            throttle *= 0.2
            brake *= 0.2

        return throttle, brake

    
    # ---------- helper functions ----------
    def _spawn_camera(self, blueprint_library):
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.image_width))
        cam_bp.set_attribute("image_size_y", str(self.image_height))
        cam_bp.set_attribute("fov", str(self.fov))

        cam_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)

    def _spawn_vehicle(self, blueprint_library):
        vehicle_bps = blueprint_library.filter(self.vehicle_filter)
        if not vehicle_bps:
            raise RuntimeError(f"No vehicle blueprint found for filter: {self.vehicle_filter}")
        vehicle_bp = vehicle_bps[0]

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found on this map.")

        spawn_point = spawn_points[self.spawn_index % len(spawn_points)]
        wp = self.world.get_map().get_waypoint(spawn_point.location)
        prev_wps = wp.previous(self.backtrack_meters)
        if prev_wps:
            spawn_point = prev_wps[0].transform
            spawn_point.location.z += 0.5 

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(self.autopilot)

    # collision sensor for testing
    def _spawn_collision_sensor(self, blueprint_library):
        """Attach a collision sensor to the vehicle to detect impacts."""
        col_bp = blueprint_library.find("sensor.other.collision")
        self._collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle
        )

        def _on_collision(event):
            self._collision_flag = True

        self._collision_sensor.listen(_on_collision)

    def _connect(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)

        if self.map_name:
            current_map = self.client.get_world().get_map().name
            if self.map_name not in current_map:
                print(f"Loading map: {self.map_name}")
                self.client.load_world(self.map_name)
                time.sleep(2.0)
        self.world = self.client.get_world()
        blueprint_library = self.world.get_blueprint_library()
        return blueprint_library
    
    def _safe_destroy(self, actor):
        if actor is None: 
            return
        
        try: 
            if hasattr(actor, "stop"):
                actor.stop()
        except Exception:
            pass

        try:
            actor.destroy()
        except Exception:
            pass

        actor = None

    @staticmethod
    def _carla_image_to_bgr(image: carla.Image) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))  
        bgr = arr[:, :, :3]  
        return bgr
   
   
    # ---------- environment test functions ----------
    def get_spawn_distance_ahead(self, min_distance: float = 15.0, time_headway_s: float = 4.0) -> float:
        if not self.vehicle:
            return float(min_distance)

        vel = self.vehicle.get_velocity()
        speed_mps = float((vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5)
        return float(max(min_distance, speed_mps * time_headway_s))

    def waypoint_ahead(self, meters: float) -> carla.Waypoint:
        if not self.world or not self.vehicle:
            raise RuntimeError("CARLA world or ego vehicle is not available.")

        world_map = self.world.get_map()
        wp = world_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        nxt = wp.next(meters)
        if not nxt:
            return wp
        return nxt[0]

    def spawn_static_vehicle_ahead(self, meters: Optional[float] = None) -> carla.Vehicle:
        if not self.world or not self.vehicle:
            raise RuntimeError("CARLA world or ego vehicle is not available.")

        if meters is None:
            meters = self.get_spawn_distance_ahead()

        bps = self.world.get_blueprint_library()
        candidates = bps.filter("vehicle.dodge.charger_2020")
        if not candidates:
            candidates = bps.filter("vehicle.*")
        bp = candidates[0]

        wp = self.waypoint_ahead(meters)
        transform = wp.transform
        transform.location.z += 0.2

        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            raise RuntimeError(f"Failed to spawn vehicle at {meters:.1f}m ahead.")

        actor.set_autopilot(False)
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

        self.spawned_objects.append(actor)
        return actor

    def spawn_pedestrian_ahead(self, meters: Optional[float] = None) -> carla.Actor:
        if not self.world or not self.vehicle:
            raise RuntimeError("CARLA world or ego vehicle is not available.")

        if meters is None:
            meters = self.get_spawn_distance_ahead()

        bps = self.world.get_blueprint_library()
        walkers = bps.filter("walker.pedestrian.*")
        if not walkers:
            raise RuntimeError("No pedestrian blueprints found.")

        bp = walkers[0]

        wp = self.waypoint_ahead(meters)
        transform = wp.transform
        transform.location.z += 0.2

        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            raise RuntimeError(f"Failed to spawn pedestrian at {meters:.1f}m ahead.")

        self.spawned_objects.append(actor)
        return actor

    def clear_spawned_objects(self) -> None:
        for actor in self.spawned_objects:
            self._safe_destroy(actor)
        self.spawned_objects.clear()