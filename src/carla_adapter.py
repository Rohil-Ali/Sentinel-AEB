from __future__ import annotations

import time
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

import carla

# Dataclass to hold vehicle state information
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
    ):
        
        self.host = host
        self.port = port
        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov
        self.vehicle_filter = vehicle_filter
        self.autopilot = autopilot

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None
        self._img_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0

        self._running = False

        self.spawned_objects: list[carla.Actor] = []  
        

    # ---------- lifecycle ----------
    def start(self):
        if self._running: 
            return
        
        blueprint_library = self._connect() # connect to CARLA server
        self._spawn_vehicle(blueprint_library) # Spawn vehicle
        self._spawn_camera(blueprint_library) # spawn camera

        # start listening to camera
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
        self._safe_destroy(self.camera)
        self._safe_destroy(self.vehicle)

        self.world = None
        self.client = None



    # ---------- frame + telemetry ----------
    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        # Returns latest BGR frame as np.ndarray, or None if no new frame arroh ived.
        if not self._running:
            return None

        try:
            image = self._img_queue.get(timeout=timeout)
            frame = self._carla_image_to_bgr(image)
            self._last_frame = frame
            self._last_frame_ts = time.time()
            return frame
        except queue.Empty:
            # Return last frame if we have one
            return self._last_frame

    def get_state(self) -> Optional[VehicleState]:
        # Return current vehicle telemetry and last applied control.
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
    
    # ---------- control ----------
    def apply_control(self, throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0) -> None:
        if not self.vehicle:
            return

        throttle = float(max(0.0, min(1.0, throttle)))
        brake = float(max(0.0, min(1.0, brake)))
        steer = float(max(-1.0, min(1.0, steer)))

        #  disable autopilot so our brakes work
        if self.vehicle.is_autopilot_enabled:
            self.vehicle.set_autopilot(False)

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

    def brake_full(self) -> None:
        self.apply_control(throttle=0.0, brake=1.0, steer=0.0)


    # ---------- environment ----------

    def set_rain(self, intensity: float) -> None:
      
        if not self.world:
            return

        intensity = float(max(0.0, min(100.0, intensity)))
        weather = self.world.get_weather()
        weather.precipitation = intensity
        weather.precipitation_deposits = intensity  
        self.world.set_weather(weather)

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
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        self.vehicle.set_autopilot(self.autopilot)

    def _connect(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)
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
        candidates = bps.filter("vehicle.*")
        if not candidates:
            raise RuntimeError("No vehicle blueprints found.")

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

    @staticmethod
    def _carla_image_to_bgr(image: carla.Image) -> np.ndarray:
        # convert CARLA image to BGR numpy array
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))  # BGRA
        bgr = arr[:, :, :3]  # drop alpha
        return bgr