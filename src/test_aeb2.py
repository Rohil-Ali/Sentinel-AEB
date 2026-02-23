# carla_aeb_controller_test.py
from __future__ import annotations

import time
import math
import queue
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import pygame
import carla

from detector import YOLODetector
from aeb_controller import AEBController


# ---------------- utils ----------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def carla_image_to_bgr(image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))  # BGRA
    return arr[:, :, :3]  # BGR


def get_speed_mps(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))


def mps_to_mph(mps: float) -> float:
    return mps * 2.23693629


def mph_to_mps(mph: float) -> float:
    return mph / 2.23693629


def set_synchronous(world: carla.World, enable: bool, fixed_dt: float = 0.05) -> None:
    settings = world.get_settings()
    settings.synchronous_mode = enable
    settings.fixed_delta_seconds = fixed_dt if enable else None
    world.apply_settings(settings)


def load_open_map(client: carla.Client, preferred: str = "Town04") -> carla.World:
    """Load a map by name. Returns current world if map not found."""
    try:
        print(f"Attempting to load map: {preferred}")
        # Try without MapLayer first (simpler)
        world = client.load_world(preferred)
        print(f"✓ Successfully loaded {preferred}")
        return world
    except Exception as e:
        print(f"✗ Failed to load {preferred}: {e}")
        print("Available maps:")
        try:
            maps = client.get_available_maps()
            for m in maps:
                print(f"  - {m}")
        except Exception:
            pass
        print("Using current world instead")
        return client.get_world()


def overlay_lines(frame: np.ndarray, lines: List[str], x: int = 10, y: int = 25) -> np.ndarray:
    out = frame.copy()
    for line in lines:
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 24
    return out


# --------------- spawning ----------------

def spawn_ego(world: carla.World, spawn_index: int = 50, backtrack_meters: float = 60.0) -> carla.Vehicle:
    """Spawn ego vehicle at specific spawn point."""
    bps = world.get_blueprint_library()
    bp = bps.filter("vehicle.tesla.model3")[0]

    spawns = world.get_map().get_spawn_points()
    spawn_point = spawns[spawn_index % len(spawns)]
    if not spawns:
        raise RuntimeError("No spawn points found on this map.")
    
    wp = world.get_map().get_waypoint(spawn_point.location)
    prev_wps = wp.previous(backtrack_meters)
    if prev_wps:
        spawn_point = prev_wps[0].transform
        spawn_point.location.z += 0.5  # Lift slightly to prevent clipping into the road

    ego = world.spawn_actor(bp, spawn_point)
    ego.set_autopilot(False)
    print(f"Spawned {backtrack_meters}m behind point {spawn_index}")
    return ego


def spawn_rgb_camera(
    world: carla.World,
    attach_to: carla.Actor,
    width: int = 960,
    height: int = 540,
    fov: int = 90,
) -> Tuple[carla.Sensor, "queue.Queue[carla.Image]"]:
    bps = world.get_blueprint_library()
    cam_bp = bps.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))

    cam_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
    cam = world.spawn_actor(cam_bp, cam_transform, attach_to=attach_to)

    q: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)

    def _on_image(img: carla.Image) -> None:
        try:
            while q.qsize() > 0:
                _ = q.get_nowait()
        except queue.Empty:
            pass
        q.put(img)

    cam.listen(_on_image)
    return cam, q


def waypoint_ahead(world: carla.World, ego: carla.Vehicle, meters: float) -> carla.Waypoint:
    m = world.get_map()
    wp = m.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    nxt = wp.next(meters)
    if not nxt:
        return wp
    return nxt[0]


def spawn_static_vehicle_ahead(world: carla.World, ego: carla.Vehicle, meters: float = 25.0) -> carla.Vehicle:
    bps = world.get_blueprint_library()
    candidates = bps.filter("vehicle.*")
    bp = candidates[0]

    wp = waypoint_ahead(world, ego, meters)
    transform = wp.transform
    transform.location.z += 0.2

    v = world.try_spawn_actor(bp, transform)
    if v is None:
        raise RuntimeError("Failed to spawn vehicle (try different distance).")
    v.set_autopilot(False)
    try:
        v.set_simulate_physics(False)
    except Exception:
        pass
    return v


def spawn_static_pedestrian_ahead(world: carla.World, ego: carla.Vehicle, meters: float = 20.0) -> carla.Actor:
    bps = world.get_blueprint_library()
    walkers = bps.filter("walker.pedestrian.*")
    if not walkers:
        raise RuntimeError("No pedestrian blueprints found.")
    bp = walkers[0]

    wp = waypoint_ahead(world, ego, meters)
    transform = wp.transform
    transform.location.z += 0.2

    ped = world.try_spawn_actor(bp, transform)
    if ped is None:
        raise RuntimeError("Failed to spawn pedestrian (try different distance).")
    return ped


# --------------- control ----------------

@dataclass
class DriverControl:
    steer: float = 0.0
    hand_brake: bool = False
    reverse: bool = False

    # manual inputs
    manual_throttle: float = 0.0
    manual_brake: float = 0.0

    # cruise
    cruise_enabled: bool = True
    target_speed_mph: float = 15.0


def init_pygame() -> None:
    pygame.init()
    pygame.display.set_mode((420, 120))
    pygame.display.set_caption("AEB Controller CARLA Test")


def update_driver_inputs(ctrl: DriverControl) -> DriverControl:
    keys = pygame.key.get_pressed()

    # Steering (A/D)
    steer_target = 0.0
    if keys[pygame.K_a]:
        steer_target = -1.0
    elif keys[pygame.K_d]:
        steer_target = 1.0
    ctrl.steer = ctrl.steer + (steer_target - ctrl.steer) * 0.25
    ctrl.steer = clamp(ctrl.steer, -1.0, 1.0)

    # Hand brake
    ctrl.hand_brake = bool(keys[pygame.K_SPACE])

    # Manual throttle/brake if cruise disabled
    if not ctrl.cruise_enabled:
        if keys[pygame.K_w]:
            ctrl.manual_throttle = clamp(ctrl.manual_throttle + 0.03, 0.0, 1.0)
        else:
            ctrl.manual_throttle = clamp(ctrl.manual_throttle - 0.02, 0.0, 1.0)

        if keys[pygame.K_s]:
            ctrl.manual_brake = clamp(ctrl.manual_brake + 0.06, 0.0, 1.0)
            ctrl.manual_throttle = 0.0
        else:
            ctrl.manual_brake = clamp(ctrl.manual_brake - 0.06, 0.0, 1.0)

    return ctrl


def cruise_throttle_brake(current_mph: float, target_mph: float) -> Tuple[float, float]:
    """
    Simple proportional cruise controller.
    Good enough for repeatable testing.
    """
    error = target_mph - current_mph

    # Throttle when below target, brake when above target
    kp_throttle = 0.08
    kp_brake = 0.05

    throttle = clamp(kp_throttle * error, 0.0, 1.0)
    brake = clamp(kp_brake * (-error), 0.0, 1.0)

    # Small deadband so it doesn't jitter
    if abs(error) < 0.5:
        throttle *= 0.2
        brake *= 0.2

    return throttle, brake


# --------------- main ----------------

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(70.0)

    world = load_open_map(client, preferred="Town04")
    set_synchronous(world, enable=True, fixed_dt=0.05)

    actors: List[carla.Actor] = []
    spawned_objects: List[carla.Actor] = []
    
    # Keep track of spawn index
    current_spawn_index = 50

    try:
        ego = spawn_ego(world, spawn_index=current_spawn_index)
        actors.append(ego)

        cam, cam_q = spawn_rgb_camera(world, ego, width=960, height=540, fov=90)
        actors.append(cam)

        # Defaults as requested
        detector = YOLODetector()     # default params
        controller = AEBController()  # default params

        # FPS tracking
        frame_count = 0
        fps_timer_start = time.time()
        current_fps = 0.0

        init_pygame()
        ctrl = DriverControl(cruise_enabled=True, target_speed_mph=15.0)

        print("\nControls:")
        print("  A/D steer, SPACE handbrake, ESC quit")
        print("  C toggle cruise control on/off")
        print("  R respawn vehicle at spawn point & clear objects")  # NEW
        print("  1 set target speed 15 mph")
        print("  2 set target speed 30 mph")
        print("  3 set target speed 40 mph")
        print("  4-7 set target speed 50-80 mph")
        print("  P spawn pedestrian ahead")
        print("  V spawn static vehicle ahead")
        print("  X clear spawned objects")
        print("  If cruise is OFF: W/S throttle/brake manually\n")

        # AEB hold so braking is visible
        hold_until = 0.0
        hold_brake = 0.0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return

                    # === NEW: RESPAWN KEY ===
                    if event.key == pygame.K_r:
                        print("Respawning vehicle and clearing objects...")
                        
                        # Clear all spawned test objects
                        for a in spawned_objects:
                            try:
                                a.destroy()
                            except Exception:
                                pass
                        spawned_objects.clear()
                        
                        # Destroy old camera
                        try:
                            cam.stop()
                            cam.destroy()
                            actors.remove(cam)
                        except Exception as e:
                            print(f"Error destroying camera: {e}")
                        
                        # Destroy old ego
                        try:
                            ego.destroy()
                            actors.remove(ego)
                        except Exception as e:
                            print(f"Error destroying ego: {e}")
                        
                        # Respawn ego at same spawn point
                        ego = spawn_ego(world, spawn_index=current_spawn_index)
                        actors.append(ego)
                        
                        # Respawn camera attached to new ego
                        cam, cam_q = spawn_rgb_camera(world, ego, width=960, height=540, fov=90)
                        actors.append(cam)
                        
                        # Reset controller state
                        controller.reset_states()
                        
                        # Reset control inputs
                        ctrl = DriverControl(cruise_enabled=True, target_speed_mph=15.0)
                        
                        # Reset AEB hold
                        hold_until = 0.0
                        hold_brake = 0.0
                        
                        print("✓ Respawn complete!")
                        continue  # Skip this frame

                    if event.key == pygame.K_c:
                        ctrl.cruise_enabled = not ctrl.cruise_enabled
                        ctrl.manual_throttle = 0.0
                        ctrl.manual_brake = 0.0
                        print(f"Cruise: {'ON' if ctrl.cruise_enabled else 'OFF'}")

                    if event.key == pygame.K_1:
                        ctrl.target_speed_mph = 15.0
                        print("Target speed set to 15 mph")
                    if event.key == pygame.K_2:
                        ctrl.target_speed_mph = 30.0
                        print("Target speed set to 30 mph")
                    if event.key == pygame.K_3:
                        ctrl.target_speed_mph = 40.0
                        print("Target speed set to 40 mph")
                    if event.key == pygame.K_4:
                        ctrl.target_speed_mph = 50.0
                        print("Target speed set to 50 mph")
                    if event.key == pygame.K_5:
                        ctrl.target_speed_mph = 60.0
                        print("Target speed set to 60 mph")
                    if event.key == pygame.K_6:
                        ctrl.target_speed_mph = 70.0
                        print("Target speed set to 70 mph")
                    if event.key == pygame.K_7:
                        ctrl.target_speed_mph = 80.0
                        print("Target speed set to 80 mph")

                    if event.key == pygame.K_p:
                        current_speed_mps = get_speed_mps(ego)
                        spawn_dist = max(15.0, current_speed_mps * 4.0) 
                        ped = spawn_static_pedestrian_ahead(world, ego, meters=spawn_dist)
                        spawned_objects.append(ped)
                        actors.append(ped)
                        print(f"Spawned pedestrian {spawn_dist:.1f}m ahead")

                    if event.key == pygame.K_v:
                        current_speed_mps = get_speed_mps(ego)
                        spawn_dist = max(15.0, current_speed_mps * 4.0)
                        veh = spawn_static_vehicle_ahead(world, ego, meters=spawn_dist)
                        spawned_objects.append(veh)
                        actors.append(veh)
                        print(f"Spawned static vehicle {spawn_dist:.1f}m ahead")

                    if event.key == pygame.K_x:
                        for a in spawned_objects:
                            try:
                                a.destroy()
                            except Exception:
                                pass
                        spawned_objects.clear()
                        print("Cleared spawned objects")

            # Tick
            world.tick()

            frame_count += 1
            # Get camera frame
            try:
                img = cam_q.get(timeout=1.0)
            except queue.Empty:
                continue
            frame = carla_image_to_bgr(img)

            # Update FPS once per second
            now_time = time.time()
            elapsed = now_time - fps_timer_start
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_timer_start = now_time

            # Update user inputs
            ctrl = update_driver_inputs(ctrl)

            # Speed
            speed_mps = get_speed_mps(ego)
            speed_mph = mps_to_mph(speed_mps)

            # Run detector + controller
            detections = detector.predict(frame)
            decision = controller.update(detections, speed_mph=speed_mph)

            # AEB hold for visibility
            now = time.time()
            if decision.brake > 0.0:
                hold_brake = decision.brake
                hold_until = now + 0.2

            aeb_active = now < hold_until
            aeb_brake = hold_brake if aeb_active else 0.0

            # Compute throttle/brake (cruise or manual)
            if ctrl.cruise_enabled:
                base_throttle, base_brake = cruise_throttle_brake(speed_mph, ctrl.target_speed_mph)
            else:
                base_throttle, base_brake = ctrl.manual_throttle, ctrl.manual_brake

            # Apply AEB override
            throttle = base_throttle
            brake = max(base_brake, aeb_brake)

            if aeb_active:
                throttle = 0.0  # AEB cuts throttle

            ego.apply_control(
                carla.VehicleControl(
                    throttle=float(clamp(throttle, 0.0, 1.0)),
                    brake=float(clamp(brake, 0.0, 1.0)),
                    steer=float(ctrl.steer),
                    hand_brake=bool(ctrl.hand_brake),
                    reverse=bool(ctrl.reverse),
                )
            )

            # Visualize
            vis = frame
            if detector.use_corridor:
                vis = detector.draw_corridor(vis)
            vis = detector.draw_detections(vis, detections)

            lines = [
                f"Speed: {speed_mph:.1f} mph | Cruise: {'ON' if ctrl.cruise_enabled else 'OFF'} | Target: {ctrl.target_speed_mph:.0f}",
                f"AEB: {decision.state} | BrakeCmd: {decision.brake:.2f} | Danger: {decision.danger_score:.2f}",
                f"Reason: {decision.reason}",
                f"Closest: {decision.closest_class} conf={decision.closest_conf:.2f} close={decision.closest_closeness:.2f}" if decision.closest_class else "Closest: None",
                f"FPS: {current_fps:.1f}",
            ]
            vis = overlay_lines(vis, lines)

            cv2.imshow("AEB Controller Test (CARLA)", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                return

    finally:
        try:
            set_synchronous(world, enable=False)
        except Exception:
            pass

        for a in actors[::-1]:
            try:
                a.destroy()
            except Exception:
                pass

        cv2.destroyAllWindows()
        pygame.quit()
        print("Clean exit.")


if __name__ == "__main__":
    main()
