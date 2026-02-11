# carla_aeb_live_test.py
from __future__ import annotations

import time
import math
import queue
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import cv2
import pygame
import carla

from detector import YOLODetector
from aeb_controller import AEBController, VehicleKinematics


@dataclass
class EgoControl:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    hand_brake: bool = False
    reverse: bool = False


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def carla_image_to_bgr(image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))  # BGRA
    return arr[:, :, :3]  # BGR


def get_speed_mps(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))


def set_synchronous(world: carla.World, enable: bool, fixed_dt: float = 0.05) -> None:
    settings = world.get_settings()
    settings.synchronous_mode = enable
    settings.fixed_delta_seconds = fixed_dt if enable else None
    world.apply_settings(settings)


def find_open_road_map(client: carla.Client, preferred: str = "Town04") -> carla.World:
    """
    Town04 is a fairly open highway map.
    If it fails, fall back to current world.
    """
    try:
        return client.load_world(preferred)
    except Exception:
        return client.get_world()


def spawn_ego(world: carla.World) -> carla.Vehicle:
    blueprints = world.get_blueprint_library()
    vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found on this map.")

    ego = world.spawn_actor(vehicle_bp, spawn_points[0])
    ego.set_autopilot(False)
    return ego


def spawn_rgb_camera(
    world: carla.World,
    attach_to: carla.Actor,
    width: int = 960,
    height: int = 540,
    fov: int = 90,
) -> Tuple[carla.Sensor, "queue.Queue[carla.Image]"]:
    blueprints = world.get_blueprint_library()
    cam_bp = blueprints.find("sensor.camera.rgb")
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
    blueprints = world.get_blueprint_library()
    candidates = blueprints.filter("vehicle.*")
    veh_bp = candidates[0]

    wp = waypoint_ahead(world, ego, meters)
    transform = wp.transform
    transform.location.z += 0.2

    v = world.try_spawn_actor(veh_bp, transform)
    if v is None:
        raise RuntimeError("Failed to spawn static vehicle. Try again or change distance.")
    v.set_autopilot(False)
    try:
        v.set_simulate_physics(False)  # keeps it parked
    except Exception:
        pass
    return v


def spawn_static_pedestrian_ahead(world: carla.World, ego: carla.Vehicle, meters: float = 20.0) -> carla.Actor:
    blueprints = world.get_blueprint_library()
    walker_bps = blueprints.filter("walker.pedestrian.*")
    if not walker_bps:
        raise RuntimeError("No pedestrian blueprints found.")
    walker_bp = walker_bps[0]

    wp = waypoint_ahead(world, ego, meters)
    transform = wp.transform

    # Put the pedestrian near lane center.
    # You can offset x/y later if you want them on the side.
    transform.location.z += 0.2

    ped = world.try_spawn_actor(walker_bp, transform)
    if ped is None:
        raise RuntimeError("Failed to spawn pedestrian. Try again or change distance.")

    # Keep pedestrian static: no AI controller spawned
    return ped


def init_pygame() -> None:
    pygame.init()
    pygame.display.set_mode((320, 240))
    pygame.display.set_caption("CARLA AEB Test Controls")


def read_controls(ctrl: EgoControl) -> EgoControl:
    """
    Keyboard controls:
      W: increase throttle
      S: brake
      A/D: steer
      Space: hand brake
      R: reset inputs
      Esc: quit
    """
    keys = pygame.key.get_pressed()

    # Throttle and brake
    if keys[pygame.K_w]:
        ctrl.throttle = clamp(ctrl.throttle + 0.03, 0.0, 1.0)
    else:
        ctrl.throttle = clamp(ctrl.throttle - 0.02, 0.0, 1.0)

    if keys[pygame.K_s]:
        ctrl.brake = clamp(ctrl.brake + 0.06, 0.0, 1.0)
        ctrl.throttle = 0.0
    else:
        ctrl.brake = clamp(ctrl.brake - 0.06, 0.0, 1.0)

    # Steering
    steer_target = 0.0
    if keys[pygame.K_a]:
        steer_target = -1.0
    elif keys[pygame.K_d]:
        steer_target = 1.0

    # Smooth steering
    ctrl.steer = ctrl.steer + (steer_target - ctrl.steer) * 0.2
    ctrl.steer = clamp(ctrl.steer, -1.0, 1.0)

    ctrl.hand_brake = bool(keys[pygame.K_SPACE])

    return ctrl


def apply_ego_control(vehicle: carla.Vehicle, ctrl: EgoControl) -> None:
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=ctrl.throttle,
            brake=ctrl.brake,
            steer=ctrl.steer,
            hand_brake=ctrl.hand_brake,
            reverse=ctrl.reverse,
        )
    )


def overlay_status(
    frame: np.ndarray,
    text_lines: List[str],
    x: int = 10,
    y: int = 25,
) -> np.ndarray:
    out = frame.copy()
    for line in text_lines:
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
    return out


def main():
    # Start CARLA server first, then run this script.
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = find_open_road_map(client, preferred="Town04")
    set_synchronous(world, enable=True, fixed_dt=0.05)

    actors_to_destroy: List[carla.Actor] = []

    try:
        ego = spawn_ego(world)
        actors_to_destroy.append(ego)

        camera, img_q = spawn_rgb_camera(world, ego, width=960, height=540, fov=90)
        actors_to_destroy.append(camera)

        # Defaults as requested
        detector = YOLODetector()         # keep detector defaults
        controller = AEBController()      # keep controller defaults

        init_pygame()
        ctrl = EgoControl()

        print("Controls:")
        print("  W/A/S/D drive, Space hand brake, R reset inputs, Esc quit")
        print("  1 spawn pedestrian ahead, 2 spawn static vehicle ahead, 3 clear spawned objects")

        # Simple braking hold so you can see it happening
        hold_brake_until = 0.0
        hold_brake_value = 0.0

        spawned_test_objects: List[carla.Actor] = []

        while True:
            # Pygame event pump
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_r:
                        ctrl = EgoControl()
                    if event.key == pygame.K_1:
                        ped = spawn_static_pedestrian_ahead(world, ego, meters=20.0)
                        spawned_test_objects.append(ped)
                        actors_to_destroy.append(ped)
                        print("Spawned pedestrian ~20m ahead.")
                    if event.key == pygame.K_2:
                        v = spawn_static_vehicle_ahead(world, ego, meters=25.0)
                        spawned_test_objects.append(v)
                        actors_to_destroy.append(v)
                        print("Spawned static vehicle ~25m ahead.")
                    if event.key == pygame.K_3:
                        for a in spawned_test_objects:
                            try:
                                a.destroy()
                            except Exception:
                                pass
                        spawned_test_objects.clear()
                        print("Cleared spawned test objects.")

            # Tick the world (sync mode)
            world.tick()

            # Read camera frame
            frame = None
            try:
                img = img_q.get(timeout=1.0)
                frame = carla_image_to_bgr(img)
            except queue.Empty:
                continue

            # Manual driving input
            ctrl = read_controls(ctrl)

            # Run detection + controller
            detections = detector.predict(frame)

            speed_mps = get_speed_mps(ego)
            decision = controller.update(
                detections=detections,
                frame_shape=frame.shape,
                vehicle=VehicleKinematics(speed_mps=speed_mps),
            )

            # Decide whether to override with AEB braking
            now = time.time()
            if decision.command.name in ("SOFT_BRAKE", "HARD_BRAKE"):
                hold_brake_value = decision.brake
                hold_brake_until = now + 0.8  # hold for visibility

            if now < hold_brake_until:
                # Override user input with AEB brake
                ctrl.throttle = 0.0
                ctrl.brake = max(ctrl.brake, hold_brake_value)

            apply_ego_control(ego, ctrl)

            # Visualize
            vis = frame
            if detector.use_corridor:
                vis = detector.draw_corridor(vis)
            vis = detector.draw_detections(vis, detections)

            mph = speed_mps * 2.23693629
            lines = [
                f"Speed: {mph:.1f} mph",
                f"State: {decision.state}",
                f"Cmd: {decision.command} brake={decision.brake:.2f}",
                f"Risk: {decision.risk:.2f} close={decision.closeness_now:.2f} appr={decision.approach_score:.2f}",
                f"Reason: {decision.reason}",
            ]
            vis = overlay_status(vis, lines)

            cv2.imshow("CARLA AEB Live Test", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                return

    finally:
        try:
            set_synchronous(world, enable=False)
        except Exception:
            pass

        for a in actors_to_destroy[::-1]:
            try:
                a.destroy()
            except Exception:
                pass

        cv2.destroyAllWindows()
        pygame.quit()
        print("Clean exit.")


if __name__ == "__main__":
    main()
