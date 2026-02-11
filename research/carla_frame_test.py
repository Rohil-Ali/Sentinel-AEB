import math
import time
import queue
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import carla
from ultralytics import YOLO


# -----------------------------
# Config (tweak these)
# -----------------------------
EGO_VEHICLE_FILTER = "vehicle.tesla.model3"

IMG_W, IMG_H = 960, 540
FOV = 90

# Spawn settings
NUM_NPC_VEHICLES = 4
NUM_PEDESTRIANS = 6
SPAWN_START_M = 15.0      # first spawn distance in front
SPAWN_GAP_M = 10.0        # spacing between spawns

# "In front" ground-truth settings
MAX_FRONT_DISTANCE_M = 60.0
FRONT_CONE_DEG = 35.0     # actors must be within this angle from forward vector

# YOLO settings
YOLO_WEIGHTS = "best.pt"
CONF = 0.25
IOU = 0.45

# Evaluate for this long
EVAL_SECONDS = 20.0

# "In front" for YOLO detections (image-space heuristic)
# We only count detections whose box center lies in this central region.
CENTER_X_MIN = 0.25
CENTER_X_MAX = 0.75
CENTER_Y_MIN = 0.20
CENTER_Y_MAX = 0.95

# Classes we care about (COCO names)
TARGET_NAMES = {'car', 'pedestrian', 'traffic light', 'traffic sign'}


@dataclass
class EvalStats:
    gt_positive_frames: int = 0
    detected_when_gt_positive: int = 0
    missed_when_gt_positive: int = 0

    gt_negative_frames: int = 0
    false_positive_when_gt_negative: int = 0
    correct_negative_when_gt_negative: int = 0


# -----------------------------
# Helper math
# -----------------------------
def vec_length(v: carla.Vector3D) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def normalize(v: carla.Vector3D) -> carla.Vector3D:
    mag = vec_length(v)
    if mag < 1e-6:
        return carla.Vector3D(0.0, 0.0, 0.0)
    return carla.Vector3D(v.x / mag, v.y / mag, v.z / mag)


def dot(a: carla.Vector3D, b: carla.Vector3D) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def angle_deg_between(a: carla.Vector3D, b: carla.Vector3D) -> float:
    a_n = normalize(a)
    b_n = normalize(b)
    d = max(-1.0, min(1.0, dot(a_n, b_n)))
    return math.degrees(math.acos(d))


# -----------------------------
# CARLA spawn + camera
# -----------------------------
def carla_image_to_bgr(image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))  # BGRA
    return arr[:, :, :3]  # BGR


def spawn_ego(world: carla.World, blueprint_library: carla.BlueprintLibrary) -> carla.Vehicle:
    vehicle_bp = blueprint_library.filter(EGO_VEHICLE_FILTER)[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found on the map.")
    ego = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    if ego is None:
        # try a few
        for sp in spawn_points[1:10]:
            ego = world.try_spawn_actor(vehicle_bp, sp)
            if ego is not None:
                break
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle.")
    ego.set_autopilot(True)
    return ego


def attach_camera(world: carla.World, blueprint_library: carla.BlueprintLibrary, ego: carla.Vehicle):
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))
    cam_bp.set_attribute("fov", str(FOV))

    cam_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
    return camera


def waypoint_in_front(map_obj: carla.Map, ego_transform: carla.Transform, distance_m: float) -> Optional[carla.Transform]:
    wp = map_obj.get_waypoint(ego_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return None
    next_wps = wp.next(distance_m)
    if not next_wps:
        return None
    # Use the first next waypoint
    return next_wps[0].transform


def spawn_npc_vehicles_in_front(world: carla.World, blueprint_library: carla.BlueprintLibrary, ego: carla.Vehicle,
                                n: int) -> List[carla.Actor]:
    spawned = []
    map_obj = world.get_map()
    ego_tf = ego.get_transform()

    vehicle_bps = blueprint_library.filter("vehicle.*")
    # Avoid bikes sometimes small; keep it simple
    vehicle_bps = [bp for bp in vehicle_bps if "isetta" not in bp.id and "carlacola" not in bp.id]

    for i in range(n):
        dist = SPAWN_START_M + i * SPAWN_GAP_M
        tf = waypoint_in_front(map_obj, ego_tf, dist)
        if tf is None:
            continue

        bp = np.random.choice(vehicle_bps)
        actor = world.try_spawn_actor(bp, tf)
        if actor:
            # Keep them still for controlled testing (no autopilot)
            actor.set_autopilot(True)
            spawned.append(actor)

    return spawned


def spawn_pedestrians_in_front(world: carla.World, blueprint_library: carla.BlueprintLibrary, ego: carla.Vehicle,
                               n: int) -> List[carla.Actor]:
    spawned = []
    map_obj = world.get_map()
    ego_tf = ego.get_transform()

    walker_bps = blueprint_library.filter("walker.pedestrian.*")

    # Place walkers near the road ahead (slightly to the side)
    for i in range(n):
        dist = SPAWN_START_M + (i % max(1, NUM_NPC_VEHICLES)) * SPAWN_GAP_M + 5.0
        tf = waypoint_in_front(map_obj, ego_tf, dist)
        if tf is None:
            continue

        # Offset left/right on the sidewalk-ish direction.
        # This isn't perfect, but works well enough for "in front" tests.
        right = tf.get_right_vector()
        side = -1.5 if (i % 2 == 0) else 1.5
        loc = tf.location + carla.Location(x=right.x * side, y=right.y * side, z=0.5)
        walker_tf = carla.Transform(loc, tf.rotation)

        bp = np.random.choice(walker_bps)
        actor = world.try_spawn_actor(bp, walker_tf)
        if actor:
            spawned.append(actor)

    return spawned


# -----------------------------
# Ground-truth: "is there a target in front?"
# -----------------------------
def is_actor_target(actor: carla.Actor) -> bool:
    tid = actor.type_id
    if tid.startswith("walker.pedestrian"):
        return True
    if tid.startswith("vehicle."):
        return True
    return False


def actor_in_front_cone(ego: carla.Vehicle, actor: carla.Actor) -> bool:
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()

    a_loc = actor.get_transform().location
    rel = carla.Vector3D(a_loc.x - ego_loc.x, a_loc.y - ego_loc.y, a_loc.z - ego_loc.z)

    dist = vec_length(rel)
    if dist < 1e-3 or dist > MAX_FRONT_DISTANCE_M:
        return False

    # must be in front (dot > 0)
    if dot(normalize(rel), normalize(forward)) <= 0.0:
        return False

    # within cone angle
    ang = angle_deg_between(forward, rel)
    return ang <= FRONT_CONE_DEG


def gt_has_target_in_front(world: carla.World, ego: carla.Vehicle) -> bool:
    actors = world.get_actors()
    for a in actors:
        if a.id == ego.id:
            continue
        if not is_actor_target(a):
            continue
        if actor_in_front_cone(ego, a):
            return True
    return False


# -----------------------------
# YOLO: "did it detect a target in front?"
# -----------------------------
def yolo_detected_target_in_front(model: YOLO, frame_bgr: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Returns:
      - bool: True if YOLO found at least one target detection in the central region
      - annotated frame for visual checking
    """
    results = model.predict(frame_bgr, conf=CONF, iou=IOU, verbose=False)
    r0 = results[0]

    annotated = r0.plot()  # BGR with boxes

    found = False

    # Boxes in xyxy, class ids, confidences
    if r0.boxes is None or len(r0.boxes) == 0:
        return False, annotated

    names = model.names  # class id -> name

    for b in r0.boxes:
        cls_id = int(b.cls.item())
        name = names.get(cls_id, str(cls_id))
        if name not in TARGET_NAMES:
            continue

        xyxy = b.xyxy[0].cpu().numpy().astype(float)
        x1, y1, x2, y2 = xyxy.tolist()

        # center normalized
        cx = ((x1 + x2) / 2.0) / IMG_W
        cy = ((y1 + y2) / 2.0) / IMG_H

        # only count "in front" detections roughly in front view
        if CENTER_X_MIN <= cx <= CENTER_X_MAX and CENTER_Y_MIN <= cy <= CENTER_Y_MAX:
            found = True
            break

    return found, annotated


# -----------------------------
# Main evaluation loop
# -----------------------------
def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    actors_to_destroy: List[carla.Actor] = []

    # Load YOLO once
    print("Loading YOLO...")
    model = YOLO(YOLO_WEIGHTS)

    # Spawn ego + camera
    ego = spawn_ego(world, blueprint_library)
    actors_to_destroy.append(ego)

    camera = attach_camera(world, blueprint_library, ego)
    actors_to_destroy.append(camera)

    # Spawn NPCs in front
    npc_vehicles = spawn_npc_vehicles_in_front(world, blueprint_library, ego, NUM_NPC_VEHICLES)
    actors_to_destroy.extend(npc_vehicles)

    pedestrians = spawn_pedestrians_in_front(world, blueprint_library, ego, NUM_PEDESTRIANS)
    actors_to_destroy.extend(pedestrians)

    print(f"Spawned: ego=1, vehicles={len(npc_vehicles)}, pedestrians={len(pedestrians)}")

    # Camera queue
    img_q: "queue.Queue[carla.Image]" = queue.Queue(maxsize=2)

    def _on_img(img: carla.Image):
        # keep only newest frame
        try:
            while img_q.qsize() > 0:
                img_q.get_nowait()
        except queue.Empty:
            pass
        img_q.put(img)

    camera.listen(_on_img)

    stats = EvalStats()

    print("Running evaluation. Press Q to quit early.")
    t_start = time.perf_counter()

    try:
        while True:
            if (time.perf_counter() - t_start) >= EVAL_SECONDS:
                break

            try:
                img = img_q.get(timeout=2.0)
            except queue.Empty:
                print("No frames received. Is CARLA running smoothly?")
                continue

            frame = carla_image_to_bgr(img)

            # Ground truth: is there any target in front?
            gt_front = gt_has_target_in_front(world, ego)

            # YOLO: did it detect any target in front (image heuristic)?
            yolo_front, annotated = yolo_detected_target_in_front(model, frame)

            # Update stats
            if gt_front:
                stats.gt_positive_frames += 1
                if yolo_front:
                    stats.detected_when_gt_positive += 1
                else:
                    stats.missed_when_gt_positive += 1
            else:
                stats.gt_negative_frames += 1
                if yolo_front:
                    stats.false_positive_when_gt_negative += 1
                else:
                    stats.correct_negative_when_gt_negative += 1

            # On-screen debug
            cv2.putText(
                annotated,
                f"GT front: {gt_front} | YOLO front: {yolo_front}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Option 2 Eval (YOLO on CARLA)", annotated)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    finally:
        # Cleanup
        try:
            camera.stop()
        except Exception:
            pass

        for a in reversed(actors_to_destroy):
            try:
                a.destroy()
            except Exception:
                pass

        cv2.destroyAllWindows()

    # Compute metrics
    print("\n--- Results ---")
    if stats.gt_positive_frames > 0:
        accuracy = 100.0 * (stats.detected_when_gt_positive / stats.gt_positive_frames)
        miss_rate = 100.0 * (stats.missed_when_gt_positive / stats.gt_positive_frames)
    else:
        accuracy = 0.0
        miss_rate = 0.0

    if stats.gt_negative_frames > 0:
        false_positive_rate = 100.0 * (stats.false_positive_when_gt_negative / stats.gt_negative_frames)
    else:
        false_positive_rate = 0.0

    print(f"GT positive frames (target in front): {stats.gt_positive_frames}")
    print(f"Detected when GT positive:           {stats.detected_when_gt_positive}")
    print(f"Missed when GT positive:             {stats.missed_when_gt_positive}")
    print(f"GT negative frames:                  {stats.gt_negative_frames}")
    print(f"False positives when GT negative:    {stats.false_positive_when_gt_negative}")

    print(f"\nDetection 'accuracy' (recall-like):  {accuracy:.2f}%")
    print(f"Miss rate:                           {miss_rate:.2f}%")
    print(f"False positive rate:                 {false_positive_rate:.2f}%")

    print("\nNote: This 'accuracy' is a simple metric: "
          "when CARLA says a target is in front, did YOLO detect at least one target.")


if __name__ == "__main__":
    main()
