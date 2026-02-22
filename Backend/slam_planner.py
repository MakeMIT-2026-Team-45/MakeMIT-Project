"""
slam_planner.py — 2D SLAM with A* path planning.

Runs on the server/laptop OUTSIDE the Docker container.
Subscribes to angle-tagged ultrasonic readings via MQTT, maintains a
Bayesian log-odds 2D occupancy grid, dead-reckons the robot pose from
published motor commands, runs A* to a user-specified goal, and publishes
motor commands + full SLAM state back at a fixed planning rate.

MQTT topics:
  Subscribe: robot/{id}/sensor/ultrasonic   {"angle_deg":45,"distance_cm":82,"ts":…}
  Subscribe: robot/{id}/drive/goal          {"x":2.0,"y":0.0}
  Publish:   robot/{id}/control/motors      {"left":1,"right":1}
  Publish:   robot/{id}/slam/state          {pose,grid_data(base64),path,goal,mode,…}
"""

import argparse
import base64
import collections
import heapq
import json
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    broker_host: str
    broker_port: int
    robot_id: int
    grid_size: int
    cell_size_m: float
    robot_speed_m_s: float
    robot_omega_rad_s: float
    waypoint_radius_m: float
    turn_threshold_rad: float
    replan_interval_sec: float
    max_sensor_range_m: float
    log_odds_free: float
    log_odds_occ: float
    log_odds_min: float
    log_odds_max: float
    occ_threshold: float
    mock: bool


def parse_args() -> PlannerConfig:
    parser = argparse.ArgumentParser(
        description=(
            "SLAM planner: builds a 2D occupancy grid from ultrasonic sweeps, "
            "runs A* to navigate to a goal, publishes motor commands via MQTT."
        )
    )
    parser.add_argument("--broker-host", type=str, default="localhost")
    parser.add_argument("--broker-port", type=int, default=1880)
    parser.add_argument("--robot-id", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=200,
                        help="Occupancy grid side length in cells (NxN).")
    parser.add_argument("--cell-size-m", type=float, default=0.05,
                        help="Meters per grid cell (default 5cm).")
    parser.add_argument("--robot-speed-m-s", type=float, default=0.2,
                        help="Estimated robot translation speed (m/s).")
    parser.add_argument("--robot-omega-rad-s", type=float, default=2.0,
                        help="Estimated robot rotation speed during full spin (rad/s).")
    parser.add_argument("--waypoint-radius-m", type=float, default=0.15,
                        help="Distance to waypoint that counts as 'reached' (m).")
    parser.add_argument("--turn-threshold-rad", type=float, default=0.3,
                        help="Angle error above which robot spins instead of drives forward (rad).")
    parser.add_argument("--replan-interval-sec", type=float, default=1.0,
                        help="How often A* replans (seconds).")
    parser.add_argument("--max-sensor-range-m", type=float, default=4.0,
                        help="Maximum trusted sensor range (m); beyond this treated as free ray.")
    parser.add_argument("--log-odds-free", type=float, default=-0.4,
                        help="Log-odds decrement for cells along free ray.")
    parser.add_argument("--log-odds-occ", type=float, default=0.85,
                        help="Log-odds increment for occupied endpoint cell.")
    parser.add_argument("--log-odds-min", type=float, default=-20.0)
    parser.add_argument("--log-odds-max", type=float, default=20.0)
    parser.add_argument("--occ-threshold", type=float, default=0.5,
                        help="Log-odds above this value is treated as occupied in A*.")
    parser.add_argument("--mock", action="store_true",
                        help="Inject synthetic sensor data; useful for testing without a Pi.")
    args = parser.parse_args()
    return PlannerConfig(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        robot_id=args.robot_id,
        grid_size=args.grid_size,
        cell_size_m=args.cell_size_m,
        robot_speed_m_s=args.robot_speed_m_s,
        robot_omega_rad_s=args.robot_omega_rad_s,
        waypoint_radius_m=args.waypoint_radius_m,
        turn_threshold_rad=args.turn_threshold_rad,
        replan_interval_sec=args.replan_interval_sec,
        max_sensor_range_m=args.max_sensor_range_m,
        log_odds_free=args.log_odds_free,
        log_odds_occ=args.log_odds_occ,
        log_odds_min=args.log_odds_min,
        log_odds_max=args.log_odds_max,
        occ_threshold=args.occ_threshold,
        mock=args.mock,
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RobotPose:
    x: float = 0.0      # meters from start, +x = initial forward
    y: float = 0.0      # meters from start, +y = initial left
    theta: float = 0.0  # radians; 0 = facing +x, increases CCW (standard math)


@dataclass
class PlannerState:
    pose: RobotPose = field(default_factory=RobotPose)
    goal: tuple[float, float] | None = None
    path: list[tuple[float, float]] = field(default_factory=list)
    current_waypoint_idx: int = 0
    current_cmd: tuple[int, int] = (0, 0)
    mode: str = "idle"          # "idle" | "navigating" | "obstacle" | "arrived"
    last_cmd_ts: float = 0.0
    last_cmd: tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def world_to_grid(
    x: float,
    y: float,
    grid_size: int,
    cell_size_m: float,
) -> tuple[int, int]:
    """Convert world coords (m) to grid (row, col). Returns (-1,-1) if out of bounds."""
    col = int(x / cell_size_m) + grid_size // 2
    row = int(y / cell_size_m) + grid_size // 2
    if 0 <= row < grid_size and 0 <= col < grid_size:
        return row, col
    return -1, -1


def grid_to_world(
    row: int,
    col: int,
    grid_size: int,
    cell_size_m: float,
) -> tuple[float, float]:
    """Convert grid cell center to world coords (m)."""
    x = (col - grid_size // 2 + 0.5) * cell_size_m
    y = (row - grid_size // 2 + 0.5) * cell_size_m
    return x, y


# ---------------------------------------------------------------------------
# SLAM core
# ---------------------------------------------------------------------------

def sensor_to_global(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    servo_angle_deg: float,
    distance_cm: float,
) -> tuple[float, float]:
    """
    Convert a servo angle + distance reading to a global (x, y) hit point.
    Servo convention: -20°=robot's right, 80°=robot's forward, 180°=robot's left.
    """
    sensor_angle_global = robot_theta + math.radians(servo_angle_deg - 80.0)
    dist_m = distance_cm / 100.0
    hit_x = robot_x + dist_m * math.cos(sensor_angle_global)
    hit_y = robot_y + dist_m * math.sin(sensor_angle_global)
    return hit_x, hit_y


def bresenham_cells(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> list[tuple[int, int]]:
    """
    Return grid cells along the line from (r0,c0) to (r1,c1) using Bresenham's
    algorithm. Does NOT include the endpoint — endpoint is marked separately.
    """
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while (r, c) != (r1, c1):
        cells.append((r, c))
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def update_occupancy_grid(
    log_odds_grid: np.ndarray,
    pose: RobotPose,
    angle_deg: float,
    distance_cm: float,
    cfg: PlannerConfig,
) -> None:
    """
    Update the log-odds grid from one sensor reading.
    Marks the ray path as free and the endpoint as occupied.
    Modifies log_odds_grid in-place.
    """
    max_dist_cm = cfg.max_sensor_range_m * 100.0
    is_max_range = distance_cm >= max_dist_cm
    if is_max_range:
        distance_cm = max_dist_cm

    hit_x, hit_y = sensor_to_global(pose.x, pose.y, pose.theta, angle_deg, distance_cm)
    robot_r, robot_c = world_to_grid(pose.x, pose.y, cfg.grid_size, cfg.cell_size_m)
    hit_r, hit_c = world_to_grid(hit_x, hit_y, cfg.grid_size, cfg.cell_size_m)

    if robot_r == -1 or hit_r == -1:
        return

    # Mark ray cells (excluding endpoint) as free
    for r, c in bresenham_cells(robot_r, robot_c, hit_r, hit_c):
        log_odds_grid[r, c] = np.clip(
            log_odds_grid[r, c] + cfg.log_odds_free,
            cfg.log_odds_min,
            cfg.log_odds_max,
        )

    # Mark endpoint as occupied only for actual reflections (not max-range)
    if not is_max_range:
        log_odds_grid[hit_r, hit_c] = np.clip(
            log_odds_grid[hit_r, hit_c] + cfg.log_odds_occ,
            cfg.log_odds_min,
            cfg.log_odds_max,
        )


def log_odds_to_uint8(log_odds_grid: np.ndarray) -> np.ndarray:
    """
    Convert float64 log-odds to uint8 for MQTT transmission.
    0   = unknown  (|log_odds| ≤ 0.5)
    100 = free     (log_odds < -0.5)
    255 = occupied (log_odds > +0.5)
    """
    result = np.zeros(log_odds_grid.shape, dtype=np.uint8)
    result[log_odds_grid < -0.5] = 100
    result[log_odds_grid >  0.5] = 255
    return result


# ---------------------------------------------------------------------------
# Pose dead-reckoning
# ---------------------------------------------------------------------------

def update_pose(
    pose: RobotPose,
    cmd: tuple[int, int],
    dt: float,
    robot_speed_m_s: float,
    robot_omega_rad_s: float,
) -> RobotPose:
    """
    Dead-reckon the robot pose given the motor command applied for dt seconds.

    Sign convention (matches drive_client.py motor GPIO encoding):
      (1,  1): forward         → translate +x direction
      (-1,-1): backward        → translate -x direction
      (-1, 1): spin CCW        → theta increases (left wheel back, right forward)
      (1, -1): spin CW         → theta decreases (left forward, right back)
      (0,  1): arc CW          → half angular + half translation
      (1,  0): arc CCW         → half angular + half translation
      (0,  0): stopped
    """
    left, right = cmd
    nx, ny, nt = pose.x, pose.y, pose.theta

    if left == 1 and right == 1:
        nx += math.cos(pose.theta) * robot_speed_m_s * dt
        ny += math.sin(pose.theta) * robot_speed_m_s * dt
    elif left == -1 and right == -1:
        nx -= math.cos(pose.theta) * robot_speed_m_s * dt
        ny -= math.sin(pose.theta) * robot_speed_m_s * dt
    elif left == -1 and right == 1:
        nt += robot_omega_rad_s * dt           # spin CCW
    elif left == 1 and right == -1:
        nt -= robot_omega_rad_s * dt           # spin CW
    elif left == 0 and right == 1:
        nt += robot_omega_rad_s * 0.5 * dt
        nx += math.cos(pose.theta) * robot_speed_m_s * 0.5 * dt
        ny += math.sin(pose.theta) * robot_speed_m_s * 0.5 * dt
    elif left == 1 and right == 0:
        nt -= robot_omega_rad_s * 0.5 * dt
        nx += math.cos(pose.theta) * robot_speed_m_s * 0.5 * dt
        ny += math.sin(pose.theta) * robot_speed_m_s * 0.5 * dt

    # Normalize theta to [-π, π]
    nt = math.atan2(math.sin(nt), math.cos(nt))
    return RobotPose(x=nx, y=ny, theta=nt)


# ---------------------------------------------------------------------------
# A* path planning
# ---------------------------------------------------------------------------

def run_astar(
    log_odds_grid: np.ndarray,
    start_pose: RobotPose,
    goal: tuple[float, float],
    cfg: PlannerConfig,
) -> list[tuple[float, float]]:
    """
    A* from start_pose to goal on the occupancy grid.
    Returns a list of world-coordinate waypoints (excluding start, including goal).
    Returns empty list if goal is unreachable.

    Cell traversal cost:
      occupied (log_odds > occ_threshold): 1000 (very expensive but not infinite)
      free/unknown: 1.0 (cardinal), sqrt(2) (diagonal)
    """
    start_r, start_c = world_to_grid(start_pose.x, start_pose.y, cfg.grid_size, cfg.cell_size_m)
    goal_r, goal_c = world_to_grid(goal[0], goal[1], cfg.grid_size, cfg.cell_size_m)

    if start_r == -1 or goal_r == -1:
        return []

    if (start_r, start_c) == (goal_r, goal_c):
        return [goal]

    def heuristic(r: int, c: int) -> float:
        return math.sqrt((r - goal_r) ** 2 + (c - goal_c) ** 2)

    # (f_score, counter, g_score, row, col)
    counter = 0
    open_set: list[tuple[float, int, float, int, int]] = [
        (heuristic(start_r, start_c), counter, 0.0, start_r, start_c)
    ]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {(start_r, start_c): 0.0}

    neighbors_8 = [
        (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
        ( 0, -1, 1.0),                         ( 0, 1, 1.0),
        ( 1, -1, math.sqrt(2)), ( 1, 0, 1.0), ( 1, 1, math.sqrt(2)),
    ]

    while open_set:
        _, _, g, r, c = heapq.heappop(open_set)

        if g > g_score.get((r, c), float("inf")):
            continue  # stale entry

        if (r, c) == (goal_r, goal_c):
            # Reconstruct path
            path_cells: list[tuple[int, int]] = []
            cur = (r, c)
            while cur in came_from:
                path_cells.append(cur)
                cur = came_from[cur]
            path_cells.reverse()
            return [grid_to_world(pr, pc, cfg.grid_size, cfg.cell_size_m) for pr, pc in path_cells]

        for dr, dc, step_cost in neighbors_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < cfg.grid_size and 0 <= nc < cfg.grid_size):
                continue

            cell_penalty = 1000.0 if log_odds_grid[nr, nc] > cfg.occ_threshold else 1.0
            tentative_g = g + step_cost * cell_penalty

            neighbor = (nr, nc)
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = (r, c)
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(nr, nc)
                counter += 1
                heapq.heappush(open_set, (f, counter, tentative_g, nr, nc))

    return []  # no path found


# ---------------------------------------------------------------------------
# Waypoint following controller
# ---------------------------------------------------------------------------

def compute_motor_cmd(
    pose: RobotPose,
    waypoint: tuple[float, float],
    turn_threshold_rad: float,
    waypoint_radius_m: float,
) -> tuple[tuple[int, int], bool]:
    """
    Compute motor command to steer toward waypoint.
    Returns ((left, right), reached_waypoint).
    """
    dx = waypoint[0] - pose.x
    dy = waypoint[1] - pose.y
    distance = math.sqrt(dx * dx + dy * dy)

    if distance < waypoint_radius_m:
        return (0, 0), True

    target_angle = math.atan2(dy, dx)
    angle_error = target_angle - pose.theta
    # Normalize to [-π, π]
    angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

    if abs(angle_error) > turn_threshold_rad:
        if angle_error > 0:
            return (-1, 1), False   # turn CCW (left)
        else:
            return (1, -1), False   # turn CW (right)
    else:
        return (1, 1), False        # drive forward


# ---------------------------------------------------------------------------
# Mock sensor injection
# ---------------------------------------------------------------------------

def _mock_room_distance_cm(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    servo_angle_deg: float,
) -> float:
    """
    Simulate a rectangular room: 3m wide (+/-1.5m in x), 3m deep (+/-1.5m in y).
    Traces a ray from robot position at the given global angle and returns the
    distance to the nearest wall.
    """
    sensor_angle = robot_theta + math.radians(servo_angle_deg - 80.0)
    dx = math.cos(sensor_angle)
    dy = math.sin(sensor_angle)

    # Room walls: x in [-1.5, 1.5], y in [-1.5, 1.5]
    walls = [
        # (normal_x, normal_y, wall_x_or_y)
        (1.0, 0.0, 1.5),   # right wall at x=1.5
        (-1.0, 0.0, -1.5), # left wall at x=-1.5
        (0.0, 1.0, 1.5),   # top wall at y=1.5
        (0.0, -1.0, -1.5), # bottom wall at y=-1.5
    ]

    min_dist = 4.0  # max sensor range
    for nx_w, ny_w, offset in walls:
        # Ray-plane intersection
        denom = nx_w * dx + ny_w * dy
        if abs(denom) < 1e-9:
            continue
        num = offset - (nx_w * robot_x + ny_w * robot_y)
        t = num / denom
        if t > 0.01:
            min_dist = min(min_dist, t)

    noise = hash((servo_angle_deg, robot_x)) % 10 - 5  # ±5 cm
    dist_cm = min_dist * 100.0 + noise
    return round(max(5.0, min(400.0, dist_cm)), 1)


# ---------------------------------------------------------------------------
# Main planner loop
# ---------------------------------------------------------------------------

def run(cfg: PlannerConfig) -> None:
    import paho.mqtt.client as mqtt_client

    log_odds_grid = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float64)
    state = PlannerState()
    sensor_queue: collections.deque[tuple[float, float, float]] = collections.deque(maxlen=500)
    goal_lock = threading.Lock()

    motor_topic  = f"robot/{cfg.robot_id}/control/motors"
    sensor_topic = f"robot/{cfg.robot_id}/sensor/ultrasonic"
    slam_topic   = f"robot/{cfg.robot_id}/slam/state"
    goal_topic   = f"robot/{cfg.robot_id}/drive/goal"

    client = mqtt_client.Client()

    def on_connect(c, userdata, flags, rc):
        print(f"[MQTT] connected to {cfg.broker_host}:{cfg.broker_port} (rc={rc})")
        c.subscribe(sensor_topic)
        c.subscribe(goal_topic)
        print(f"[MQTT] subscribed to {sensor_topic}, {goal_topic}")

    def on_message(c, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            if msg.topic == sensor_topic:
                angle = float(data["angle_deg"])
                dist  = float(data["distance_cm"])
                ts    = float(data.get("ts", time.time()))
                sensor_queue.appendleft((angle, dist, ts))
            elif msg.topic == goal_topic:
                with goal_lock:
                    state.goal = (float(data["x"]), float(data["y"]))
                    state.mode = "navigating"
                    state.path = []
                    state.current_waypoint_idx = 0
                print(f"[goal] set to {state.goal}")
        except Exception as exc:
            print(f"[MQTT] bad message on {msg.topic}: {exc}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(cfg.broker_host, cfg.broker_port)
    client.loop_start()

    print(f"SLAM planner  robot_id={cfg.robot_id}  grid={cfg.grid_size}x{cfg.grid_size}  cell={cfg.cell_size_m*100:.0f}cm")
    print(f"Thresholds    turn={cfg.turn_threshold_rad:.2f}rad  waypoint_r={cfg.waypoint_radius_m}m")
    print(f"Replan        every {cfg.replan_interval_sec}s")
    if cfg.mock:
        print("Mode: MOCK (synthetic sensor data, no Pi required)")
    print("Send goal:  mosquitto_pub -t robot/1/drive/goal -m '{\"x\":2.0,\"y\":0.0}'")
    print("Press Ctrl+C to stop.\n")

    last_replan_ts = 0.0
    state.last_cmd_ts = time.time()

    # Mock sweep state
    mock_angle = 0.0
    mock_direction = 1

    try:
        while True:
            t0 = time.monotonic()
            now = time.time()

            # 1. Dead-reckon pose from the last published motor command
            dt = now - state.last_cmd_ts
            if dt > 0 and state.last_cmd_ts > 0:
                state.pose = update_pose(
                    state.pose, state.last_cmd, dt,
                    cfg.robot_speed_m_s, cfg.robot_omega_rad_s,
                )
            state.last_cmd_ts = now

            # 2. In mock mode, inject synthetic sensor readings
            if cfg.mock:
                for _ in range(6):  # inject a few readings per loop
                    dist_cm = _mock_room_distance_cm(
                        state.pose.x, state.pose.y, state.pose.theta, mock_angle
                    )
                    sensor_queue.appendleft((mock_angle, dist_cm, now))
                    mock_angle += mock_direction * 10.0
                    if mock_angle >= 180.0:
                        mock_angle = 180.0
                        mock_direction = -1
                    elif mock_angle <= 0.0:
                        mock_angle = 0.0
                        mock_direction = 1

            # 3. Drain sensor queue → update occupancy grid
            drained = 0
            while sensor_queue:
                angle_deg, distance_cm, _ts = sensor_queue.pop()
                update_occupancy_grid(log_odds_grid, state.pose, angle_deg, distance_cm, cfg)
                drained += 1

            # 4. Get current goal
            with goal_lock:
                goal = state.goal

            # 5. A* replan if needed
            if goal is not None and state.mode not in ("arrived", "idle"):
                goal_dist = math.sqrt(
                    (state.pose.x - goal[0]) ** 2 + (state.pose.y - goal[1]) ** 2
                )
                if goal_dist < cfg.waypoint_radius_m:
                    state.mode = "arrived"
                    state.path = []
                elif now - last_replan_ts > cfg.replan_interval_sec or not state.path:
                    state.path = run_astar(log_odds_grid, state.pose, goal, cfg)
                    state.current_waypoint_idx = 0
                    last_replan_ts = now
                    if not state.path:
                        state.mode = "obstacle"
                    else:
                        state.mode = "navigating"

            # 6. Compute motor command
            cmd = (0, 0)
            if goal is None or state.mode in ("idle", "arrived"):
                state.mode = "idle" if goal is None else state.mode
                cmd = (0, 0)
            elif state.mode == "obstacle":
                cmd = (0, 0)
            elif state.mode == "navigating" and state.path:
                if state.current_waypoint_idx < len(state.path):
                    waypoint = state.path[state.current_waypoint_idx]
                    cmd, reached = compute_motor_cmd(
                        state.pose, waypoint,
                        cfg.turn_threshold_rad, cfg.waypoint_radius_m,
                    )
                    if reached:
                        state.current_waypoint_idx += 1
                else:
                    # Past all waypoints — drive straight toward goal
                    with goal_lock:
                        if state.goal:
                            cmd, _ = compute_motor_cmd(
                                state.pose, state.goal,
                                cfg.turn_threshold_rad, cfg.waypoint_radius_m,
                            )

            # 7. Publish motor command
            state.last_cmd = cmd
            state.current_cmd = cmd
            client.publish(motor_topic, json.dumps({"left": cmd[0], "right": cmd[1]}))

            # 8. Build and publish SLAM state
            uint8_grid = log_odds_to_uint8(log_odds_grid)
            grid_b64 = base64.b64encode(uint8_grid.tobytes()).decode("ascii")

            slam_payload = {
                "pose": {
                    "x": round(state.pose.x, 3),
                    "y": round(state.pose.y, 3),
                    "theta": round(state.pose.theta, 4),
                },
                "goal": {"x": goal[0], "y": goal[1]} if goal else None,
                "path": [[round(p[0], 3), round(p[1], 3)] for p in state.path],
                "current_cmd": {"left": cmd[0], "right": cmd[1]},
                "mode": state.mode,
                "grid_resolution_m": cfg.cell_size_m,
                "grid_width": cfg.grid_size,
                "grid_height": cfg.grid_size,
                "grid_data": grid_b64,
                "ts": now,
            }
            client.publish(slam_topic, json.dumps(slam_payload))

            if cfg.mock:
                occ_count = int(np.sum(uint8_grid == 255))
                free_count = int(np.sum(uint8_grid == 100))
                print(
                    f"[planner] mode={state.mode:12s}  pose=({state.pose.x:.2f},{state.pose.y:.2f})  "
                    f"cmd={cmd}  readings_drained={drained}  occ={occ_count}  free={free_count}"
                )

            # 9. Sleep for remainder of planning interval
            elapsed = time.monotonic() - t0
            sleep_t = cfg.replan_interval_sec - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[planner] shutting down — stopping robot...")
        client.publish(motor_topic, json.dumps({"left": 0, "right": 0}))
    finally:
        client.loop_stop()
        client.disconnect()
        print("[planner] done.")


if __name__ == "__main__":
    run(parse_args())
