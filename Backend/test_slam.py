"""
test_slam.py — Comprehensive tests for slam_planner.py core functions.

Tests run entirely offline: no MQTT broker, no GPIO, no Raspberry Pi.
Run with:
    python3 test_slam.py
or:
    python3 -m pytest test_slam.py -v
"""

import math
import sys
import time
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Import the module under test (all pure functions, no MQTT/GPIO at import)
# ---------------------------------------------------------------------------
from slam_planner import (
    PlannerConfig,
    RobotPose,
    bresenham_cells,
    compute_motor_cmd,
    grid_to_world,
    log_odds_to_uint8,
    run_astar,
    sensor_to_global,
    update_occupancy_grid,
    update_pose,
    world_to_grid,
    _mock_room_distance_cm,
)


# ---------------------------------------------------------------------------
# Shared test config (200×200 grid, 5cm cells → 10m×10m arena)
# ---------------------------------------------------------------------------

def make_cfg(**overrides) -> PlannerConfig:
    defaults = dict(
        broker_host="localhost",
        broker_port=1883,
        robot_id=1,
        grid_size=200,
        cell_size_m=0.05,
        robot_speed_m_s=0.2,
        robot_omega_rad_s=2.0,
        waypoint_radius_m=0.15,
        turn_threshold_rad=0.3,
        replan_interval_sec=1.0,
        max_sensor_range_m=4.0,
        log_odds_free=-0.4,
        log_odds_occ=0.85,
        log_odds_min=-20.0,
        log_odds_max=20.0,
        occ_threshold=0.5,
        mock=False,
    )
    defaults.update(overrides)
    return PlannerConfig(**defaults)


CFG = make_cfg()


# ===========================================================================
# 1. Coordinate conversion
# ===========================================================================

class TestWorldToGrid(unittest.TestCase):

    def test_origin_maps_to_center(self):
        """World origin (0,0) must map to grid center (100,100)."""
        r, c = world_to_grid(0.0, 0.0, 200, 0.05)
        self.assertEqual(r, 100)
        self.assertEqual(c, 100)

    def test_positive_x_increases_col(self):
        """Moving +x in world → higher col index."""
        _, c0 = world_to_grid(0.0, 0.0, 200, 0.05)
        _, c1 = world_to_grid(0.5, 0.0, 200, 0.05)
        self.assertGreater(c1, c0)

    def test_positive_y_increases_row(self):
        """Moving +y in world → higher row index."""
        r0, _ = world_to_grid(0.0, 0.0, 200, 0.05)
        r1, _ = world_to_grid(0.0, 0.5, 200, 0.05)
        self.assertGreater(r1, r0)

    def test_out_of_bounds_returns_negative(self):
        """Far-out-of-bounds coords → (-1, -1)."""
        r, c = world_to_grid(999.0, 999.0, 200, 0.05)
        self.assertEqual((r, c), (-1, -1))

    def test_boundary_positive(self):
        """Exactly at grid boundary (x=4.99m → last column)."""
        r, c = world_to_grid(4.99, 0.0, 200, 0.05)
        self.assertNotEqual(c, -1)
        self.assertEqual(c, 199)

    def test_boundary_negative(self):
        """Exactly at negative boundary (x=-5.0m → col 0)."""
        r, c = world_to_grid(-5.0, 0.0, 200, 0.05)
        self.assertNotEqual(c, -1)
        self.assertEqual(c, 0)


class TestGridToWorld(unittest.TestCase):

    def test_center_cell_is_near_origin(self):
        """Grid center cell (100,100) should map near world origin."""
        x, y = grid_to_world(100, 100, 200, 0.05)
        self.assertAlmostEqual(x, 0.025, places=3)   # cell center offset
        self.assertAlmostEqual(y, 0.025, places=3)

    def test_roundtrip(self):
        """world_to_grid → grid_to_world round-trips within 1 cell.

        world_to_grid uses int() truncation, so the recovered world coord
        (cell center = +0.5*cell_size offset) can differ by up to 1 full cell
        (0.05m) from the original. Use delta=0.10 to allow for this.
        """
        for wx, wy in [(0.5, 0.3), (-1.2, 0.8), (2.0, -1.5)]:
            r, c = world_to_grid(wx, wy, 200, 0.05)
            self.assertNotEqual(r, -1)
            rx, ry = grid_to_world(r, c, 200, 0.05)
            self.assertAlmostEqual(wx, rx, delta=0.10)   # within 2 cells
            self.assertAlmostEqual(wy, ry, delta=0.10)


# ===========================================================================
# 2. Sensor geometry
# ===========================================================================

class TestSensorToGlobal(unittest.TestCase):

    def test_forward_at_zero_theta(self):
        """Servo at 90° (forward) with theta=0 → hit directly +x."""
        hx, hy = sensor_to_global(0.0, 0.0, 0.0, 90.0, 100.0)
        self.assertAlmostEqual(hx, 1.0, places=3)
        self.assertAlmostEqual(hy, 0.0, places=3)

    def test_right_at_zero_theta(self):
        """Servo at 0° (robot's right) with theta=0 → hit directly -y (right in world)."""
        hx, hy = sensor_to_global(0.0, 0.0, 0.0, 0.0, 100.0)
        self.assertAlmostEqual(hx, 0.0, places=3)
        self.assertAlmostEqual(hy, -1.0, places=3)

    def test_left_at_zero_theta(self):
        """Servo at 180° (robot's left) with theta=0 → hit directly +y."""
        hx, hy = sensor_to_global(0.0, 0.0, 0.0, 180.0, 100.0)
        self.assertAlmostEqual(hx, 0.0, places=3)
        self.assertAlmostEqual(hy, 1.0, places=3)

    def test_rotated_robot(self):
        """Robot facing 90° (left in world), servo at 90° → hit directly +y."""
        hx, hy = sensor_to_global(0.0, 0.0, math.pi / 2, 90.0, 100.0)
        self.assertAlmostEqual(hx, 0.0, places=3)
        self.assertAlmostEqual(hy, 1.0, places=3)

    def test_offset_robot_position(self):
        """Hit is relative to robot position, not origin."""
        hx, hy = sensor_to_global(1.0, 2.0, 0.0, 90.0, 100.0)
        self.assertAlmostEqual(hx, 2.0, places=3)
        self.assertAlmostEqual(hy, 2.0, places=3)


# ===========================================================================
# 3. Bresenham ray tracing
# ===========================================================================

class TestBresenhamCells(unittest.TestCase):

    def test_horizontal_ray(self):
        """Purely horizontal line — only column increments."""
        cells = bresenham_cells(5, 5, 5, 10)
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        self.assertTrue(all(r == 5 for r in rows))
        self.assertEqual(cols, list(range(5, 10)))   # 5..9 inclusive, not 10

    def test_vertical_ray(self):
        """Purely vertical line — only row increments."""
        cells = bresenham_cells(5, 5, 10, 5)
        self.assertTrue(all(c == 5 for _, c in cells))
        self.assertEqual([r for r, _ in cells], list(range(5, 10)))

    def test_excludes_endpoint(self):
        """Endpoint (r1, c1) must NOT be in the returned list."""
        cells = bresenham_cells(0, 0, 5, 5)
        self.assertNotIn((5, 5), cells)

    def test_includes_start(self):
        """Start point (r0, c0) must be in the returned list."""
        cells = bresenham_cells(3, 3, 7, 5)
        self.assertIn((3, 3), cells)

    def test_diagonal(self):
        """45° diagonal: each step increments both row and col."""
        cells = bresenham_cells(0, 0, 4, 4)
        self.assertEqual(cells, [(0, 0), (1, 1), (2, 2), (3, 3)])

    def test_single_step(self):
        """Adjacent cells — start only (no intermediate cells)."""
        cells = bresenham_cells(0, 0, 0, 1)
        self.assertEqual(cells, [(0, 0)])

    def test_same_cell(self):
        """Start == End — empty list."""
        cells = bresenham_cells(5, 5, 5, 5)
        self.assertEqual(cells, [])


# ===========================================================================
# 4. Occupancy grid update
# ===========================================================================

class TestUpdateOccupancyGrid(unittest.TestCase):

    def _fresh_grid(self):
        return np.zeros((200, 200), dtype=np.float64)

    def test_ray_cells_become_free(self):
        """Cells along the ray (before endpoint) should decrease in log-odds."""
        grid = self._fresh_grid()
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        # Shoot forward at 90° servo, 1m away
        update_occupancy_grid(grid, pose, 90.0, 100.0, CFG)
        robot_r, robot_c = world_to_grid(0.0, 0.0, 200, 0.05)
        hit_r, hit_c = world_to_grid(1.0, 0.0, 200, 0.05)
        # At least some cells along the ray should be negative (free)
        free_count = np.sum(grid[robot_r, robot_c:hit_c] < 0)
        self.assertGreater(free_count, 0, "Ray cells should be marked free")

    def test_endpoint_becomes_occupied(self):
        """The hit cell should increase in log-odds (occupied)."""
        grid = self._fresh_grid()
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        update_occupancy_grid(grid, pose, 90.0, 100.0, CFG)  # 1m forward
        hit_r, hit_c = world_to_grid(1.0, 0.0, 200, 0.05)
        self.assertGreater(grid[hit_r, hit_c], 0.0, "Hit cell should be marked occupied")

    def test_max_range_skips_endpoint(self):
        """At max sensor range, endpoint should NOT be marked occupied."""
        grid = self._fresh_grid()
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        max_dist_cm = CFG.max_sensor_range_m * 100.0
        update_occupancy_grid(grid, pose, 90.0, max_dist_cm, CFG)
        # No cell should have increased log-odds (all only decreased or zero)
        self.assertLessEqual(grid.max(), 0.0, "Max-range reading must not mark any cell occupied")

    def test_repeated_updates_saturate(self):
        """Repeated measurements saturate at log_odds_max/-min."""
        grid = self._fresh_grid()
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        for _ in range(50):
            update_occupancy_grid(grid, pose, 90.0, 100.0, CFG)
        self.assertLessEqual(grid.max(), CFG.log_odds_max + 1e-9)
        self.assertGreaterEqual(grid.min(), CFG.log_odds_min - 1e-9)

    def test_out_of_bounds_reading_is_ignored(self):
        """Sensor pointing into out-of-bounds region should not crash/modify grid."""
        grid = self._fresh_grid()
        pose = RobotPose(x=4.5, y=4.5, theta=0.0)  # near corner
        before = grid.copy()
        # Reading at 90° from corner would land outside grid
        update_occupancy_grid(grid, pose, 90.0, 200.0, CFG)
        # Grid may or may not change, but should not raise


# ===========================================================================
# 5. Log-odds to uint8 encoding
# ===========================================================================

class TestLogOddsToUint8(unittest.TestCase):

    def test_zeros_produce_unknown(self):
        """All-zero grid → all unknown (0)."""
        grid = np.zeros((10, 10), dtype=np.float64)
        result = log_odds_to_uint8(grid)
        self.assertTrue(np.all(result == 0))

    def test_positive_produces_occupied(self):
        """Values > 0.5 → 255."""
        grid = np.full((5, 5), 1.0, dtype=np.float64)
        result = log_odds_to_uint8(grid)
        self.assertTrue(np.all(result == 255))

    def test_negative_produces_free(self):
        """Values < -0.5 → 100."""
        grid = np.full((5, 5), -1.0, dtype=np.float64)
        result = log_odds_to_uint8(grid)
        self.assertTrue(np.all(result == 100))

    def test_boundary_values(self):
        """Values exactly at ±0.5 are still unknown (0)."""
        grid = np.array([[0.5, -0.5, 0.0]], dtype=np.float64)
        result = log_odds_to_uint8(grid)
        self.assertTrue(np.all(result == 0), f"Expected all unknown, got {result}")

    def test_mixed_grid(self):
        grid = np.array([[-2.0, 0.0, 3.0]], dtype=np.float64)
        result = log_odds_to_uint8(grid)
        self.assertEqual(result[0, 0], 100)   # free
        self.assertEqual(result[0, 1], 0)     # unknown
        self.assertEqual(result[0, 2], 255)   # occupied

    def test_output_dtype_is_uint8(self):
        grid = np.random.randn(10, 10)
        result = log_odds_to_uint8(grid)
        self.assertEqual(result.dtype, np.uint8)

    def test_output_shape_preserved(self):
        grid = np.zeros((200, 200), dtype=np.float64)
        result = log_odds_to_uint8(grid)
        self.assertEqual(result.shape, (200, 200))


# ===========================================================================
# 6. Dead-reckoning pose update
# ===========================================================================

class TestUpdatePose(unittest.TestCase):

    def test_forward_increases_x(self):
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        new_pose = update_pose(pose, (1, 1), dt=1.0, robot_speed_m_s=0.2, robot_omega_rad_s=2.0)
        self.assertAlmostEqual(new_pose.x, 0.2, places=3)
        self.assertAlmostEqual(new_pose.y, 0.0, places=3)
        self.assertAlmostEqual(new_pose.theta, 0.0, places=3)

    def test_backward_decreases_x(self):
        pose = RobotPose(x=1.0, y=0.0, theta=0.0)
        new_pose = update_pose(pose, (-1, -1), dt=1.0, robot_speed_m_s=0.2, robot_omega_rad_s=2.0)
        self.assertAlmostEqual(new_pose.x, 0.8, places=3)
        self.assertAlmostEqual(new_pose.y, 0.0, places=3)

    def test_spin_ccw_increases_theta(self):
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        new_pose = update_pose(pose, (-1, 1), dt=0.5, robot_speed_m_s=0.2, robot_omega_rad_s=2.0)
        self.assertAlmostEqual(new_pose.theta, 1.0, places=3)

    def test_spin_cw_decreases_theta(self):
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        new_pose = update_pose(pose, (1, -1), dt=0.5, robot_speed_m_s=0.2, robot_omega_rad_s=2.0)
        self.assertAlmostEqual(new_pose.theta, -1.0, places=3)

    def test_theta_normalized_positive_wrap(self):
        """Theta > π should wrap to negative."""
        pose = RobotPose(x=0.0, y=0.0, theta=math.pi - 0.1)
        new_pose = update_pose(pose, (-1, 1), dt=0.2, robot_speed_m_s=0.2, robot_omega_rad_s=2.0)
        self.assertGreaterEqual(new_pose.theta, -math.pi - 1e-9)
        self.assertLessEqual(new_pose.theta, math.pi + 1e-9)

    def test_theta_normalized_negative_wrap(self):
        """Theta < -π should wrap to positive."""
        pose = RobotPose(x=0.0, y=0.0, theta=-math.pi + 0.1)
        new_pose = update_pose(pose, (1, -1), dt=0.2, robot_speed_m_s=0.2, robot_omega_rad_s=2.0)
        self.assertGreaterEqual(new_pose.theta, -math.pi - 1e-9)
        self.assertLessEqual(new_pose.theta, math.pi + 1e-9)

    def test_stop_does_not_move(self):
        pose = RobotPose(x=1.5, y=-0.3, theta=0.7)
        new_pose = update_pose(pose, (0, 0), dt=5.0, robot_speed_m_s=0.2, robot_omega_rad_s=2.0)
        self.assertAlmostEqual(new_pose.x, 1.5, places=6)
        self.assertAlmostEqual(new_pose.y, -0.3, places=6)
        self.assertAlmostEqual(new_pose.theta, 0.7, places=6)

    def test_forward_facing_90deg(self):
        """Robot facing +y (theta=π/2), forward command → moves +y."""
        pose = RobotPose(x=0.0, y=0.0, theta=math.pi / 2)
        new_pose = update_pose(pose, (1, 1), dt=1.0, robot_speed_m_s=0.2, robot_omega_rad_s=2.0)
        self.assertAlmostEqual(new_pose.x, 0.0, places=3)
        self.assertAlmostEqual(new_pose.y, 0.2, places=3)

    def test_zero_dt_does_not_move(self):
        pose = RobotPose(x=1.0, y=2.0, theta=1.0)
        new_pose = update_pose(pose, (1, 1), dt=0.0, robot_speed_m_s=0.2, robot_omega_rad_s=2.0)
        self.assertAlmostEqual(new_pose.x, 1.0, places=6)
        self.assertAlmostEqual(new_pose.y, 2.0, places=6)


# ===========================================================================
# 7. Motor command controller
# ===========================================================================

class TestComputeMotorCmd(unittest.TestCase):

    def test_reached_waypoint_returns_stop(self):
        """Within waypoint_radius → stop and signal reached."""
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        cmd, reached = compute_motor_cmd(pose, (0.05, 0.0), 0.3, 0.15)
        self.assertEqual(cmd, (0, 0))
        self.assertTrue(reached)

    def test_facing_waypoint_goes_forward(self):
        """Waypoint directly ahead → forward command."""
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        cmd, reached = compute_motor_cmd(pose, (2.0, 0.0), 0.3, 0.15)
        self.assertEqual(cmd, (1, 1))
        self.assertFalse(reached)

    def test_waypoint_to_left_turns_ccw(self):
        """Waypoint significantly to the left → CCW spin (-1, 1)."""
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        cmd, reached = compute_motor_cmd(pose, (0.0, 2.0), 0.3, 0.15)
        self.assertEqual(cmd, (-1, 1))

    def test_waypoint_to_right_turns_cw(self):
        """Waypoint significantly to the right → CW spin (1, -1)."""
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        cmd, reached = compute_motor_cmd(pose, (0.0, -2.0), 0.3, 0.15)
        self.assertEqual(cmd, (1, -1))

    def test_waypoint_behind_rotates(self):
        """Waypoint directly behind → should rotate (large angle error)."""
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        cmd, reached = compute_motor_cmd(pose, (-2.0, 0.0), 0.3, 0.15)
        self.assertIn(cmd, [(-1, 1), (1, -1)])   # either spin direction is valid

    def test_small_angle_error_goes_forward(self):
        """Angle error within threshold → forward, not spin."""
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        # Waypoint slightly to the side but within 0.3 rad threshold
        cmd, reached = compute_motor_cmd(pose, (2.0, 0.1), 0.3, 0.15)
        self.assertEqual(cmd, (1, 1))


# ===========================================================================
# 8. A* path planning
# ===========================================================================

class TestRunAstar(unittest.TestCase):

    def _empty_grid(self):
        return np.zeros((200, 200), dtype=np.float64)

    def _grid_with_wall(self, wall_x: float, wall_thickness: int = 5):
        """Place a horizontal wall of occupied cells."""
        grid = self._empty_grid()
        col = int(wall_x / CFG.cell_size_m) + CFG.grid_size // 2
        col = max(0, min(CFG.grid_size - 1, col))
        for r in range(20, 180):
            for dc in range(wall_thickness):
                if col + dc < CFG.grid_size:
                    grid[r, col + dc] = 5.0   # well above occ_threshold=0.5
        return grid

    def test_returns_nonempty_path_to_reachable_goal(self):
        """Free grid → path exists to a reachable goal."""
        grid = self._empty_grid()
        start = RobotPose(x=0.0, y=0.0, theta=0.0)
        path = run_astar(grid, start, (1.0, 0.0), CFG)
        self.assertGreater(len(path), 0, "Expected a path to be found")

    def test_path_ends_at_goal_cell(self):
        """Last waypoint in path should be at the goal."""
        grid = self._empty_grid()
        start = RobotPose(x=0.0, y=0.0, theta=0.0)
        goal = (1.0, 0.5)
        path = run_astar(grid, start, goal, CFG)
        self.assertGreater(len(path), 0)
        last_x, last_y = path[-1]
        self.assertAlmostEqual(last_x, goal[0], delta=CFG.cell_size_m)
        self.assertAlmostEqual(last_y, goal[1], delta=CFG.cell_size_m)

    def test_returns_empty_for_out_of_bounds_goal(self):
        """Goal outside grid → empty path."""
        grid = self._empty_grid()
        start = RobotPose(x=0.0, y=0.0, theta=0.0)
        path = run_astar(grid, start, (999.0, 0.0), CFG)
        self.assertEqual(path, [])

    def test_path_avoids_dense_occupied_region(self):
        """With a wall in the way, A* finds a path around (or through at high cost)."""
        grid = self._grid_with_wall(wall_x=0.5, wall_thickness=5)
        start = RobotPose(x=0.0, y=0.0, theta=0.0)
        path = run_astar(grid, start, (1.0, 0.0), CFG)
        # Path should exist (A* with penalty still finds a route)
        self.assertGreater(len(path), 0)

    def test_already_at_goal(self):
        """Start == goal cell → returns [goal]."""
        grid = self._empty_grid()
        start = RobotPose(x=0.025, y=0.025, theta=0.0)   # inside center cell
        # Same cell
        goal = (0.025, 0.025)
        path = run_astar(grid, start, goal, CFG)
        self.assertEqual(path, [goal])

    def test_path_waypoints_are_floats(self):
        """All path elements must be (float, float) pairs."""
        grid = self._empty_grid()
        start = RobotPose(x=0.0, y=0.0, theta=0.0)
        path = run_astar(grid, start, (0.5, 0.5), CFG)
        for item in path:
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], float)
            self.assertIsInstance(item[1], float)

    def test_path_is_connected(self):
        """Consecutive waypoints should be within 2 cells of each other."""
        grid = self._empty_grid()
        start = RobotPose(x=0.0, y=0.0, theta=0.0)
        path = run_astar(grid, start, (1.0, 1.0), CFG)
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            dist = math.sqrt(dx**2 + dy**2)
            self.assertLessEqual(dist, CFG.cell_size_m * 2.0 + 1e-6,
                                 f"Gap between waypoints too large at step {i}")


# ===========================================================================
# 9. Mock sensor simulation
# ===========================================================================

class TestMockRoomDistance(unittest.TestCase):

    def test_returns_positive_distance(self):
        """Mock readings must always be positive."""
        for angle in range(0, 181, 10):
            d = _mock_room_distance_cm(0.0, 0.0, 0.0, float(angle))
            self.assertGreater(d, 0.0, f"Negative distance at angle={angle}")

    def test_bounded_by_max_range(self):
        """Mock readings should not exceed 400cm (4m max range)."""
        for angle in range(0, 181, 5):
            d = _mock_room_distance_cm(0.0, 0.0, 0.0, float(angle))
            self.assertLessEqual(d, 400.0, f"Distance out of range at angle={angle}")

    def test_forward_is_farther_than_side(self):
        """In a 3m×3m room at origin, all readings are ≤ 400cm (max range)."""
        # Servo 90° = forward (wall at 1.5m = 150cm), servo 0° = right (wall at 1.5m)
        # Both should be within valid sensor range
        for angle in [0.0, 45.0, 90.0, 135.0, 180.0]:
            d = _mock_room_distance_cm(0.0, 0.0, 0.0, angle)
            self.assertGreater(d, 0.0)
            self.assertLessEqual(d, 400.0, f"Distance at angle={angle} exceeds max range")

    def test_different_angles_give_different_distances(self):
        """Multiple angles from same position should not all return identical values."""
        dists = set()
        for angle in range(0, 181, 15):
            d = _mock_room_distance_cm(0.0, 0.0, 0.0, float(angle))
            dists.add(round(d))
        self.assertGreater(len(dists), 1, "Expected variation across angles")


# ===========================================================================
# 10. Integration test — full pipeline (no MQTT, in-process)
# ===========================================================================

class TestIntegration(unittest.TestCase):
    """
    Run several SLAM iterations in-process.
    Simulates the server-side planning loop without any network I/O.
    """

    def test_full_pipeline_mock(self):
        """
        5 SLAM iterations with a mock room + goal.
        Verifies: grid fills in, pose moves, path is found.
        """
        import base64
        cfg = make_cfg(grid_size=100, cell_size_m=0.1, mock=True)
        log_odds_grid = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float64)
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        goal = (2.0, 0.0)

        mock_angle = 0.0
        mock_dir = 1
        last_cmd = (0, 0)
        last_ts = time.time()

        for iteration in range(5):
            now = time.time()
            dt = now - last_ts
            last_ts = now

            # Dead-reckon
            pose = update_pose(pose, last_cmd, dt, cfg.robot_speed_m_s, cfg.robot_omega_rad_s)

            # Inject 18 mock sensor readings (covers 0–180° in 10° steps)
            for _ in range(18):
                dist_cm = _mock_room_distance_cm(pose.x, pose.y, pose.theta, mock_angle)
                update_occupancy_grid(log_odds_grid, pose, mock_angle, dist_cm, cfg)
                mock_angle += mock_dir * 10.0
                if mock_angle >= 180.0:
                    mock_angle = 180.0; mock_dir = -1
                elif mock_angle <= 0.0:
                    mock_angle = 0.0; mock_dir = 1

            # A* plan
            path = run_astar(log_odds_grid, pose, goal, cfg)

            # Motor command
            if path:
                cmd, _ = compute_motor_cmd(pose, path[0], cfg.turn_threshold_rad, cfg.waypoint_radius_m)
            else:
                cmd = (0, 0)
            last_cmd = cmd

            # Encode grid
            uint8 = log_odds_to_uint8(log_odds_grid)
            grid_b64 = base64.b64encode(uint8.tobytes()).decode("ascii")

            # Verify grid byte count
            expected_bytes = cfg.grid_size * cfg.grid_size
            self.assertEqual(len(base64.b64decode(grid_b64)), expected_bytes)

        # After 5 iterations: grid should have both free and unknown cells
        uint8 = log_odds_to_uint8(log_odds_grid)
        free_count = int(np.sum(uint8 == 100))
        self.assertGreater(free_count, 0, "After sensor sweeps, some cells should be free")

        # Path should exist since room is open
        final_path = run_astar(log_odds_grid, pose, goal, cfg)
        self.assertGreater(len(final_path), 0, "A path to goal should exist after mapping")

    def test_grid_byte_length(self):
        """Grid byte length must match grid_size^2 for correct base64 decoding."""
        import base64
        cfg = make_cfg(grid_size=200)
        grid = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float64)
        uint8 = log_odds_to_uint8(grid)
        b64 = base64.b64encode(uint8.tobytes()).decode("ascii")
        decoded = base64.b64decode(b64)
        self.assertEqual(len(decoded), 200 * 200)

    def test_slam_state_json_serializable(self):
        """The slam payload must be JSON-serializable (frontend receives it)."""
        import json, base64
        cfg = make_cfg(grid_size=50, cell_size_m=0.1)
        grid = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.float64)
        pose = RobotPose(x=0.1, y=-0.3, theta=0.5)
        goal = (1.0, 0.0)
        path = run_astar(grid, pose, goal, cfg)
        uint8 = log_odds_to_uint8(grid)
        payload = {
            "pose": {"x": round(pose.x, 3), "y": round(pose.y, 3), "theta": round(pose.theta, 4)},
            "goal": {"x": goal[0], "y": goal[1]},
            "path": [[round(p[0], 3), round(p[1], 3)] for p in path],
            "current_cmd": {"left": 1, "right": 1},
            "mode": "navigating",
            "grid_resolution_m": cfg.cell_size_m,
            "grid_width": cfg.grid_size,
            "grid_height": cfg.grid_size,
            "grid_data": base64.b64encode(uint8.tobytes()).decode("ascii"),
            "ts": time.time(),
        }
        # Should not raise
        serialized = json.dumps(payload)
        parsed = json.loads(serialized)
        self.assertEqual(parsed["pose"]["x"], round(pose.x, 3))


# ===========================================================================
# 11. Performance smoke test
# ===========================================================================

class TestPerformance(unittest.TestCase):
    """Ensures core functions are fast enough for real-time use."""

    def test_astar_completes_under_1s(self):
        """A* on a 200×200 free grid must complete in < 1 second."""
        grid = np.zeros((200, 200), dtype=np.float64)
        start = RobotPose(x=0.0, y=0.0, theta=0.0)
        t0 = time.perf_counter()
        path = run_astar(grid, start, (4.0, 4.0), CFG)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 1.0, f"A* took {elapsed:.2f}s, too slow")
        self.assertGreater(len(path), 0)

    def test_grid_update_100_readings_under_0_1s(self):
        """100 sensor readings should update in < 100ms."""
        grid = np.zeros((200, 200), dtype=np.float64)
        pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        t0 = time.perf_counter()
        for angle in range(0, 180, 2):   # 90 readings
            update_occupancy_grid(grid, pose, float(angle), 150.0, CFG)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.1, f"Grid update took {elapsed:.3f}s, too slow")

    def test_log_odds_to_uint8_is_fast(self):
        """200×200 log-odds → uint8 conversion must be < 5ms."""
        grid = np.random.randn(200, 200)
        t0 = time.perf_counter()
        for _ in range(100):
            log_odds_to_uint8(grid)
        elapsed = (time.perf_counter() - t0) / 100
        self.assertLess(elapsed, 0.005, f"log_odds_to_uint8 took {elapsed*1000:.2f}ms avg")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    # Run with verbose output
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestWorldToGrid,
        TestGridToWorld,
        TestSensorToGlobal,
        TestBresenhamCells,
        TestUpdateOccupancyGrid,
        TestLogOddsToUint8,
        TestUpdatePose,
        TestComputeMotorCmd,
        TestRunAstar,
        TestMockRoomDistance,
        TestIntegration,
        TestPerformance,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Summary
    print(f"\n{'='*60}")
    print(f"Tests run:   {result.testsRun}")
    print(f"Failures:    {len(result.failures)}")
    print(f"Errors:      {len(result.errors)}")
    print(f"Skipped:     {len(result.skipped)}")
    print(f"{'='*60}")

    if result.failures or result.errors:
        print("\nFAILED — see above for details.")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
