#!/usr/bin/env python3
"""
G1 LiDAR Obstacle Detector — Real Robot (PointCloud2)
=====================================================
Subscribes to the 3D point cloud published by the Unitree LiDAR driver
(/utlidar/cloud by default), applies a 3D bounding-box safety filter
in front of the robot, and publishes "stop" on /command when an obstacle
is detected inside the safety zone.

Usage (default topic):
    ros2 run g1_controller g1_lidar

Usage (custom topic):
    ros2 run g1_controller g1_lidar --ros-args -p lidar_topic:=/unilidar/cloud
"""

import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


def extract_xyz(msg: PointCloud2):
    """
    Extract X, Y, Z float32 arrays from a PointCloud2 message.
    Reads field offsets from the message header — compatible with any
    point cloud layout (Unitree L1, L2, Hesai, Velodyne, etc.).
    Returns three empty arrays if the cloud contains no points.
    """
    field_offsets = {f.name: f.offset for f in msg.fields}
    x_off = field_offsets.get('x', 0)
    y_off = field_offsets.get('y', 4)
    z_off = field_offsets.get('z', 8)

    n_points = msg.width * msg.height
    point_step = msg.point_step

    if n_points == 0 or point_step == 0:
        return np.empty(0, dtype=np.float32), \
               np.empty(0, dtype=np.float32), \
               np.empty(0, dtype=np.float32)

    # Reshape raw bytes to (n_points, point_step) for vectorised slicing
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n_points, point_step)

    x = np.frombuffer(raw[:, x_off : x_off + 4].copy().tobytes(), dtype=np.float32)
    y = np.frombuffer(raw[:, y_off : y_off + 4].copy().tobytes(), dtype=np.float32)
    z = np.frombuffer(raw[:, z_off : z_off + 4].copy().tobytes(), dtype=np.float32)

    return x, y, z


class G1LidarDetector(Node):
    def __init__(self):
        super().__init__('g1_lidar_detector')

        # ── SAFETY ZONE PARAMETERS (all in metres) ──
        # Forward depth of the danger zone
        self.declare_parameter('safety_distance', 0.6)
        # Half-width of the zone (centred on the robot's forward axis)
        self.declare_parameter('safety_width', 0.4)
        # Vertical range — avoids floor reflections and overhead obstacles
        self.declare_parameter('safety_z_min', 0.05)
        self.declare_parameter('safety_z_max', 1.8)
        # Minimum number of points inside the box to confirm an obstacle
        # (avoids false positives from sensor noise)
        self.declare_parameter('min_points', 5)
        # Topic published by the Unitree LiDAR driver
        self.declare_parameter('lidar_topic', '/utlidar/cloud')

        lidar_topic = self.get_parameter('lidar_topic').value

        # ── SUBSCRIPTIONS ──
        self.cloud_sub = self.create_subscription(
            PointCloud2, lidar_topic, self.cloud_callback, 10)

        # ── PUBLISHERS ──
        self.cmd_pub = self.create_publisher(String, '/command', 10)

        self.obstacle_detected = False
        self.get_logger().info(
            f"LiDAR obstacle detector ready — listening on '{lidar_topic}'")

    def cloud_callback(self, msg: PointCloud2):
        safety_dist = self.get_parameter('safety_distance').value
        safety_width = self.get_parameter('safety_width').value
        z_min        = self.get_parameter('safety_z_min').value
        z_max        = self.get_parameter('safety_z_max').value
        min_pts      = self.get_parameter('min_points').value

        x, y, z = extract_xyz(msg)

        # Discard NaN / Inf values (produced at the sensor's max range)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[valid], y[valid], z[valid]

        if len(x) == 0:
            return

        # ── 3-D BOUNDING BOX FILTER ──
        # Convention: X = forward, Y = lateral, Z = vertical
        # (matches the LiDAR frame on the Unitree G1)
        in_box = (
            (x > 0.05)           & (x < safety_dist) &  # forward
            (np.abs(y) < safety_width)                &  # lateral
            (z > z_min)          & (z < z_max)           # vertical
        )
        obstacle_count = int(np.sum(in_box))

        if obstacle_count >= min_pts:
            closest = float(np.min(x[in_box]))
            if not self.obstacle_detected:
                self.obstacle_detected = True
                self.get_logger().error(
                    f"!!! OBSTACLE DETECTED — {obstacle_count} pts, "
                    f"closest at {closest:.2f} m !!! Sending STOP.")
                stop_msg = String()
                stop_msg.data = "stop"
                self.cmd_pub.publish(stop_msg)
        else:
            if self.obstacle_detected:
                self.obstacle_detected = False
                self.get_logger().info("Frontal zone clear.")


def main(args=None):
    rclpy.init(args=args)
    node = G1LidarDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping LiDAR detector.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()