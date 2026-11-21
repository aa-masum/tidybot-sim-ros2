#!/usr/bin/env python3
"""Reactive obstacle avoidance using the main lidar and the low forward sensor."""
import math
import random

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def sector_min(scan, lo_deg, hi_deg):
    """Minimum valid range inside the angular sector [lo_deg, hi_deg]."""
    ranges = np.array(scan.ranges, dtype=float)
    ranges[~np.isfinite(ranges)] = scan.range_max
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
    mask = (angles >= math.radians(lo_deg)) & (angles <= math.radians(hi_deg))
    if not mask.any():
        return scan.range_max
    return float(ranges[mask].min())


class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')

        self.declare_parameter('max_speed', 0.25)
        self.declare_parameter('turn_speed', 0.8)
        self.declare_parameter('stop_dist', 0.45)
        self.declare_parameter('clear_dist', 0.8)
        self.declare_parameter('slow_dist', 1.1)

        self.max_speed = self.get_parameter('max_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value
        self.stop_dist = self.get_parameter('stop_dist').value
        self.clear_dist = self.get_parameter('clear_dist').value
        self.slow_dist = self.get_parameter('slow_dist').value

        self.scan = None
        self.low = None
        self.turn_dir = 0.0          # +1 left, -1 right, 0 not turning
        self.wander_bias = 0.0
        self.wander_until = 0.0
        self.next_wander = self.now() + random.uniform(8.0, 15.0)

        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.create_subscription(LaserScan, '/scan_low', self.on_low, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.control)
        self.get_logger().info('Obstacle avoider started')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_scan(self, msg):
        self.scan = msg

    def on_low(self, msg):
        self.low = msg

    def control(self):
        cmd = Twist()
        if self.scan is None or self.low is None:
            self.pub.publish(cmd)
            return

        front = min(sector_min(self.scan, -25, 25),
                    sector_min(self.low, -35, 35))
        left = min(sector_min(self.scan, 20, 90), sector_min(self.low, 15, 35))
        right = min(sector_min(self.scan, -90, -20), sector_min(self.low, -35, -15))

        # State 1: turning in place until the front is clear
        if self.turn_dir != 0.0:
            if front > self.clear_dist:
                self.turn_dir = 0.0
            else:
                cmd.angular.z = self.turn_dir * self.turn_speed
                self.pub.publish(cmd)
                return

        # State 2: obstacle too close, start turning toward the freer side
        if front < self.stop_dist:
            self.turn_dir = 1.0 if left >= right else -1.0
            cmd.angular.z = self.turn_dir * self.turn_speed
            self.pub.publish(cmd)
            return

        # State 3: drive forward, slow down and steer away from nearby obstacles
        speed = self.max_speed
        if front < self.slow_dist:
            ratio = (front - self.stop_dist) / (self.slow_dist - self.stop_dist)
            speed = max(0.05, self.max_speed * ratio)

        steer = 0.6 * (min(left, 1.5) - min(right, 1.5))
        steer = max(-0.6, min(0.6, steer))

        # Small random heading bias from time to time so the robot explores
        t = self.now()
        if t > self.next_wander:
            self.wander_bias = random.uniform(-0.5, 0.5)
            self.wander_until = t + 1.0
            self.next_wander = t + random.uniform(8.0, 15.0)
        if t < self.wander_until:
            steer += self.wander_bias

        cmd.linear.x = speed
        cmd.angular.z = steer
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = ObstacleAvoider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
