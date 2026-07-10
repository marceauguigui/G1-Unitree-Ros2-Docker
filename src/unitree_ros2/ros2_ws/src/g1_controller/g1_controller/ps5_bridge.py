#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from g1_controller.g1_config import dico_joints_inv

# ── PS5 CONTROLLER BUTTON / AXIS MAPPING ──
# Index values for the DualSense controller
BTN_CROSS    = 0
BTN_CIRCLE   = 1
BTN_TRIANGLE = 2
BTN_SQUARE   = 3
BTN_L1       = 4
BTN_R1       = 5
BTN_L2       = 6
BTN_R2       = 7
BTN_SHARE    = 8
BTN_OPTIONS  = 9
BTN_PS       = 10 # PlayStation / Home button

AXE_STICK_R_V = 4 # Right stick vertical axis (push up = positive)

class Ps5Bridge(Node):
    def __init__(self):
        super().__init__('ps5_bridge')
        self.cmd_pub = self.create_publisher(String, '/command', 10) # Publish string commands to the controller
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10) # Subscribe to raw joystick input
        
        self.selected_joint = 0 # Currently selected joint index for LOW mode
        self.l1_pressed = False # Edge-detection flags: track previous button state to fire once per press
        self.r1_pressed = False
        
        self.get_logger().info("PS5 bridge node started")

    def joy_callback(self, msg: Joy):
        # ── 1. MODES & POSES (Boutons simples à constantes) ──
        if msg.buttons[BTN_PS] == 1: 
            self.send_cmd("stop")               # Emergency stop (highest priority)
        elif msg.buttons[BTN_OPTIONS] == 1:
            self.send_cmd("mode HIGH")          # Switch to HIGH mode
        elif msg.buttons[BTN_SHARE] == 1:
            self.send_cmd("mode LOW")           # Switch to LOW mode
        elif msg.buttons[BTN_CROSS] == 1:
            self.send_cmd("squat_workout")      # IF HIGH launch squat sequence
        elif msg.buttons[BTN_CIRCLE] == 1:
            self.send_cmd("coucou")             # IF HIGH launch wave sequence
        elif msg.buttons[BTN_TRIANGLE] == 1:
            self.send_cmd("stand")              # IF HIGH standing pose
        elif msg.buttons[BTN_SQUARE] == 1:
            self.send_cmd("chair")              # IF HIGH chair pose

        # ── 2. JOINT SELECTION (L1 / R1) ──
        # Cycle through all 29 joints
        if msg.buttons[BTN_L1] == 1 and not self.l1_pressed: 
            self.selected_joint = (self.selected_joint - 1) % 29 
            self.get_logger().info(f"Selected joint : {dico_joints_inv[self.selected_joint]} (ID: {self.selected_joint})")
        self.l1_pressed = (msg.buttons[BTN_L1] == 1)

        if msg.buttons[BTN_R1] == 1 and not self.r1_pressed: 
            self.selected_joint = (self.selected_joint + 1) % 29 
            self.get_logger().info(f"Selected joint : {dico_joints_inv[self.selected_joint]} (ID: {self.selected_joint})")
        self.r1_pressed = (msg.buttons[BTN_R1] == 1)

        # ── 3. JOINT MOVEMENT (Right Joystick Vertical) ──
        # Apply a deadzone of ±0.1 to avoid drift from stick centre
        val = msg.axes[AXE_STICK_R_V] * 1.0
        
        if abs(msg.axes[AXE_STICK_R_V]) > 0.1:
            # Send the numeric joint ID
            cmd = f"joint {self.selected_joint} {val:.2f}"
            self.send_cmd(cmd)

    def send_cmd(self, text): # Publish a command string to /command and log it
        msg = String()
        msg.data = text
        self.cmd_pub.publish(msg)
        if "joint" not in text:
            self.get_logger().info(f"cmd : {text}")

def main():
    rclpy.init()
    node = Ps5Bridge()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()