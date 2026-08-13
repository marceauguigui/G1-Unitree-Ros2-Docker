#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from g1_controller.g1_config import dico_joints_inv
 
# ── PS5 controller button / axis mapping ──
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
BTN_PS       = 10  # PlayStation / home button
 
# Axes mapping
AXE_STICK_L_H = 0  # left stick horizontal (lateral movement / vy)
AXE_STICK_L_V = 1  # left stick vertical (forward/backward / vx)
AXE_STICK_R_H = 3  # right stick horizontal (yaw rotation / dyaw)
AXE_STICK_R_V = 4  # right stick vertical (joint movement for LOW mode)
 
STICK_DEADZONE = 0.1
 
 
class Ps5Bridge(Node):
    def __init__(self):
        super().__init__('ps5_bridge')
        self.cmd_pub = self.create_publisher(String, '/command', 10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
 
        # ── Control mode cycling (LOW <-> HIGH <-> RL); STOP is out-of-band, PS button only ──
        self.AVAILABLE_MODES = ["LOW", "HIGH", "RL"]
        self.current_mode_idx = 0  # local guess only, see module docstring
 
        self.selected_joint = 0  # currently selected joint index, LOW mode
 
        # ── Edge-detection flags: fire once per press, not once per frame while held ──
        self.l1_pressed = False
        self.r1_pressed = False
        self.options_pressed = False
        self.share_pressed = False
 
        self.get_logger().info("PS5 bridge node started.")
 
    def joy_callback(self, msg: Joy):
        current_mode = self.AVAILABLE_MODES[self.current_mode_idx]
 
        # ── 1. Emergency stop (highest priority, always active regardless of mode) ──
        if msg.buttons[BTN_PS] == 1:
            self.send_cmd("stop")
            return
 
        # ── 2. Mode selector ──
        if msg.buttons[BTN_OPTIONS] == 1 and not self.options_pressed:
            self.current_mode_idx = (self.current_mode_idx + 1) % len(self.AVAILABLE_MODES)
            selected_mode = self.AVAILABLE_MODES[self.current_mode_idx]
            self.send_cmd(f"mode {selected_mode}")
        self.options_pressed = (msg.buttons[BTN_OPTIONS] == 1)
 
        if msg.buttons[BTN_SHARE] == 1 and not self.share_pressed:
            self.current_mode_idx = (self.current_mode_idx - 1) % len(self.AVAILABLE_MODES)
            selected_mode = self.AVAILABLE_MODES[self.current_mode_idx]
            self.send_cmd(f"mode {selected_mode}")
        self.share_pressed = (msg.buttons[BTN_SHARE] == 1)
 
        # ── 3. Poses & sequences (HIGH mode only) ──
        if current_mode == "HIGH":
            if msg.buttons[BTN_CROSS] == 1:
                self.send_cmd("squat_workout")   # sequence: squat down then back up
            elif msg.buttons[BTN_CIRCLE] == 1:
                self.send_cmd("coucou")          # sequence: waving arm
            elif msg.buttons[BTN_TRIANGLE] == 1:
                self.send_cmd("stand")           # static pose
            elif msg.buttons[BTN_SQUARE] == 1:
                self.send_cmd("chair")           # static pose
            elif msg.buttons[BTN_R2] == 1:
                self.send_cmd("squat")           # static pose (distinct from the squat_workout sequence above)
 
        # ── 4. Joint selection (L1 / R1), LOW mode only ──
        if current_mode == "LOW":
            if msg.buttons[BTN_L1] == 1 and not self.l1_pressed:
                self.selected_joint = (self.selected_joint - 1) % 29
                self.get_logger().info(f"Selected joint: {dico_joints_inv[self.selected_joint]} (ID: {self.selected_joint})")
            self.l1_pressed = (msg.buttons[BTN_L1] == 1)
 
            if msg.buttons[BTN_R1] == 1 and not self.r1_pressed:
                self.selected_joint = (self.selected_joint + 1) % 29
                self.get_logger().info(f"Selected joint: {dico_joints_inv[self.selected_joint]} (ID: {self.selected_joint})")
            self.r1_pressed = (msg.buttons[BTN_R1] == 1)
 
            # ── 5. Manual joint movement (right stick vertical) ──
            if abs(msg.axes[AXE_STICK_R_V]) > STICK_DEADZONE:
                val = msg.axes[AXE_STICK_R_V] * 0.5
                cmd = f"joint {self.selected_joint} {val:.4f}"
                self.send_cmd(cmd)
 
        # ── 6. RL mode velocity commands (left stick + right stick horizontal) ──
        if current_mode == "RL":
            raw_vx = msg.axes[AXE_STICK_L_V]    # pushing stick up = positive forward velocity
            raw_vy = msg.axes[AXE_STICK_L_H]    # lateral left/right
            raw_dyaw = msg.axes[AXE_STICK_R_H]  # rotation in place
 
            vx = raw_vx * 1.0 if abs(raw_vx) > STICK_DEADZONE else 0.0      # max linear velocity ~1.0 m/s
            vy = raw_vy * 0.8 if abs(raw_vy) > STICK_DEADZONE else 0.0      # max lateral velocity ~0.8 m/s
            dyaw = raw_dyaw * 1.0 if abs(raw_dyaw) > STICK_DEADZONE else 0.0  # max angular velocity ~1.0 rad/s
 
            cmd = f"vel {vx:.2f} {vy:.2f} {dyaw:.2f}"
            self.send_cmd(cmd)
 
    def send_cmd(self, text):
        """Publish a command string to /command and log it."""
        msg = String()
        msg.data = text
        self.cmd_pub.publish(msg)
        # Don't continuously log high-frequency "joint"/"vel" commands to avoid spamming the terminal.
        if "joint" not in text and "vel" not in text:
            self.get_logger().info(f"Published command: {text}")
 
 
def main():
    rclpy.init()
    node = Ps5Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()