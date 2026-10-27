#!/usr/bin/env python3
import sys, tty, termios, threading, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from g1_controller.g1_config import dico_joints_inv

MODES = ["LOW", "HIGH", "RL"]
PUBLISH_HZ = 20.0
HOLD_TIMEOUT = 0.4   # secondes : key hold if more than timeout

ESC, SPACE, TAB, BACKSPACE = '\x1b', ' ', '\t', '\x7f'

HIGH_ACTIONS = {
    '1': "squat_workout", '&': "squat_workout",
    '2': "coucou",        'é': "coucou",
    '3': "stand",         '"': "stand",
    '4': "chair",         "'": "chair",
    '5': "squat",         '(': "squat",
}


class KeyboardBridge(Node):
    def __init__(self):
        super().__init__('keyboard_bridge')
        self.cmd_pub = self.create_publisher(String, '/command', 10)
        self.mode_idx = 0
        self.selected_joint = 0
        self.last_seen = {}
        self.running = True

        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)       # key by key, Ctrl+C always active

        threading.Thread(target=self.read_loop, daemon=True).start()
        self.create_timer(1.0 / PUBLISH_HZ, self.tick)
        self.get_logger().info(
            "Keyboard bridge (terminal). Garde le focus sur CE terminal. "
            "Esc/Espace = stop, Tab/Retour = mode.")

    def send_cmd(self, text):
        msg = String()
        msg.data = text
        self.cmd_pub.publish(msg)
        if "joint" not in text and "vel" not in text:
            self.get_logger().info(f"Published command: {text}")

    def held(self):
        now = time.time()
        return {k for k, t in self.last_seen.items() if now - t < HOLD_TIMEOUT}

    def read_loop(self):
        while self.running:
            k = sys.stdin.read(1)
            if not k:
                continue
            k = k.lower()
            is_new = k not in self.held()
            self.last_seen[k] = time.time()
            if is_new:
                self.on_press(k)

    def on_press(self, k):
        mode = MODES[self.mode_idx]

        if k in (ESC, SPACE):
            self.send_cmd("stop")
            return

        if k == TAB:
            self.mode_idx = (self.mode_idx + 1) % len(MODES)
            self.send_cmd(f"mode {MODES[self.mode_idx]}")
        elif k == BACKSPACE:
            self.mode_idx = (self.mode_idx - 1) % len(MODES)
            self.send_cmd(f"mode {MODES[self.mode_idx]}")
        elif mode == "HIGH":
            if k in HIGH_ACTIONS:
                self.send_cmd(HIGH_ACTIONS[k])
        elif mode == "LOW":
            if k == 'q':
                self.selected_joint = (self.selected_joint - 1) % 29
            elif k == 'd':
                self.selected_joint = (self.selected_joint + 1) % 29
            else:
                return
            self.get_logger().info(
                f"Selected joint: {dico_joints_inv[self.selected_joint]} "
                f"(ID: {self.selected_joint})")

    def tick(self):
        mode = MODES[self.mode_idx]
        h = self.held()

        if mode == "RL":
            vx   = (1.0 if 'z' in h else 0.0) - (1.0 if 's' in h else 0.0)
            vy   = (0.8 if 'q' in h else 0.0) - (0.8 if 'd' in h else 0.0)
            dyaw = (1.0 if 'a' in h else 0.0) - (1.0 if 'e' in h else 0.0)
            self.send_cmd(f"vel {vx:.2f} {vy:.2f} {dyaw:.2f}")
        elif mode == "LOW":
            val = (0.5 if 'z' in h else 0.0) - (0.5 if 's' in h else 0.0)
            if val != 0.0:
                self.send_cmd(f"joint {self.selected_joint} {val:.4f}")

    def restore_terminal(self):
        self.running = False
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def main():
    rclpy.init()
    node = KeyboardBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_terminal()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()