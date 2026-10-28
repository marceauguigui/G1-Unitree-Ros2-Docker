#!/usr/bin/env python3
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from g1_controller.g1_config import dico_joints_inv

# ════════════════════════════════════════════════════════════
#  CONFIGURATION: everything adjustable is here
# ════════════════════════════════════════════════════════════

# ── General ──
PUBLISH_HZ    = 20.0                  # publish rate for vel / joint commands
MODES         = ["LOW", "HIGH", "RL"] # order in which modes cycle
START_MODE    = "LOW"                 # mode at startup (local guess, the controller may differ)
NUM_JOINTS    = 29                    # number of selectable joints in LOW mode
COMMAND_TOPIC = "/command"

# ── Speeds / amplitudes ──
SPEED_VX     = 1.0    # m/s    forward / backward (RL)
SPEED_VY     = 0.8    # m/s    lateral (RL)
SPEED_DYAW   = 1.0    # rad/s  rotation (RL)
JOINT_STEP   = 0.5    # value sent per message in LOW mode (same as stick fully pushed)

# ── Window ──
WINDOW_SIZE   = (420, 140)
WINDOW_TITLE  = "G1 keyboard control"
FONT_SIZE     = 28
COLOR_BG      = (25, 25, 30)
COLOR_TEXT    = (230, 230, 230)

# ── Keys (pygame names: "a", "z", "space", "escape", "tab", "backspace",
#    "up", "left", "return", "left shift", ...). Several keys allowed per list. ──
KEYS = {
    # Global
    "stop":         ["escape", "space"],
    "mode_next":    ["tab"],
    "mode_prev":    ["backspace"],

    # RL: velocities
    "rl_forward":   ["z"],
    "rl_backward":  ["s"],
    "rl_left":      ["q"],
    "rl_right":     ["d"],
    "rl_yaw_left":  ["a"],    # positive yaw
    "rl_yaw_right": ["e"],    # negative yaw

    # LOW: joints
    "low_inc":        ["z"],  # increment (+JOINT_STEP)
    "low_dec":        ["s"],  # decrement (-JOINT_STEP)
    "low_joint_prev": ["q"],
    "low_joint_next": ["d"],
}

# ── HIGH: typed character -> command sent.
#    AZERTY symbols (& é " ' ( ) are included so you don't need to press Shift. ──
HIGH_ACTIONS = {
    '1': "stand",         '&': "stand",
    '2': "coucou",        'é': "coucou",
    '3': "squat_workout", '"': "squat_workout",
    '4': "chair",         "'": "chair",
    '5': "squat",         '(': "squat",
}

# ════════════════════════════════════════════════════════════
#  END OF CONFIGURATION
# ════════════════════════════════════════════════════════════


def resolve_keys(names):
    """Convert a list of key names into pygame key codes."""
    codes = set()
    for n in names:
        try:
            codes.add(pygame.key.key_code(n))
        except ValueError:
            raise ValueError(f"Unknown key name in KEYS: '{n}'")
    return codes


class KeyboardBridge(Node):
    def __init__(self):
        super().__init__('keyboard_bridge')
        self.cmd_pub = self.create_publisher(String, COMMAND_TOPIC, 10)
        self.mode_idx = MODES.index(START_MODE)
        self.selected_joint = 0
        self.held = set()

        pygame.init()
        self.keys = {action: resolve_keys(names) for action, names in KEYS.items()}
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(WINDOW_TITLE)
        self.font = pygame.font.SysFont(None, FONT_SIZE)

        self.create_timer(1.0 / PUBLISH_HZ, self.tick)
        self.get_logger().info("Keyboard bridge (pygame). To control, keep focus on the window.")

    # ── Helpers ──
    def is_down(self, action):
        """True if at least one key of the action is held."""
        return bool(self.keys[action] & self.held)

    def is_key(self, action, key):
        return key in self.keys[action]

    def send_cmd(self, text):
        msg = String()
        msg.data = text
        self.cmd_pub.publish(msg)
        # Don't log high-frequency "joint"/"vel" commands to avoid spamming the terminal.
        if "joint" not in text and "vel" not in text:
            self.get_logger().info(f"Published command: {text}")

    def change_mode(self, step):
        self.mode_idx = (self.mode_idx + step) % len(MODES)
        self.send_cmd(f"mode {MODES[self.mode_idx]}")

    # ── Events ──
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.send_cmd("stop")
                raise KeyboardInterrupt
            elif ev.type == pygame.WINDOWFOCUSLOST:
                self.held.clear()             # focus lost: no key is held anymore
            elif ev.type == pygame.KEYUP:
                self.held.discard(ev.key)
            elif ev.type == pygame.KEYDOWN:
                self.held.add(ev.key)
                self.on_press(ev)             # no auto-repeat in pygame by default

    def on_press(self, ev):
        mode = MODES[self.mode_idx]

        if self.is_key("stop", ev.key):
            self.send_cmd("stop")
        elif self.is_key("mode_next", ev.key):
            self.change_mode(+1)
        elif self.is_key("mode_prev", ev.key):
            self.change_mode(-1)
        elif mode == "HIGH":
            if ev.unicode in HIGH_ACTIONS:
                self.send_cmd(HIGH_ACTIONS[ev.unicode])
        elif mode == "LOW":
            if self.is_key("low_joint_prev", ev.key):
                self.selected_joint = (self.selected_joint - 1) % NUM_JOINTS
            elif self.is_key("low_joint_next", ev.key):
                self.selected_joint = (self.selected_joint + 1) % NUM_JOINTS
            else:
                return
            self.get_logger().info(
                f"Selected joint: {dico_joints_inv[self.selected_joint]} "
                f"(ID: {self.selected_joint})")

    # ── Display ──
    def draw(self):
        self.screen.fill(COLOR_BG)
        lines = [f"Mode: {MODES[self.mode_idx]}"]
        if MODES[self.mode_idx] == "LOW":
            lines.append(f"Joint: {dico_joints_inv[self.selected_joint]} ({self.selected_joint})")
        for i, t in enumerate(lines):
            self.screen.blit(self.font.render(t, True, COLOR_TEXT), (15, 20 + 35 * i))
        pygame.display.flip()

    # ── Continuous commands ──
    def tick(self):
        self.handle_events()
        mode = MODES[self.mode_idx]

        if mode == "RL":
            vx   = SPEED_VX   * (self.is_down("rl_forward")  - self.is_down("rl_backward"))
            vy   = SPEED_VY   * (self.is_down("rl_left")     - self.is_down("rl_right"))
            dyaw = SPEED_DYAW * (self.is_down("rl_yaw_left") - self.is_down("rl_yaw_right"))
            self.send_cmd(f"vel {vx:.2f} {vy:.2f} {dyaw:.2f}")
        elif mode == "LOW":
            val = JOINT_STEP * (self.is_down("low_inc") - self.is_down("low_dec"))
            if val != 0.0:
                self.send_cmd(f"joint {self.selected_joint} {val:.4f}")
        self.draw()


def main():
    rclpy.init()
    node = KeyboardBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()