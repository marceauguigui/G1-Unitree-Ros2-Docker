"""RL teleoperation (RSL-RL + mjlab), MODE-based control (LOW/HIGH/RL/STOP) via
KEYBOARD (F1-F3, 4-8) AND PS5 controller (mirrors ps5_bridge.py's logic, but
read directly through pygame — no ROS2/rclpy needed).

Controller mapping (mirrors ps5_bridge.py):
  PS            -> immediate STOP (highest priority)
  OPTIONS       -> next mode (LOW -> HIGH -> RL -> LOW ...)
  SHARE         -> previous mode
  [HIGH mode]   CROSS=squat_workout, CIRCLE=coucou, TRIANGLE=stand,
                SQUARE=chair, R2=stand (⚠️ "stand_isaac" no longer exists, redirected)
  [LOW mode]    L1/R1=previous/next joint, right stick vertical=joint nudge
  [RL mode]     left stick=vx/vy, right stick horizontal=yaw"""

import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

try:
    import pygame
except ImportError:
    print("[ERROR] Pygame is required for the controller. Install it with: pip install pygame")
    sys.exit(1)

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    print("[ERROR] Pynput is required for global keyboard input. Install it with: pip install pynput")
    sys.exit(1)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

CALIBRATE = os.environ.get("TELEOP_CALIBRATE") == "1"

# ── PS5 button/axis mapping, taken from ps5_bridge.py ──
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
BTN_PS       = 10

AXE_STICK_L_H = 0  # lateral (vy)
AXE_STICK_L_V = 1  # forward/backward (vx)
AXE_STICK_R_H = 3  # yaw (RL mode)
AXE_STICK_R_V = 4  # joint nudge (LOW mode)

AVAILABLE_MODES = ["LOW", "HIGH", "RL"]  # same list/order as ps5_bridge.py (STOP = out-of-cycle, PS button only)


# ═══════════════════════════════════════════════════════════════════════════
# Elements copied directly from g1_config.py
# ═══════════════════════════════════════════════════════════════════════════

nbr_joints = 29

dico_joints = {
    "hip_pitch_l": 0, "hip_roll_l": 1, "hip_yaw_l": 2, "knee_pitch_l": 3, "ankle_pitch_l": 4, "ankle_roll_l": 5,
    "hip_pitch_r": 6, "hip_roll_r": 7, "hip_yaw_r": 8, "knee_pitch_r": 9, "ankle_pitch_r": 10, "ankle_roll_r": 11,
    "waist_yaw": 12, "waist_roll": 13, "waist_pitch": 14,
    "should_pitch_l": 15, "should_roll_l": 16, "should_yaw_l": 17, "elbow_pitch_l": 18,
    "wrist_roll_l": 19, "wrist_pitch_l": 20, "wrist_yaw_l": 21,
    "should_pitch_r": 22, "should_roll_r": 23, "should_yaw_r": 24, "elbow_pitch_r": 25,
    "wrist_roll_r": 26, "wrist_pitch_r": 27, "wrist_yaw_r": 28,
}
dico_joints_inv = {v: k for k, v in dico_joints.items()}

JOINTS_MIN_MAX = [
    (-2.5307, 2.8798), (-0.5236, 2.9671), (-2.7576, 2.7576), (-0.087267, 2.8798), (-0.87267, 0.5236), (-0.2618, 0.2618),
    (-2.5307, 2.8798), (-2.9671, 0.5236), (-2.7576, 2.7576), (-0.087267, 2.8798), (-0.87267, 0.5236), (-0.2618, 0.2618),
    (-2.618, 2.618), (-0.52, 0.52), (-0.52, 0.52),
    (-3.0892, 2.6704), (-1.5882, 2.2515), (-2.618, 2.618), (-1.0472, 2.0944), (-1.9722, 1.9722), (-1.6144, 1.6144), (-1.6144, 1.6144),
    (-3.0892, 2.6704), (-2.2515, 1.5882), (-2.618, 2.618), (-1.0472, 2.0944), (-1.9722, 1.9722), (-1.6144, 1.6144), (-1.6144, 1.6144),
]

_positions_deg = {
    "stand": np.zeros(nbr_joints),
    "chair": np.array([-90, 0, 0, 90, 0, 0, -90, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "squat": np.array([-70, 0, -20, 70, -25, 0, -70, 0, -20, 70, -25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "coucou1": np.array([0]*22 + [-90, 0, 0, 0, -90, 0, 0]),
    "coucou2": np.array([0]*22 + [-90, 0, 0, 0, -90, -30, 0]),
    "coucou3": np.array([0]*22 + [-90, 0, 0, 0, -90, 30, 0]),
}
positions_rad = {k: np.array([math.radians(a) for a in v], dtype=np.float32) for k, v in _positions_deg.items()}

sequences = {
    "coucou": [("coucou1", 2.0), ("coucou2", 1.0), ("coucou3", 1.0)],
    "squat_workout": [("squat", 3.0), ("stand", 1.0)],
}

DEFAULT_POS = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0,
    -0.312, 0, 0, 0.669, -0.363, 0,
    0, 0, 0,
    0.2, 0.2, 0, 0.6, 0, 0, 0,
    0.2, -0.2, 0, 0.6, 0, 0, 0,
], dtype=np.float32)

ACTION_SCALE = np.array([
    0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
    0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
    0.55, 0.44, 0.44,
    0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
    0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
], dtype=np.float32)


def interpolate_cubic_vectorized(start_array, end_array, time, duration):
    t = np.clip(time / duration, 0.0, 1.0)
    phase = 3 * (t**2) - 2 * (t**3)
    return (1.0 - phase) * start_array + phase * end_array


def play_sequence(sequence, step_index, step_time, current_position, dt_val, forced_duration=None):
    if sequence is None or step_index >= len(sequence):
        return current_position, step_index, step_time, current_position, True

    pose_name, duration = sequence[step_index]
    if forced_duration is not None:
        duration = forced_duration
    is_finished = False

    if pose_name in positions_rad:
        target_pose = positions_rad[pose_name]
        step_time += dt_val
        target_list = interpolate_cubic_vectorized(current_position, target_pose, step_time, duration)
        if step_time >= duration:
            step_index += 1
            step_time = 0.0
            current_position = np.copy(target_list)
            if step_index >= len(sequence):
                is_finished = True
    else:
        is_finished = True
        target_list = current_position

    return target_list, step_index, step_time, current_position, is_finished


# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TeleopConfig:
    checkpoint_file: str | None = None
    wandb_run_path: str | None = None
    num_envs: int | None = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False


class TeleopInputReader:
    """Controller (full PS5 mapping, mirrored from ps5_bridge.py) + global
    keyboard fallback (F1-F3, 4-8) if no controller is detected."""

    _POSE_KEYS = {"4": "stand", "5": "squat", "6": "chair", "7": "coucou", "8": "squat_workout"}

    def __init__(self, deadzone=0.1):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.joystick.init()
        self.joystick = None

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[INFO] Controller detected: {self.joystick.get_name()} "
                  f"({self.joystick.get_numbuttons()} buttons, {self.joystick.get_numaxes()} axes)")
        else:
            print("[INFO] No controller detected. Keyboard active (F1-F3, 4-8).")

        self.deadzone = deadzone

        # Keyboard
        self._pressed_keys: set = set()
        self._prev_pressed_keys: set = set()
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_key_press, on_release=self._on_key_release
        )
        self._listener.daemon = True
        self._listener.start()

        # Controller: state for edge-detection + mode cycling + selected joint (LOW mode)
        self._mode_cycle_idx = AVAILABLE_MODES.index("HIGH")
        self.selected_joint = 0
        self._prev_buttons: list[int] = []

    def _on_key_press(self, key):
        self._pressed_keys.add(key)

    def _on_key_release(self, key):
        self._pressed_keys.discard(key)

    def _key_down(self, char_or_key) -> bool:
        if isinstance(char_or_key, str):
            try:
                return pynput_keyboard.KeyCode.from_char(char_or_key) in self._pressed_keys
            except Exception:
                return False
        return char_or_key in self._pressed_keys

    def get_commands(self):
        """Velocity for RL mode: controller (sticks) takes priority, keyboard otherwise."""
        pygame.event.pump()
        cmd_x, cmd_y, cmd_yaw = 0.0, 0.0, 0.0

        if self.joystick is not None:
            v_y = self.joystick.get_axis(AXE_STICK_L_H)
            v_x = self.joystick.get_axis(AXE_STICK_L_V)
            num_axes = self.joystick.get_numaxes()
            v_yaw = self.joystick.get_axis(AXE_STICK_R_H) if num_axes > AXE_STICK_R_H else 0.0
            if abs(v_x) > self.deadzone: cmd_x = -v_x
            if abs(v_y) > self.deadzone: cmd_y = -v_y * 0.8  # same 0.8 factor as ps5_bridge.py
            if abs(v_yaw) > self.deadzone: cmd_yaw = -v_yaw

        if cmd_x == 0.0 and cmd_y == 0.0 and cmd_yaw == 0.0:
            speed = 1.0
            Key = pynput_keyboard.Key
            if self._key_down(Key.up):    cmd_x = speed
            if self._key_down(Key.down):  cmd_x = -speed
            if self._key_down(Key.left):  cmd_y = speed
            if self._key_down(Key.right): cmd_y = -speed
            if self._key_down("1"): cmd_yaw = speed
            if self._key_down("3"): cmd_yaw = -speed

        return cmd_x, cmd_y, cmd_yaw

    def poll_mode_and_pose(self):
        """Keyboard: F1/F2/F3 (mode) + 4-8 (poses/sequences), rising edge."""
        current = set(self._pressed_keys)
        new_keys = current - self._prev_pressed_keys
        self._prev_pressed_keys = current

        mode, pose = None, None
        Key = pynput_keyboard.Key
        for k in new_keys:
            if k == Key.f1:
                mode = "HIGH"
            elif k == Key.f2:
                mode = "RL"
            elif k == Key.f3:
                mode = "STOP"
            elif isinstance(k, pynput_keyboard.KeyCode) and k.char in self._POSE_KEYS:
                pose = self._POSE_KEYS[k.char]
        return mode, pose

    def poll_gamepad_events(self):
        """Controller: mirrors ps5_bridge.py. Returns
        (stop, mode, pose, joint_select_delta, joint_move_val)."""
        stop = False
        mode = None
        pose = None
        joint_select_delta = 0
        joint_move_val = 0.0

        if self.joystick is None:
            return stop, mode, pose, joint_select_delta, joint_move_val

        pygame.event.pump()
        nbtn = self.joystick.get_numbuttons()
        buttons = [self.joystick.get_button(i) for i in range(nbtn)]
        if not self._prev_buttons or len(self._prev_buttons) != nbtn:
            self._prev_buttons = [0] * nbtn

        def pressed(idx):
            return idx < nbtn and buttons[idx] == 1

        def just_pressed(idx):
            return idx < nbtn and buttons[idx] == 1 and self._prev_buttons[idx] == 0

        if CALIBRATE:
            for i in range(nbtn):
                if buttons[i] == 1 and self._prev_buttons[i] == 0:
                    print(f"[CALIBRATE] Button pressed: index {i}")

        current_mode = AVAILABLE_MODES[self._mode_cycle_idx]

        # 1. STOP (highest priority)
        if pressed(BTN_PS):
            stop = True

        # 2. Mode cycling
        if just_pressed(BTN_OPTIONS):
            self._mode_cycle_idx = (self._mode_cycle_idx + 1) % len(AVAILABLE_MODES)
            mode = AVAILABLE_MODES[self._mode_cycle_idx]
        if just_pressed(BTN_SHARE):
            self._mode_cycle_idx = (self._mode_cycle_idx - 1) % len(AVAILABLE_MODES)
            mode = AVAILABLE_MODES[self._mode_cycle_idx]

        # 3. Poses/sequences (HIGH mode)
        if current_mode == "HIGH":
            if pressed(BTN_CROSS):
                pose = "squat_workout"
            elif pressed(BTN_CIRCLE):
                pose = "coucou"
            elif pressed(BTN_TRIANGLE):
                pose = "stand"
            elif pressed(BTN_SQUARE):
                pose = "chair"

        # 4. Joint selection + nudge (LOW mode)
        if current_mode == "LOW":
            if just_pressed(BTN_L1):
                joint_select_delta = -1
                self.selected_joint = (self.selected_joint - 1) % nbr_joints
                print(f"[INFO] Selected joint: {dico_joints_inv[self.selected_joint]} (ID {self.selected_joint})")
            if just_pressed(BTN_R1):
                joint_select_delta = 1
                self.selected_joint = (self.selected_joint + 1) % nbr_joints
                print(f"[INFO] Selected joint: {dico_joints_inv[self.selected_joint]} (ID {self.selected_joint})")

            num_axes = self.joystick.get_numaxes()
            if num_axes > AXE_STICK_R_V:
                axis_val = self.joystick.get_axis(AXE_STICK_R_V)
                if abs(axis_val) > 0.1:
                    joint_move_val = axis_val * 0.5 * 0.02  # same scale as ps5_bridge.py + g1_ros2_controller

        self._prev_buttons = buttons
        return stop, mode, pose, joint_select_delta, joint_move_val

    def close(self):
        self._listener.stop()


def verify_joint_order(env) -> bool:
    candidates = [
        lambda: env.unwrapped.scene["robot"].joint_names,
        lambda: env.unwrapped.scene.entities["robot"].joint_names,
        lambda: env.unwrapped.action_manager.get_term("joint_pos").joint_names,
    ]
    names = None
    for fn in candidates:
        try:
            names = list(fn())
            break
        except Exception:
            continue

    if names is None:
        print("\n[⚠️  WARNING] Could not automatically retrieve the joint order "
            "exposed by the mjlab environment. LOW/HIGH/STOP modes assume "
            "an order identical to dico_joints (SDK order) — verify manually. "
            "RL mode is not affected by this uncertainty.\n")
        return False

    expected = [dico_joints_inv[i] for i in range(nbr_joints)]
    print("\n[JOINT ORDER VERIFICATION]")
    print(f"  Expected (SDK)                  : {expected}")
    print(f"  Found    (mjlab env, {len(names)} joints) : {names}")
    if len(names) != nbr_joints:
        print(f"[⚠️  WARNING] Different number of joints ({len(names)} vs {nbr_joints})!")
        return False
    print("  -> Compare these two lists yourself before trusting the\n"
          "     HIGH/LOW/STOP modes (RL mode is not affected).\n")
    return True


class HybridPolicyWrapper:
    def __init__(self, base_policy, env, gamepad: TeleopInputReader, device: str):
        self.base_policy = base_policy
        self.env = env
        self.gamepad = gamepad
        self.device = device

        self.default_pos = torch.tensor(DEFAULT_POS, dtype=torch.float32, device=device)
        self.action_scale = torch.tensor(ACTION_SCALE, dtype=torch.float32, device=device)
        self.mins_np = np.array([lim[0] for lim in JOINTS_MIN_MAX], dtype=np.float32)
        self.maxs_np = np.array([lim[1] for lim in JOINTS_MIN_MAX], dtype=np.float32)

        self.control_mode = "HIGH"
        self._last_mode = None

        self._high_time = 0.0
        self._high_duration = 0.5
        self._high_start_pos = None
        self._high_current_target_pose_name = None
        self._active_sequence = None
        self._active_sequence_name = None
        self._sequence_step = 0
        self._sequence_time = 0.0

        self._low_target_np = None

        if hasattr(self.env.unwrapped, "command_manager"):
            self.env.unwrapped.command_manager.compute = lambda *args, **kwargs: None
            print("[INFO] Automatic command manager disabled (Manual Mode ON).")

    def _read_joint_pos_np(self) -> np.ndarray:
        candidates = [
            lambda: self.env.unwrapped.scene["robot"].data.joint_pos[0],
            lambda: self.env.unwrapped.scene.entities["robot"].data.joint_pos[0],
        ]
        for fn in candidates:
            try:
                t = fn()
                return t.detach().cpu().numpy().astype(np.float32)
            except Exception:
                continue
        raise RuntimeError(
            "Could not read joint_pos from the mjlab environment. Inspect "
            "`env.unwrapped.scene` yourself and adjust _read_joint_pos_np()."
        )

    def _target_to_action(self, target_np: np.ndarray) -> torch.Tensor:
        target_np = np.clip(target_np, self.mins_np, self.maxs_np)
        target_t = torch.tensor(target_np, dtype=torch.float32, device=self.device)
        action = (target_t - self.default_pos) / self.action_scale
        return action.unsqueeze(0)

    def _set_mode(self, new_mode: str):
        if new_mode != self.control_mode:
            self.control_mode = new_mode
            self._active_sequence = None
            self._high_current_target_pose_name = None
            print(f"\n[INFO] Mode -> {self.control_mode}")

    def __call__(self, obs):
        # ── Keyboard (always active, fallback if no controller) ──
        kb_mode, kb_pose = self.gamepad.poll_mode_and_pose()

        # ── Controller (takes priority if present) ──
        gp_stop, gp_mode, gp_pose, _joint_sel_delta, joint_move_val = self.gamepad.poll_gamepad_events()

        q_np = self._read_joint_pos_np()

        if self.control_mode != self._last_mode:
            self._high_start_pos = q_np.copy()
            self._high_time = 0.0
            self._low_target_np = q_np.copy()
            self._last_mode = self.control_mode

        # STOP has absolute priority (keyboard F3 or PS button)
        if gp_stop or kb_mode == "STOP":
            self._set_mode("STOP")
        elif gp_mode is not None:
            self._set_mode(gp_mode)
        elif kb_mode is not None:
            self._set_mode(kb_mode)

        pose_trigger = gp_pose if gp_pose is not None else kb_pose
        if pose_trigger is not None and self.control_mode == "HIGH":
            if pose_trigger in sequences:
                self._active_sequence_name = pose_trigger
                self._active_sequence = sequences[pose_trigger]
                self._sequence_step = 0
                self._sequence_time = 0.0
                self._high_current_target_pose_name = None
            elif pose_trigger in positions_rad:
                self._high_current_target_pose_name = pose_trigger
                self._high_start_pos = q_np.copy()
                self._high_time = 0.0
                self._active_sequence = None

        if self.control_mode == "STOP":
            return self._target_to_action(q_np)

        if self.control_mode == "LOW":
            if abs(joint_move_val) > 1e-9:
                idx = self.gamepad.selected_joint
                self._low_target_np[idx] = float(
                    np.clip(self._low_target_np[idx] + joint_move_val, self.mins_np[idx], self.maxs_np[idx])
                )
            return self._target_to_action(self._low_target_np)

        if self.control_mode == "HIGH":
            if self._active_sequence is not None:
                state_name = self._active_sequence[self._sequence_step][0]
                max_dist = float(np.max(np.abs(positions_rad[state_name] - q_np)))
                dyn_duration = max(max_dist * 2, 0.5)
                (target_np, self._sequence_step, self._sequence_time, _, is_finished) = play_sequence(
                    self._active_sequence, self._sequence_step, self._sequence_time, q_np, 1.0 / 50.0, dyn_duration
                )
                if is_finished:
                    self._active_sequence = None
                    self._active_sequence_name = None
                return self._target_to_action(target_np)

            if self._high_current_target_pose_name is not None:
                pose_name = self._high_current_target_pose_name
                self._high_time += 1.0 / 50.0
                if self._high_duration <= 0:
                    self._high_duration = 0.5
                target_np = interpolate_cubic_vectorized(
                    self._high_start_pos, positions_rad[pose_name], self._high_time, self._high_duration
                )
                return self._target_to_action(target_np)

            return self._target_to_action(self._high_start_pos)

        # ── RL: original path, unchanged ──
        cmd_x, cmd_y, cmd_yaw = self.gamepad.get_commands()

        if hasattr(self.env.unwrapped, "command_manager"):
            cmd_name = "twist" if "twist" in self.env.unwrapped.command_manager._terms else None
            if not cmd_name and "motion" in self.env.unwrapped.command_manager._terms:
                cmd_name = "motion"
            if cmd_name:
                term = self.env.unwrapped.command_manager.get_term(cmd_name)
                term.command[:, 0] = cmd_x * 1.0
                term.command[:, 1] = cmd_y * 1.0
                term.command[:, 2] = cmd_yaw * 1.0
                if hasattr(term, "vel_command_b"):
                    term.vel_command_b[:, 0] = cmd_x * 1.0
                    term.vel_command_b[:, 1] = cmd_y * 1.0
                    term.vel_command_b[:, 2] = cmd_yaw * 1.0

        if abs(cmd_x) > 0.01 or abs(cmd_y) > 0.01 or abs(cmd_yaw) > 0.01:
            print(f"🕹️ RL | Forward: {cmd_x:.2f} | Lateral: {cmd_y:.2f} | Rotation: {cmd_yaw:.2f}      ", end="\r")

        # o = obs["actor"]
        # print(f"[mjlab] base_lin_vel={o[0,0:3].tolist()}")
        # print(f"[mjlab] ang_vel={o[0,3:6].tolist()}")
        # print(f"[mjlab] grav={o[0,6:9].tolist()}")
        # print(f"[mjlab] joint_pos_rel[:5]={o[0,9:14].tolist()}")
        # print(f"[mjlab] joint_vel_rel[:5]={o[0,38:43].tolist()}")
        # print(f"[mjlab] command={o[0,96:99].tolist()}")
        # self._debug_done = True

        return self.base_policy(obs)


def run_teleop(task_id: str, cfg: TeleopConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    if hasattr(env_cfg, "sim"):
        env_cfg.sim.nconmax = 1000
        env_cfg.sim.njmax = 4000

    if cfg.no_terminations:
        env_cfg.terminations = {}

    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()

    if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        print(f"[INFO]: Manual load: {resume_path.name}")
    elif cfg.wandb_run_path is not None:
        resume_path, _ = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path))
        print(f"[INFO]: W&B load: {resume_path.name}")
    else:
        print("[INFO]: No checkpoint specified. Automatically searching for the latest model...")
        if not log_root_path.exists():
            raise ValueError(f"Root folder {log_root_path} does not exist. Have you trained the robot?")
        run_dirs = sorted([d for d in log_root_path.iterdir() if d.is_dir()], key=os.path.getmtime)
        if not run_dirs:
            raise ValueError(f"No training run found in {log_root_path}")
        latest_run = run_dirs[-1]
        checkpoints = sorted(latest_run.glob("model_*.pt"), key=os.path.getmtime)
        if not checkpoints:
            raise ValueError(f"No model_*.pt file found in {latest_run}")
        resume_path = checkpoints[-1]
        print(f"[INFO]: ✅ Loading latest checkpoint: {resume_path}")

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)

    base_policy = runner.get_inference_policy(device=device)

    verify_joint_order(env)

    print("\n" + "=" * 55)
    print("[INFO] G1 CONTROLS:")
    print("  Controller: PS=STOP | OPTIONS/SHARE=cycle mode (LOW<->HIGH<->RL)")
    print("    HIGH : CROSS=squat_workout CIRCLE=coucou TRIANGLE=stand SQUARE=chair R2=stand")
    print("    LOW  : L1/R1=prev/next joint, right stick vertical=nudge")
    print("    RL   : left stick=vx/vy, right stick horizontal=yaw")
    print("  Keyboard (fallback): F1=HIGH F2=RL F3=STOP | 4-8=poses/sequences")
    print("  Startup: HIGH mode.")
    if CALIBRATE:
        print("  [CALIBRATION MODE ACTIVE] Pressed button indices are displayed live.")
    print("=" * 55 + "\n")

    teleop_input = TeleopInputReader(deadzone=0.1)
    teleop_policy = HybridPolicyWrapper(base_policy, env, gamepad=teleop_input, device=device)

    resolved_viewer = cfg.viewer
    if resolved_viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"

    try:
        if resolved_viewer == "native":
            NativeMujocoViewer(env, teleop_policy).run()
        elif resolved_viewer == "viser":
            ViserPlayViewer(env, teleop_policy).run()
        else:
            raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
    finally:
        teleop_input.close()
        env.close()


def main():
    import mjlab.tasks  # noqa: F401
    import src.tasks

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        TeleopConfig,
        args=remaining_args,
        default=TeleopConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    run_teleop(chosen_task, args)


if __name__ == "__main__":
    main()