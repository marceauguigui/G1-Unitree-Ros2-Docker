#!/usr/bin/env python3
"""g1_ros2_main.py — G1 real-time controller (LOW / HIGH / RL / STOP modes)

Current policy: FLAT terrain, 99-dim observation (no height_scan — no terrain
sensor is available on the real robot):
    obs = base_lin_vel(3) + ang_vel(3) + projected_gravity(3) + joint_pos_rel(29)
        + joint_vel_rel(29) + last_action(29) + command_vel(3)

── base_lin_vel source (sim2sim vs. real robot) ──────────────────────────────
Chosen AUTOMATICALLY via the ROS2 parameter "mode", set from the launch file's
`mode:=sim` / `mode:=real` argument (the same one that already selects between
launching unitree_mujoco.py or real_g1_bridge — see g1.launch.py). No manual
code edit needed between sim2sim and real-robot runs.
 
  mode = "sim" (default if the parameter isn't set) — sim2sim, unitree_mujoco:
      Subscribes to /base_lin_vel (Float32MultiArray), published by the
      simulator from a MuJoCo <velocimeter> sensor placed at the same site
      mjlab uses during training (imu_in_pelvis). This sensor is NATIVELY in
      the robot's local/body frame — no rotation needed here.
 
  mode = "real" — real robot:
      No simulated velocimeter exists on hardware, so this falls back to
      /sportmodestate (published by the robot's onboard state estimator),
      read via a raw DDS subscription (unitree_sdk2py) rather than a ROS2
      message type — the ROS2 package `unitree_go` (which would define
      `SportModeState`) is not built in this workspace, so we bypass it and
      subscribe directly to the DDS topic "rt/sportmodestate" with the raw
      IDL type. This works because the real robot firmware publishes this
      topic over native DDS, independently of which ROS2 packages are built.
      /sportmodestate is assumed to be in WORLD frame (confirmed for the
      simulation bridge's `frame_vel` sensor; NOT independently verified on
      real hardware). get_local_lin_vel() rotates it into the body frame the
      policy expects. See the verification procedure at the end of this file
      before trusting this on the real robot.
      ALSO UNVERIFIED: whether /sportmodestate keeps publishing after
      release_internal_controller() (real_g1_bridge) releases the robot's
      "sport mode" — the topic name suggests it may be tied to that same
      subsystem. Check for a steady stream of "[sportstate] still receiving"
      log lines (see _sportstate_dds_callback) once the internal controller
      is released, before relying on this in RL mode.

── Other notes ────────────────────────────────────────────────────────────────
- Starts in LOW mode by default. LOW sets target_position = current_position
  on the very first state update, so there's zero position error at startup —
  no jump, no need for a gain ramp.
- Fall detection currently only LOGS (does not force STOP) — see the
  commented-out line in control_loop_callback. TODO before real hardware:
  uncomment it so a real fall actually cuts to STOP.
- Kp / Kd / DEFAULT_DOF_POS / ACTION_SCALE are assumed identical between the
  flat and rough training runs (same physical actuators — these don't depend
  on which terrain was used, only height_scan does).
"""

import os
import threading
import time
import collections

import numpy as np
import torch
import rclpy
from rclpy.node import Node

from unitree_hg.msg import LowCmd, LowState, MotorCmd
from std_msgs.msg import String, Float32MultiArray

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from g1_controller import g1_config
from g1_controller.g1_config import (
    nbr_joints, dt, dico_joints, dico_joints_inv, JOINTS_MIN_MAX, positions_rad,
    sequences, Kp, Kd, ACTION_SCALE, DEFAULT_DOF_POS, get_projected_gravity,
    get_local_lin_vel,
)
# Print raw observation terms to the log once per second, for sim2sim
PRINT_OBS_DEBUG = False

# Only used when running in "real" mode
DDS_DOMAIN_ID = 0
DDS_INTERFACE = "eno1"

EXPECTED_OBS_DIM = 99  # flat policy, no height_scan


class G1Ros2Controller(Node):
    def __init__(self):
        super().__init__('g1_ros2_main')

        # ── State variables ──
        self.control_mode = "LOW"
        self.target_position = np.zeros(nbr_joints)
        self.current_position = None
        self.time = 0.0
        self.clock = 0.0
        self.dynamic_duration = 5.0

        self.current_cmd = "idle"
        self.active_sequence = None
        self.sequence_step = 0

        # ── ROS2 Subscribers ──
        self.state_sub = self.create_subscription(LowState, '/lowstate', self.state_callback, 10)
        self.command_sub = self.create_subscription(String, '/command', self.command_callback, 10)

        # ── ROS2 Publishers ──
        self.cmd_pub = self.create_publisher(LowCmd, '/lowcmd', 10)

        # ── Control loop ──
        self.control_timer = self.create_timer(dt, self.control_loop_callback)
        self.get_logger().warn(f"Node started. Mode: {self.control_mode}")

        # ── ROS2 parameters ──
        self.declare_parameter('mode', 'sim')
        robot_mode = self.get_parameter('mode').value
        self.use_sportstate_for_lin_vel = (robot_mode == 'real')

        # ── base_lin_vel source ──
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self._sportstate_msg_count = 0
    
        if self.use_sportstate_for_lin_vel:
            # Raw DDS subscription, not a ROS2 message type — see module docstring.
            # ChannelFactoryInitialize must be called exactly once per process.
            # Running a raw DDS participant alongside rclpy in the same process
            # is fine: multiple DDS participants per process are natively supported.
            self._sportstate_lock = threading.Lock()
            try:
                ChannelFactoryInitialize(DDS_DOMAIN_ID, DDS_INTERFACE)
                self.sportstate_sub_raw = ChannelSubscriber("rt/sportmodestate", SportModeState_)
                self.sportstate_sub_raw.Init(self._sportstate_dds_callback, 10)
                self.get_logger().info("Raw DDS subscription to rt/sportmodestate initialized (real robot mode).")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize raw DDS subscription for sportmodestate: {e}")
        else:
            self.base_lin_vel_sub = self.create_subscription(
                Float32MultiArray, '/base_lin_vel', self._base_lin_vel_sim_callback, 10
            )
            self.get_logger().info("Subscribed to /base_lin_vel (sim2sim mode, native local-frame velocimeter).")

        # ── RL policy & related state ──
        self.policy = None
        self.default_dof_pos = DEFAULT_DOF_POS
        self.action_scale = ACTION_SCALE
        self.last_action = np.zeros(nbr_joints)
        self.command_vel = np.zeros(3, dtype=np.float32)

        self.expected_input_dim = None
        self.history_length = 1
        self.obs_buffer = collections.deque(maxlen=self.history_length)

        # UPDATE THIS PATH to the .pt exported from the FLAT checkpoint, with observation normalization
        # make sure you're using the corrected export script).
        policy_path = "src/unitree_ros2/ros2_ws/src/g1_exported_policies/mjpolicy_0.pt"
        try:
            if os.path.exists(policy_path):
                self.policy = torch.jit.load(policy_path)
                self.policy.eval()
                for param in self.policy.parameters():
                    if param.ndim == 2:
                        self.expected_input_dim = param.shape[1]
                        break
                self.get_logger().info(f"RL policy (TorchScript) loaded: {policy_path}")
                self.get_logger().info(f"Expected policy input dimension: {self.expected_input_dim}")
                if self.expected_input_dim != EXPECTED_OBS_DIM:
                    self.get_logger().error(
                        f"Expected {EXPECTED_OBS_DIM} dims (flat, no height_scan), got "
                        f"{self.expected_input_dim} — check that you're loading the FLAT "
                        f"checkpoint, not the rough one!")
            else:
                self.get_logger().warn(f"Policy file not found at {policy_path}! RL mode disabled.")
        except Exception as e:
            self.get_logger().error(f"Failed to load RL model: {e}")

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _base_lin_vel_sim_callback(self, msg: Float32MultiArray):
        self.base_lin_vel = np.array(msg.data, dtype=np.float32)

    def _sportstate_dds_callback(self, msg: SportModeState_):   # Runs on unitree_sdk2py's internal callback thread, NOT the rclpy
        with self._sportstate_lock:
            self.base_lin_vel = np.array(msg.velocity, dtype=np.float32)

    def state_callback(self, msg: LowState):
        g1_config.low_state = msg
        if self.current_position is None:
            self.current_position = np.array([msg.motor_state[i].q for i in range(nbr_joints)])
            self.target_position = np.copy(self.current_position)
            self.get_logger().info("First position captured!")  

    def command_callback(self, msg: String):
        if not msg.data or msg.data == "idle":
            return
        cmd_text = msg.data.split()
        if not cmd_text:
            return

        if msg.data == "stop":  # Emergency stop: freeze all joints with safety damping.
            self.control_mode = "STOP"
            self.active_sequence = None
            self.current_cmd = "idle"
            self.get_logger().error("!!! STOP mode activated !!!")
            return

        # 1. MODE SWITCH
        elif cmd_text[0] == "mode":
            next_mode = cmd_text[1].upper()
            if next_mode != self.control_mode:
                if next_mode == "RL" and self.policy is None:
                    self.get_logger().error("Cannot switch to RL mode: no policy loaded!")
                    return

                self.control_mode = next_mode
                self.time = 0.0

                if self.control_mode == "RL":
                    self.last_action = np.zeros(nbr_joints)
                    self.obs_buffer.clear()
                    self.get_logger().info("RL mode activated: action history and observation buffer reset.")

                # Re-sync current position to avoid commanding a stale target left over from a previous mode/session.
                if g1_config.low_state is not None:
                    self.current_position = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)])

                self.current_cmd = "idle"
                self.active_sequence = None
                self.get_logger().info(f"Control mode switched to: {self.control_mode}")

        # 2. LOW MODE: controlling joints one at a time.
        elif self.control_mode == "LOW" and cmd_text[0] == "joint":
            if g1_config.low_state is not None:
                self.current_position = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)])
            try:
                joint_input = cmd_text[1]
                val = 0.02 * float(cmd_text[2])
                if joint_input.isdigit():
                    idx = int(joint_input)
                elif joint_input in dico_joints:
                    idx = dico_joints[joint_input]
                else:
                    self.get_logger().warn(f"Unknown joint: '{joint_input}'")
                    return
                if 0 <= idx < nbr_joints:
                    min_val, max_val = JOINTS_MIN_MAX[idx]
                    self.target_position[idx] = float(np.clip(self.target_position[idx] + val, min_val, max_val))
                    self.get_logger().info(f"Joint {dico_joints_inv[idx]} [{idx}] -> target: {self.target_position[idx]:.4f}")
            except Exception as e:
                self.get_logger().warn(f"Joint format error: 'joint [name/idx] [val]' -> {e}")

        # 3. HIGH MODE: predefined poses or sequences.
        elif self.control_mode == "HIGH":
            self.time = 0.0
            if g1_config.low_state is not None:
                self.current_position = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)])
            if hasattr(g1_config, 'sequences') and msg.data in sequences:
                self.active_sequence = sequences[msg.data]
                self.sequence_step = 0
                self.get_logger().info(f"Sequence started in HIGH mode: {msg.data}")
            else:
                self.active_sequence = None
                self.current_cmd = msg.data
                if self.current_cmd in positions_rad:
                    max_dist = np.max(np.abs(positions_rad[self.current_cmd] - self.current_position))
                    self.dynamic_duration = max(max_dist * 2, 0.5)
                    self.get_logger().info(f"New fixed pose requested: {self.current_cmd} (duration: {self.dynamic_duration:.2f}s)")

        # 4. RL MODE: updates velocity targets from joystick/keyboard commands.
        elif self.control_mode == "RL":
            if cmd_text[0] in ["vel", "cmd"] and len(cmd_text) >= 4:
                try:
                    self.command_vel = np.array([float(cmd_text[1]), float(cmd_text[2]), float(cmd_text[3])], dtype=np.float32)
                except ValueError:
                    self.get_logger().warn(f"Invalid velocity command format: {msg.data}")

    # ── Main control loop ─────────────────────────────────────────────────

    def control_loop_callback(self):
        if self.current_position is None or g1_config.low_state is None:
            return

        cmd = LowCmd()
        cmd.mode_machine = g1_config.low_state.mode_machine

        self.clock += dt
        if self.clock >= 60.0:
            self.clock = 0.0

        quat = g1_config.low_state.imu_state.quaternion
        rpy = g1_config.get_rpy_from_quat(quat)
        if rpy is None:
            self.get_logger().warn("IMU RPY unavailable, skipping control step")
            return
        roll, pitch, yaw = rpy

        # ── Fall detection ——
        if self.control_mode != "STOP" and rpy is not None:
            if abs(roll) > np.radians(30) or abs(pitch) > np.radians(30):
                # self.control_mode = "STOP"  #  TODO: ENABLE BEFORE REAL HARDWARE
                self.get_logger().error(f"🚨 FALL DETECTED! (Roll: {np.degrees(roll):.1f}°, Pitch: {np.degrees(pitch):.1f}°) -> STOP mode would activate.")

        # ── MODE: STOP (emergency damping) ──
        if self.control_mode == "STOP":
            for i in range(nbr_joints):
                motor_cmd = MotorCmd()
                motor_cmd.mode = 1
                motor_cmd.q = float(g1_config.low_state.motor_state[i].q)
                motor_cmd.kp = 0.0  # zero stiffness
                motor_cmd.kd = 2.0  # safety damping to resist free fall
                cmd.motor_cmd[i] = motor_cmd
            self.cmd_pub.publish(cmd)
            return

        # ── MODE: LOW ──
        if self.control_mode == "LOW":
            pass  # target_position is updated directly in command_callback

        # ── MODE: HIGH ──
        elif self.control_mode == "HIGH":
            if self.active_sequence is not None:
                state_name = self.active_sequence[self.sequence_step][0]
                max_dist = np.max(np.abs(positions_rad[state_name] - self.current_position))
                dynamic_step_duration = max(max_dist * 2, 0.5)
                (self.target_position, self.sequence_step, self.time, self.current_position, is_finished) = g1_config.play_sequence(
                    self.active_sequence, self.sequence_step, self.time, self.current_position, dt, dynamic_step_duration)
                if is_finished:
                    self.get_logger().info("Sequence completed")
                    self.active_sequence = None
            elif self.current_cmd in positions_rad:
                self.time += dt
                self.target_position = g1_config.interpolate_cubic_vectorized(
                    self.current_position, positions_rad[self.current_cmd], self.time, self.dynamic_duration)

        # ── MODE: RL (99-dim observation, no height_scan) ──
        elif self.control_mode == "RL":
            if self.policy is None:
                self.get_logger().error("No RL model loaded!")
                return

            q = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)], dtype=np.float32)
            dq = np.array([g1_config.low_state.motor_state[i].dq for i in range(nbr_joints)], dtype=np.float32)
            ang_vel = np.array(g1_config.low_state.imu_state.gyroscope, dtype=np.float32)
            projected_gravity = g1_config.get_projected_gravity(quat)

            joint_pos_rel = (q - self.default_dof_pos).astype(np.float32)
            joint_vel_rel = dq.astype(np.float32)

            # base_lin_vel: rotate world->body frame only when reading from /sportmodestate (real robot path).
            # The sim2sim /base_lin_vel source is already in the body frame natively
            if self.use_sportstate_for_lin_vel:
                with self._sportstate_lock:
                    base_lin_vel_world = self.base_lin_vel.copy()
                base_lin_vel_local = get_local_lin_vel(quat, base_lin_vel_world)
            else:
                base_lin_vel_local = self.base_lin_vel.copy()


            obs_single = np.concatenate([
                base_lin_vel_local,   # 3
                ang_vel,              # 3
                projected_gravity,    # 3
                joint_pos_rel,        # 29
                joint_vel_rel,        # 29
                self.last_action,     # 29
                self.command_vel,     # 3
            ]).astype(np.float32)     # total = 99

            if PRINT_OBS_DEBUG and round(self.clock, 2) % 1 == 0:
                self.get_logger().info(f"[obs] base_lin_vel={base_lin_vel_local.tolist()}")
                self.get_logger().info(f"[obs] ang_vel={ang_vel.tolist()}")
                self.get_logger().info(f"[obs] grav={projected_gravity.tolist()}")
                self.get_logger().info(f"[obs] joint_pos_rel[:5]={joint_pos_rel[:5].tolist()}")
                self.get_logger().info(f"[obs] joint_vel_rel[:5]={joint_vel_rel[:5].tolist()}")
                self.get_logger().info(f"[obs] command={self.command_vel.tolist()}")

            if len(self.obs_buffer) == 0:
                for _ in range(self.history_length):
                    self.obs_buffer.append(obs_single)
            else:
                self.obs_buffer.append(obs_single)

            obs_flat = np.concatenate(list(self.obs_buffer)).astype(np.float32)
            obs_tensor = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                actions_tensor = self.policy(obs_tensor)
                actions = actions_tensor.squeeze(0).cpu().numpy()

            mins = np.array([x[0] for x in JOINTS_MIN_MAX], dtype=np.float32)
            maxs = np.array([x[1] for x in JOINTS_MIN_MAX], dtype=np.float32)
            actions_min = (mins - self.default_dof_pos) / self.action_scale
            actions_max = (maxs - self.default_dof_pos) / self.action_scale
            actions = np.clip(actions, actions_min, actions_max)

            if round(self.clock, 2) % 1 == 0:
                self.get_logger().info(f"RL actions -> min: {actions.min():.2f}, max: {actions.max():.2f}")
            self.last_action = actions

            self.target_position = self.default_dof_pos + actions * self.action_scale

        # ── CLAMP TARGETS to safe hardware limits ──
        mins = [limit[0] for limit in JOINTS_MIN_MAX]
        maxs = [limit[1] for limit in JOINTS_MIN_MAX]
        self.target_position = np.clip(self.target_position, mins, maxs)

        for i in range(nbr_joints):
            motor_cmd = MotorCmd()
            motor_cmd.mode = 1
            motor_cmd.q = float(self.target_position[i])
            motor_cmd.kp = float(Kp[i])
            motor_cmd.kd = float(Kd[i])
            cmd.motor_cmd[i] = motor_cmd

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = G1Ros2Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down G1 Controller Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()