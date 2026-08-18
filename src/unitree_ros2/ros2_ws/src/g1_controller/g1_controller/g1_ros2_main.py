#!/usr/bin/env python3
"""g1_ros2_main.py — G1 real-time controller (LOW / HIGH / RL / STOP modes)

Current policy: FLAT terrain, 99-dim observation (no height_scan — no terrain
sensor is available on the real robot):
    obs = base_lin_vel(3) + ang_vel(3) + projected_gravity(3) + joint_pos_rel(29)
        + joint_vel_rel(29) + last_action(29) + command_vel(3)

- Fall detection currently only LOGS (does not force STOP) — see the
  commented-out line in control_loop_callback. TODO before real hardware:
  uncomment it so a real fall actually cuts to STOP.
"""

import os
import collections

import numpy as np
import torch
import rclpy
from rclpy.node import Node

from unitree_hg.msg import LowCmd, LowState, MotorCmd
from std_msgs.msg import String, Float32MultiArray
from g1_controller import g1_config
from g1_controller.g1_config import (nbr_joints, dt, dico_joints, dico_joints_inv, JOINTS_MIN_MAX, positions_rad,
                                     sequences, Kp, Kd, ACTION_SCALE, DEFAULT_DOF_POS,)

EXPECTED_OBS_DIM = 99  # 99 for flat policy, no height_scan

class G1Ros2Controller(Node):
    def __init__(self):
        super().__init__('g1_ros2_main')

        # ── State variables ──
        self.control_mode = "LOW"                       # Control mode
        self.current_position = None                    # Current position of the motors
        self.target_position = np.zeros(nbr_joints)     # Target position of the motors
        self.clock = 0.0                                # Clock for logs
        
        self.time = 0.0                                 # Time reference for positions in HIGH mode
        self.current_cmd = "idle"                       # Position defined in HIGH mode
        self.active_sequence = None                     # Sequence defined in HIGH mode
        self.dynamic_duration = 5.0                     # Time to move to the defined position of the sequence
        self.sequence_step = 0                          # Index of the position of the sequence

        # ── ROS2 Subscribers ──
        self.state_sub = self.create_subscription(LowState, '/lowstate', self.state_callback, 10)                                   # Motor & IMU state from the robot or simulator
        self.command_sub = self.create_subscription(String, '/command', self.command_callback, 10)                                  # Text commands (mode, joint, pose, vel...)
        self.base_lin_vel_sub = self.create_subscription(Float32MultiArray, '/base_lin_vel', self._base_lin_vel_sim_callback, 10)   # Linear velocity of the base ONLY IN SIM

        # ── ROS2 Publishers ──
        self.cmd_pub = self.create_publisher(LowCmd, '/lowcmd', 10)                                                                 # Commands sent joint per joint

        # ── Control loop ──
        self.control_timer = self.create_timer(dt, self.control_loop_callback)      # Fixed-rate control loop, same dt as training
        self.get_logger().warn(f"Node started. Mode: {self.control_mode}")

        # ── base_lin_vel source ──
        self.base_lin_vel = np.zeros(3, dtype=np.float32)       # Linear velocity of the base (local frame)
    
        # ── RL policy & related state ──
        self.policy = None
        self.default_dof_pos = DEFAULT_DOF_POS                          # Default position used in the Reinforcement Learning
        self.action_scale = ACTION_SCALE                                # Action scale used in the Reinforcement Learning
        self.last_action = np.zeros(nbr_joints)                         # Previous action
        self.command_vel = np.zeros(3, dtype=np.float32)                # Velocity input

        self.expected_input_dim = None                                  # Input dimension expected by the loaded policy
        self.history_length = 1                                         # Number of stacked observation frames
        self.obs_buffer = collections.deque(maxlen=self.history_length) # Buffer of past observations

        policy_path = "src/g1_exported_policies/isaac_policy_flat.pt"     # UPDATE THIS PATH to the .pt wanted
        try:
            if os.path.exists(policy_path):
                self.policy = torch.jit.load(policy_path)                                   # Load the exported TorchScript policy
                self.policy.eval()
                for param in self.policy.parameters():
                    if param.ndim == 2:
                        self.expected_input_dim = param.shape[1]
                        break
                self.get_logger().info(f"RL policy (TorchScript) loaded: {policy_path}")
                self.get_logger().info(f"Expected policy input dimension: {self.expected_input_dim}")
                if self.expected_input_dim != EXPECTED_OBS_DIM:
                    self.get_logger().error(
                        f"Expected {EXPECTED_OBS_DIM} dims flat got "
                        f"{self.expected_input_dim} — check that you're loading the right dimension policy !")
            else:
                self.get_logger().warn(f"Policy file not found at {policy_path}! RL mode disabled.")
        except Exception as e:
            self.get_logger().error(f"Failed to load RL model: {e}")

    # ── CALLBACKS ──────────────────────────────────────────────────────────

    def _base_lin_vel_sim_callback(self, msg: Float32MultiArray):
        self.base_lin_vel = np.array(msg.data, dtype=np.float32)    # Update base linear velocity from the simulator

    def state_callback(self, msg: LowState):
        g1_config.low_state = msg                                                                   # Store latest low level state
        if self.current_position is None:
            self.current_position = np.array([msg.motor_state[i].q for i in range(nbr_joints)])     # Capture starting position at the beginning
            self.target_position = np.copy(self.current_position)                                   # Target = current at boot to avoid a jump
            self.get_logger().info("First position captured!")  

    def command_callback(self, msg: String):
        if not msg.data or msg.data == "idle":
            return
        cmd_text = msg.data.split()     # Split "cmd arg1 arg2 ..." into tokens
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
            next_mode = cmd_text[1].upper()     # Requested control mode
            if next_mode != self.control_mode:
                if next_mode == "RL" and self.policy is None:
                    self.get_logger().error("Cannot switch to RL mode: no policy loaded!")
                    return

                self.control_mode = next_mode
                self.time = 0.0     # Reset HIGH mode interpolation timer on every switch

                if self.control_mode == "RL":
                    self.last_action = np.zeros(nbr_joints)     # Clear previous action history
                    self.obs_buffer.clear()                     # Clear stacked observation buffer
                    self.get_logger().info("RL mode activated: action history and observation buffer reset.")

                if g1_config.low_state is not None:     # Sync current position to avoid jumps from a previous mode
                    self.current_position = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)])

                self.current_cmd = "idle"
                self.active_sequence = None
                self.get_logger().info(f"Control mode switched to: {self.control_mode}")

        # 2. LOW MODE: controlling joints one at a time.
        elif self.control_mode == "LOW" and cmd_text[0] == "joint":
            if g1_config.low_state is not None:     # Sync current position to avoid jumps from a previous mode
                self.current_position = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)])
            try:
                joint_input = cmd_text[1]           # Joint name
                val = 0.02 * float(cmd_text[2])     # Increment step (scale down more precision)
                if joint_input.isdigit():
                    idx = int(joint_input)
                elif joint_input in dico_joints:
                    idx = dico_joints[joint_input]
                else:
                    self.get_logger().warn(f"Unknown joint: '{joint_input}'")
                    return
                if 0 <= idx < nbr_joints:       # Hardware safety limits for this joint
                    min_val, max_val = JOINTS_MIN_MAX[idx]
                    self.target_position[idx] = float(np.clip(self.target_position[idx] + val, min_val, max_val))   # Apply and clamp the increment
                    self.get_logger().info(f"Joint {dico_joints_inv[idx]} [{idx}] -> target: {self.target_position[idx]:.4f}")
            except Exception as e:
                self.get_logger().warn(f"Joint format error: 'joint [name/idx] [val]' -> {e}")

        # 3. HIGH MODE: predefined poses or sequences.
        elif self.control_mode == "HIGH":
            self.time = 0.0                                     # Reset interpolation timer for the new pose/sequence
            if g1_config.low_state is not None:                 # Sync current position to avoid jumps from a previous mode
                self.current_position = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)])
            if hasattr(g1_config, 'sequences') and msg.data in sequences:
                self.active_sequence = sequences[msg.data]      # Start a predefined sequence
                self.sequence_step = 0
                self.get_logger().info(f"Sequence started in HIGH mode: {msg.data}")
            else:
                self.active_sequence = None
                self.current_cmd = msg.data                     # Target a single fixed pose
                if self.current_cmd in positions_rad:
                    max_dist = np.max(np.abs(positions_rad[self.current_cmd] - self.current_position))  # Max joint displacement needed
                    self.dynamic_duration = max(max_dist * 2, 0.5)                                      # Dynamic duration according to the max joint displacement
                    self.get_logger().info(f"New fixed pose requested: {self.current_cmd} (duration: {self.dynamic_duration:.2f}s)")

        # 4. RL MODE: updates velocity targets from joystick/keyboard commands.
        elif self.control_mode == "RL":
            if cmd_text[0] in ["vel", "cmd"] and len(cmd_text) >= 4:
                try:
                    self.command_vel = np.array([float(cmd_text[1]), float(cmd_text[2]), float(cmd_text[3])], dtype=np.float32)     # vx, vy, yaw
                except ValueError:
                    self.get_logger().warn(f"Invalid velocity command format: {msg.data}")

    # ── MAIN CONTROL LOOP ─────────────────────────────────────────────────

    def control_loop_callback(self):
        if self.current_position is None or g1_config.low_state is None:
            return

        cmd = LowCmd()                                                              # Low-level command, filled below
        cmd.mode_machine = g1_config.low_state.mode_machine

        self.clock += dt                                                            # Logging 60s clock
        if self.clock >= 60.0:
            self.clock = 0.0

        quat = g1_config.low_state.imu_state.quaternion                             # Current base orientation
        rpy = g1_config.get_rpy_from_quat(quat)                                     # Roll / pitch / yaw derived from the quaternion
        if rpy is None:
            self.get_logger().warn("IMU RPY unavailable, skipping control step")
            return
        roll, pitch, yaw = rpy

        # ── Fall detection ——
        if self.control_mode != "STOP" and rpy is not None:
            if abs(roll) > np.radians(30) or abs(pitch) > np.radians(30):           # 30° tilt threshold
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
                state_name = self.active_sequence[self.sequence_step][0]                        # Name of the current sequence step's pose
                max_dist = np.max(np.abs(positions_rad[state_name] - self.current_position))    # Max joint displacement needed per step
                dynamic_step_duration = max(max_dist * 2, 0.5)                                  # Dynamic duration according to the max joint displacement per step
                (self.target_position, self.sequence_step, self.time, self.current_position, is_finished) = g1_config.play_sequence(
                    self.active_sequence, self.sequence_step, self.time, self.current_position, dt, dynamic_step_duration)
                if is_finished:
                    self.get_logger().info("Sequence completed")
                    self.active_sequence = None
            elif self.current_cmd in positions_rad:
                self.time += dt
                self.target_position = g1_config.interpolate_cubic_vectorized(
                    self.current_position, positions_rad[self.current_cmd], self.time, self.dynamic_duration)       # Smooth cubic move to the target pose

        # ── MODE: RL ──
        elif self.control_mode == "RL":
            if self.policy is None:
                self.get_logger().error("No RL model loaded!")
                return

            q = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)], dtype=np.float32)       # Current joint positions
            dq = np.array([g1_config.low_state.motor_state[i].dq for i in range(nbr_joints)], dtype=np.float32)     # Current joint velocities
            ang_vel = np.array(g1_config.low_state.imu_state.gyroscope, dtype=np.float32)                           # Base angular velocity
            projected_gravity = g1_config.get_projected_gravity(quat)                                               # Gravity vector expressed in the base frame
            joint_pos_rel = (q - self.default_dof_pos).astype(np.float32)                                           # Joint positions relative to the default pose
            joint_vel_rel = dq.astype(np.float32)                                                                   # Joint velocities (already relative)
            base_lin_vel_local = self.base_lin_vel.copy()                                                           # Base linear velocity, local frame

            obs_single = np.concatenate([
                base_lin_vel_local,   # 3
                ang_vel,              # 3
                projected_gravity,    # 3
                joint_pos_rel,        # 29
                joint_vel_rel,        # 29
                self.last_action,     # 29
                self.command_vel,     # 3
            ]).astype(np.float32)     # total = 99

            if len(self.obs_buffer) == 0:
                for _ in range(self.history_length):
                    self.obs_buffer.append(obs_single)                                  # Fill the buffer with the first observation on entry
            else:
                self.obs_buffer.append(obs_single)

            obs_flat = np.concatenate(list(self.obs_buffer)).astype(np.float32)         # Flatten the stacked observation history
            obs_tensor = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)       # Add the batch dimension expected by the policy

            with torch.no_grad():
                actions_tensor = self.policy(obs_tensor)                                # Forward pass through the policy
                actions = actions_tensor.squeeze(0).cpu().numpy()

            mins = np.array([x[0] for x in JOINTS_MIN_MAX], dtype=np.float32)           # Hardware lower limits
            maxs = np.array([x[1] for x in JOINTS_MIN_MAX], dtype=np.float32)           # Hardware upper limits
            actions_min = (mins - self.default_dof_pos) / self.action_scale             # Lower limits expressed in action space
            actions_max = (maxs - self.default_dof_pos) / self.action_scale             # Upper limits expressed in action space
            actions = np.clip(actions, actions_min, actions_max)                        # Prevent an out-of-range joint target

            if round(self.clock, 2) % 1 == 0:
                self.get_logger().info(f"RL actions -> min: {actions.min():.2f}, max: {actions.max():.2f}")
            self.last_action = actions                                                  # Stored for next tick's observation

            self.target_position = self.default_dof_pos + actions * self.action_scale   # Convert the policy action into a joint target

        # ── CLAMP TARGETS to safe hardware limits ──
        mins = [limit[0] for limit in JOINTS_MIN_MAX]
        maxs = [limit[1] for limit in JOINTS_MIN_MAX]
        self.target_position = np.clip(self.target_position, mins, maxs)                # Final safety clamp, applies to every mode above

        for i in range(nbr_joints):
            motor_cmd = MotorCmd()
            motor_cmd.mode = 1
            motor_cmd.q = float(self.target_position[i])        # Target position
            motor_cmd.kp = float(Kp[i])                         # Position gain
            motor_cmd.kd = float(Kd[i])                         # Damping gain
            cmd.motor_cmd[i] = motor_cmd

        self.cmd_pub.publish(cmd)                               # Send the command to the robot or simulator


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