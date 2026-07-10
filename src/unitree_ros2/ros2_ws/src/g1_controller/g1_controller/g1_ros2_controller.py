#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node

from unitree_hg.msg import LowCmd, LowState, MotorCmd
from std_msgs.msg import String

from g1_controller import g1_config
from g1_controller.g1_config import nbr_joints, dt, positions_rad, dico_joints, dico_joints_inv

class G1Ros2Controller(Node):
    def __init__(self):
        super().__init__('g1_ros2_controller')
        
        # ── State variables ──
        self.control_mode = "HIGH"       # "HIGH" (poses/sequences) or "LOW" (manual)
        self.target_position = np.zeros(nbr_joints)
        self.current_position = None
        self.time = 0.0
        self.clock = 0.0 # 1min clock
        self.dynamic_duration = 5

        # ── Sequence and States dedicated variables ──
        self.current_cmd = "idle"        # Stores the currently active fixed state
        self.active_sequence = None      # Stores the list of steps for the active sequence
        self.sequence_step = 0           # Index of the current sequence step

        # ── Subscriptions ──
        self.state_sub = self.create_subscription(LowState, '/lowstate', self.state_callback, 10)
        self.command_sub = self.create_subscription(String, '/command', self.command_callback, 10)

        # ── Publishers ──
        self.cmd_pub = self.create_publisher(LowCmd, '/lowcmd', 10)

        # ── Control loop ──
        self.control_timer = self.create_timer(dt, self.control_loop_callback)
        self.get_logger().info("Node started. Mode: HIGH")

        # ── ROS 2 Parameters ──
        self.declare_parameter('Kp', 1.0) #('name', default value)
        self.declare_parameter('Kd', 1.0)

    def state_callback(self, msg: LowState):
        g1_config.low_state = msg
        # Initial capture to avoid None values at startup
        if self.current_position is None:
            self.current_position = np.array([msg.motor_state[i].q for i in range(nbr_joints)])
            self.target_position = np.copy(self.current_position)
            self.get_logger().info("First position captured!")

    def command_callback(self, msg: String):
        if not msg.data or msg.data == "idle":
            return
        cmd_text = msg.data.split()
        if not cmd_text: return

        # Emergency stop: freeze all joints with safety damping
        if msg.data == "stop":
            self.control_mode = "STOP"
            self.active_sequence = None
            self.current_cmd = "idle"
            self.get_logger().error("!!! STOP mode activated !!!")
            return

        # 1. MODE SWITCH
        elif cmd_text[0] == "mode":
            next_mode = cmd_text[1].upper()
            if next_mode != self.control_mode:
                self.control_mode = next_mode
                self.time = 0.0
                # Capture the real position at the moment of the new command to avoid jumps
                if g1_config.low_state is not None:
                    self.current_position = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)])
                self.current_cmd = "idle"
                self.active_sequence = None
                self.get_logger().info(f"Mode switched to : {self.control_mode}")
            return

        # 2. LOW for controling joints 1 by 1
        elif self.control_mode == "LOW" and cmd_text[0] == "joint":
            # Capture the real position at the moment of the new command to avoid jumps
            if g1_config.low_state is not None:
                self.current_position = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)])
            try:
                joint_input = cmd_text[1]
                val = 0.02 * float(cmd_text[2]) # SENSIBILITY FOR RIGHT JOYSTICK
                # Accept joint by numeric index or by name
                if joint_input.isdigit():
                    idx = int(joint_input)
                elif joint_input in dico_joints:
                    idx = dico_joints[joint_input]
                else:
                    self.get_logger().warn(f"Unknown joint: '{joint_input}'")
                    return
                if (0 <= idx < nbr_joints):
                    min_val, max_val = g1_config.JOINTS_MIN_MAX[idx]
                    self.target_position[idx] = float(np.clip(self.target_position[idx] + val, min_val, max_val))
                    self.get_logger().info(f"Joint {dico_joints_inv[idx]} [{idx}] -> target: {self.target_position[idx]:.4f}")
            except Exception as e:
                self.get_logger().warn(f"Joint format error: 'joint [name/idx] [val]' -> {e}")
            
        # 3. HIGH for predefine poses or sequences
        elif self.control_mode == "HIGH":
            self.time = 0.0
            # Capture the real position at the moment of the new command to avoid jumps
            if g1_config.low_state is not None :
                self.current_position = np.array([g1_config.low_state.motor_state[i].q for i in range(nbr_joints)])
            # CAS A : SEQUENCE
            if hasattr(g1_config, 'sequences') and msg.data in g1_config.sequences:
                self.active_sequence = g1_config.sequences[msg.data]
                self.sequence_step = 0
                self.get_logger().info(f"Sequence started in HIGH mode: {msg.data}")
            
            # CAS B : STATIC STATE
            else:
                self.active_sequence = None # Interrupt any active sequence
                self.current_cmd = msg.data
                if self.current_cmd in positions_rad :
                    max_dist = np.max(np.abs(positions_rad[self.current_cmd] - self.current_position)) # Look at the motor with the biggest angle
                    self.dynamic_duration = max(max_dist*2, 0.5) # Speed adapt with a minimum of 0.5s
                    self.get_logger().info(f"New fixed pose requested: {self.current_cmd} (Duration: {self.dynamic_duration:.2f}s)")

    def control_loop_callback(self):
        if (self.current_position is None or g1_config.low_state is None):
            return
        
        param_Kp = self.get_parameter('Kp').value
        param_Kd = self.get_parameter('Kd').value
        cmd = LowCmd()
        cmd.mode_machine = g1_config.low_state.mode_machine

        self.clock += dt
        if self.clock >= 60.0:
            self.clock = 0.0

        # Read IMU orientation and evaluate robot stability
        quat = g1_config.low_state.imu_state.quaternion
        rpy = g1_config.get_rpy_from_quat(quat)
        if rpy is not None:
            roll, pitch, yaw = rpy
        status = g1_config.evaluate_stability_pitch(pitch)
        # Print each second
        if round(self.clock, 2)%1 == 0:
            self.get_logger().info(f"{status} ({pitch:.1f}°)")

        if self.control_mode == "STOP":
            for i in range(nbr_joints):
                motor_cmd = MotorCmd()
                motor_cmd.mode = 1
                motor_cmd.q = float(g1_config.low_state.motor_state[i].q)
                motor_cmd.kp = 0.0  # Zero torque
                motor_cmd.kd = 2.0  # Safety damping to resist free fall
                cmd.motor_cmd[i] = motor_cmd
            self.cmd_pub.publish(cmd)
            return
        
        # ── MODE (LOW) ──
        if self.control_mode == "LOW":
            pass

        # ── MODE (HIGH) ──
        elif self.control_mode == "HIGH":
            
            # Case A: Sequence
            if self.active_sequence is not None:
                state_name = self.active_sequence[self.sequence_step][0] # Get the name of the active state
                max_dist = np.max(np.abs(positions_rad[state_name] - self.current_position)) # Look at the motor with the biggest angle
                dynamic_step_duration = max(max_dist*2, 0.5) # Speed adapt with a minimum of 0.5s
                (self.target_position, self.sequence_step, self.time, self.current_position, is_finished) = g1_config.play_sequence(
                self.active_sequence, self.sequence_step, self.time, self.current_position, dt, dynamic_step_duration)
                
                if round(self.clock, 2)%1 == 0:
                    self.get_logger().info(f"Pose requested: {state_name} (Duration: {dynamic_step_duration:.2f}s)")
                
                if is_finished:
                    self.get_logger().info("Sequence completed")
                    self.active_sequence = None

            # Case B: Static State
            elif self.current_cmd in positions_rad:
                self.time += dt
                self.target_position = g1_config.interpolate_cubic_vectorized(self.current_position, positions_rad[self.current_cmd], self.time, self.dynamic_duration)

        mins = [limit[0] for limit in g1_config.JOINTS_MIN_MAX]
        maxs = [limit[1] for limit in g1_config.JOINTS_MIN_MAX]
        self.target_position = np.clip(self.target_position, mins, maxs)
        
        # ── APPLY COMPUTED TARGETS TO ALL MOTORS ──
        for i in range(nbr_joints):
            motor_cmd = MotorCmd()
            motor_cmd.mode = 1
            motor_cmd.q = float(self.target_position[i])
            motor_cmd.kp = float(g1_config.Kp[i]*param_Kp)
            motor_cmd.kd = float(g1_config.Kd[i]*param_Kd)
            cmd.motor_cmd[i] = motor_cmd

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = G1Ros2Controller()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info("Arrêt")
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()