#!/usr/bin/env python3
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

# ── GENERAL CONFIG ──

nbr_joints = 29     # Total number of controlled joints
dt = 0.02           # Control loop at 50 Hz

# ── JOINT MAPPING (29 DOF - SDK / ROS2 ORDER) ──

dico_joints = {
    "hip_pitch_l": 0, "hip_roll_l": 1, "hip_yaw_l": 2, "knee_pitch_l": 3, "ankle_pitch_l": 4, "ankle_roll_l": 5,
    "hip_pitch_r": 6, "hip_roll_r": 7, "hip_yaw_r": 8, "knee_pitch_r": 9, "ankle_pitch_r": 10, "ankle_roll_r": 11,
    "waist_yaw": 12, "waist_roll": 13, "waist_pitch": 14,
    "should_pitch_l": 15, "should_roll_l": 16, "should_yaw_l": 17, "elbow_pitch_l": 18,
    "wrist_roll_l": 19, "wrist_pitch_l": 20, "wrist_yaw_l": 21,
    "should_pitch_r": 22, "should_roll_r": 23, "should_yaw_r": 24, "elbow_pitch_r": 25,
    "wrist_roll_r": 26, "wrist_pitch_r": 27, "wrist_yaw_r": 28
}
dico_joints_inv = {v: k for k, v in dico_joints.items()}        # Index -> name, for readable logs

# ── JOINT LIMITS (SDK ORDER) ──

JOINTS_MIN_MAX = [
    (-2.5307, 2.8798),       # 0: hip_pitch_l
    (-0.5236, 2.9671),       # 1: hip_roll_l
    (-2.7576, 2.7576),       # 2: hip_yaw_l
    (-0.087267, 2.8798),     # 3: knee_pitch_l
    (-0.87267, 0.5236),      # 4: ankle_pitch_l
    (-0.2618, 0.2618),       # 5: ankle_roll_l
    (-2.5307, 2.8798),       # 6: hip_pitch_r
    (-2.9671, 0.5236),       # 7: hip_roll_r
    (-2.7576, 2.7576),       # 8: hip_yaw_r
    (-0.087267, 2.8798),     # 9: knee_pitch_r
    (-0.87267, 0.5236),      # 10: ankle_pitch_r
    (-0.2618, 0.2618),       # 11: ankle_roll_r
    (-2.618, 2.618),         # 12: waist_yaw
    (-0.52, 0.52),           # 13: waist_roll
    (-0.52, 0.52),           # 14: waist_pitch
    (-3.0892, 2.6704),       # 15: should_pitch_l
    (-1.5882, 2.2515),       # 16: should_roll_l
    (-2.618, 2.618),         # 17: should_yaw_l
    (-1.0472, 2.0944),       # 18: elbow_pitch_l
    (-1.9722, 1.9722),       # 19: wrist_roll_l
    (-1.6144, 1.6144),       # 20: wrist_pitch_l
    (-1.6144, 1.6144),       # 21: wrist_yaw_l
    (-3.0892, 2.6704),       # 22: should_pitch_r
    (-2.2515, 1.5882),       # 23: should_roll_r
    (-2.618, 2.618),         # 24: should_yaw_r
    (-1.0472, 2.0944),       # 25: elbow_pitch_r
    (-1.9722, 1.9722),       # 26: wrist_roll_r
    (-1.6144, 1.6144),       # 27: wrist_pitch_r
    (-1.6144, 1.6144),       # 28: wrist_yaw_r
]

# ── POSES IN DEGREES (SDK ORDER) ──

positions = {
    "stand": np.zeros(nbr_joints),
    "chair": np.array([ -90, 0, 0, 90, 0, 0,
                        -90, 0, 0, 90, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0]),
    "squat": np.array([ -70, 0, -20, 70, -25, 0,
                        -70, 0, -20, 70, -25, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0]),
    "coucou1": np.array([0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         -90, 0, 0, 0, -90, 0, 0]),
    "coucou2": np.array([0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         -90, 0, 0, 0, -90, -30, 0]),
    "coucou3": np.array([0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0,
                         -90, 0, 0, 0, -90, 30, 0])
}

DEFAULT_POS = np.array([-0.312, 0, 0, 0.669, -0.363, 0,
                            -0.312, 0, 0, 0.669, -0.363, 0,
                            0, 0, 0,
                            0.2, 0.2, 0, 0.6, 0, 0, 0,
                            0.2, -0.2, 0, 0.6, 0, 0, 0], dtype=np.float32)      # Neutral pose used by the RL policy

# Poses converted to radians
positions_rad = {key: np.array([math.radians(angle) for angle in values], dtype=np.float32) for key, values in positions.items()}

# ── SEQUENCES ──

sequences = {
    "coucou": [("coucou1", 2.0), ("coucou2", 1.0), ("coucou3", 1.0)],
    "squat_workout": [("squat", 3.0), ("stand", 1.0)]
}

# ── GAINS ASSERVISSEMENT (Kp / Kd) ──

# Mujoco values, not adapted to RL

# Kp = np.array([ 60, 60, 60, 100, 40, 40,
#                 60, 60, 60, 100, 40, 40,
#                 60, 40, 40,
#                 40, 40, 40, 40, 40, 40, 40,
#                 40, 40, 40, 40, 40, 40, 40], dtype=np.float32)

# Kd = np.array([ 1, 1, 1, 2, 1, 1,
#                 1, 1, 1, 2, 1, 1,
#                 1, 1, 1,
#                 1, 1, 1, 1, 1, 1, 1,
#                 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)

# MJlab values

Kp = np.array([40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
               40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
               40.2, 28.5, 28.5,
               14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
               14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8], dtype=np.float32)

Kd = np.array([2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
               2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
               2.6, 1.8, 1.8,
               0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
               0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1], dtype=np.float32)

# ── RL OBSERVATION SCALARS ──

ACTION_SCALE = np.array([   0.55,0.35,0.55,0.35,0.44,0.44,
                            0.55,0.35,0.55,0.35,0.44,0.44,
                            0.55,0.44,0.44,
                            0.44,0.44,0.44,0.44,0.44,0.07,0.07,
                            0.44,0.44,0.44,0.44,0.44,0.07,0.07
                            ],dtype=np.float32)

# ── GLOBAL SDK STATE ──

low_state: LowState_ = None     # Latest /lowstate message received, updated by g1_ros2_main.py
crc = CRC()                     # CRC computation for LowCmd_ messages (see real_g1_bridge.py)

# ── UTILITY FUNCTIONS ──

def interpolate_cubic_vectorized(start_array, end_array, time, duration):       # Smooth cubic interpolation between two positions
    t = np.clip(time / duration, 0.0, 1.0)          # Normalized progress between 0 and 1
    phase = 3 * (t**2) - 2 * (t**3)                 # Cubic curve (zero velocity at both ends)
    return (1.0 - phase) * start_array + phase * end_array

def play_sequence(sequence, step_index, step_time, current_position, dt_val, forced_duration=None):     # Plays a sequence of predefined poses
    if sequence is None or step_index >= len(sequence):
        return current_position, step_index, step_time, current_position, True

    pose_name, duration = sequence[step_index]      # Target pose and default duration for the current step
    if forced_duration is not None:
        duration = forced_duration                  # Dynamic duration proportional to distance

    is_finished = False

    if pose_name in positions_rad:
        target_pose = positions_rad[pose_name]
        step_time += dt_val

        target_list = interpolate_cubic_vectorized(current_position, target_pose, step_time, duration)

        if step_time >= duration:
            step_index += 1                             # Move to the next step of the sequence
            step_time = 0.0
            current_position = np.copy(target_list)     # New starting point for the next step

            if step_index >= len(sequence):
                is_finished = True
    else:
        is_finished = True
        target_list = current_position

    return target_list, step_index, step_time, current_position, is_finished

def get_rpy_from_quat(quat):        # Converts quaternion [w, x, y, z] to Roll-Pitch-Yaw Euler angles in rad
    if quat is None or len(quat) < 4:
        return 0.0, 0.0, 0.0

    q = np.array(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return 0.0, 0.0, 0.0
    q = q / norm        # Normalization, safety against a malformed quaternion

    try:
        # SciPy expects [x, y, z, w]
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return float(roll), float(pitch), float(yaw)
    except Exception:
        return 0.0, 0.0, 0.0

def get_projected_gravity(quat):        # Computes the gravity vector [0, 0, -1] projected into the robot's local frame quat: [w, x, y, z]
    if quat is None or len(quat) < 4:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)

    q = np.array(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    q = q / norm

    r = R.from_quat([q[1], q[2], q[3], q[0]])       # Transform the [0, 0, -1] vector into the local frame
    grav_body = r.inv().apply([0.0, 0.0, -1.0])     # Inverse rotation: world frame -> body frame
    return grav_body.astype(np.float32)

def get_local_lin_vel(quat, vel_world):     # Converts a linear velocity from the world frame to the body's local frame
    q = np.array(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    r = R.from_quat([q[1], q[2], q[3], q[0]])  # SDK: [w,x,y,z] -> SciPy: [x,y,z,w]
    return r.inv().apply(vel_world).astype(np.float32)