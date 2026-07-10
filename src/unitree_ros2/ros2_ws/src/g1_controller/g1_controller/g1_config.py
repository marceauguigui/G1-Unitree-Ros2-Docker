#!/usr/bin/env python3
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

# CONFIGURATION

nbr_joints = 29
dt = 0.02   # 50 Hz control loop

# ── G1 Robot Joint Mapping (29 DOF) ──

dico_joints = {
    "hip_yaw_l": 0, "hip_roll_l": 1, "hip_pitch_l": 2, "knee_pitch_l": 3, "ankle_pitch_l": 4, "ankle_roll_l": 5,
    "hip_yaw_r": 6, "hip_roll_r": 7, "hip_pitch_r": 8, "knee_pitch_r": 9, "ankle_pitch_r": 10, "ankle_roll_r": 11,
    "waist_yaw": 12, "waist_roll": 13, "waist_pitch": 14,
    "should_pitch_l": 15, "should_roll_l": 16, "should_yaw_l": 17, "elbow_pitch_l": 18, "wrist_yaw_l": 19, "wrist_pitch_l": 20, "wrist_roll_l": 21,
    "should_pitch_r": 22, "should_roll_r": 23, "should_yaw_r": 24, "elbow_pitch_r": 25, "wrist_yaw_r": 26, "wrist_pitch_r": 27, "wrist_roll_r": 28
}
# Reverse mapping: index → joint name (used for logging and display)
dico_joints_inv = {
    0: "hip_yaw_l", 1: "hip_roll_l", 2: "hip_pitch_l", 3: "knee_pitch_l", 4: "ankle_pitch_l", 5: "ankle_roll_l",
    6: "hip_yaw_r", 7: "hip_roll_r", 8: "hip_pitch_r", 9: "knee_pitch_r", 10: "ankle_pitch_r", 11: "ankle_roll_r",
    12: "waist_yaw", 13: "waist_roll", 14: "waist_pitch",
    15: "should_pitch_l", 16: "should_roll_l", 17: "should_yaw_l", 18: "elbow_pitch_l", 19: "wrist_yaw_l", 20: "wrist_pitch_l", 21: "wrist_roll_l",
    22: "should_pitch_r", 23: "should_roll_r", 24: "should_yaw_r", 25: "elbow_pitch_r", 26: "wrist_yaw_r", 27: "wrist_pitch_r", 28: "wrist_roll_r"
}

# ── Joints limits ──

JOINTS_MIN_MAX = [
    (-2.7576, 2.7576),       # 0: hip_yaw_l (L_LEG_HIP_YAW)
    (-0.5236, 2.9671),       # 1: hip_roll_l (L_LEG_HIP_ROLL)
    (-2.5307, 2.8798),       # 2: hip_pitch_l (L_LEG_HIP_PITCH)
    (-0.087267, 2.8798),     # 3: knee_pitch_l (L_LEG_KNEE)
    (-0.87267, 0.5236),      # 4: ankle_pitch_l (L_LEG_ANKLE_PITCH)
    (-0.2618, 0.2618),       # 5: ankle_roll_l (L_LEG_ANKLE_ROLL)
    (-2.7576, 2.7576),       # 6: hip_yaw_r (R_LEG_HIP_YAW)
    (-2.9671, 0.5236),       # 7: hip_roll_r (R_LEG_HIP_ROLL)
    (-2.5307, 2.8798),       # 8: hip_pitch_r (R_LEG_HIP_PITCH)
    (-0.087267, 2.8798),     # 9: knee_pitch_r (R_LEG_KNEE)
    (-0.87267, 0.5236),      # 10: ankle_pitch_r (R_LEG_ANKLE_PITCH)
    (-0.2618, 0.2618),       # 11: ankle_roll_r (R_LEG_ANKLE_ROLL)
    (-2.618, 2.618),         # 12: waist_yaw (WAIST_YAW)
    (-0.52, 0.52),           # 13: waist_roll (WAIST_ROLL)
    (-0.52, 0.52),           # 14: waist_pitch (WAIST_PITCH)
    (-3.0892, 2.6704),       # 15: should_pitch_l (L_SHOULDER_PITCH)
    (-1.5882, 2.2515),       # 16: should_roll_l (L_SHOULDER_ROLL)
    (-2.618, 2.618),         # 17: should_yaw_l (L_SHOULDER_YAW)
    (-1.0472, 2.0944),       # 18: elbow_pitch_l (L_ELBOW)
    (-1.614429558, 1.614429558), # 19: wrist_yaw_l (L_WRIST_YAW)
    (-1.614429558, 1.614429558), # 20: wrist_pitch_l (L_WRIST_PITCH)
    (-1.972222054, 1.972222054), # 21: wrist_roll_l (L_WRIST_ROLL)
    (-3.0892, 2.6704),       # 22: should_pitch_r (R_SHOULDER_PITCH)
    (-2.2515, 1.5882),       # 23: should_roll_r (R_SHOULDER_ROLL)
    (-2.618, 2.618),         # 24: should_yaw_r (R_SHOULDER_YAW)
    (-1.0472, 2.0944),       # 25: elbow_pitch_r (R_ELBOW)
    (-1.614429558, 1.614429558), # 26: wrist_yaw_r (R_WRIST_YAW)
    (-1.614429558, 1.614429558), # 27: wrist_pitch_r (R_WRIST_PITCH)
    (-1.972222054, 1.972222054), # 28: wrist_roll_r (R_WRIST_ROLL)
]

# --- LEFT LEG (0-5) ---
# 0: Hip Yaw (L)    | 1: Hip Roll (L)   | 2: Hip Pitch (L)  | # 3: Knee Pitch (L) | 4: Ankle Pitch (L)| 5: Ankle Roll (L)

# --- RIGHT LEG (6-11) ---
# 6: Hip Yaw (R)    | 7: Hip Roll (R)   | 8: Hip Pitch (R)  | # 9: Knee Pitch (R) | 10: Ankle Pitch (R)| 11: Ankle Roll (R)

# --- WAIST / TORSO (12-14) ---
# 12: Waist Yaw     | 13: Waist Roll    | 14: Waist Pitch

# --- LEFT ARM (15-21) ---
# 15: Should. Pitch (L)| 16: Should. Roll (L) | 17: Should. Yaw (L) | # 18: Elbow Pitch (L)  | 19: Wrist Yaw (L)   | 20: Wrist Pitch (L) | 21: Wrist Roll (L)

# --- RIGHT ARM (22-28) ---
# 22: Should. Pitch (R)| 23: Should. Roll (R) | 24: Should. Yaw (R) | # 25: Elbow Pitch (R)  | 26: Wrist Yaw (R)   | 27: Wrist Pitch (R) | 28: Wrist Roll (R)

# Poses in degrees
positions = {
    "stand": np.zeros(nbr_joints),
    "chair": np.array([ -90, 0, 0, 90, 0, 0,
                        -90, 0, 0, 90, 0, 0,
                        0, 0, 0,
                        0 ,0, 0, 0, 0, 0, 0,
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

# Pre-converted poses in radians for maths
positions_rad = {key: np.array([math.radians(angle) for angle in values]) for key, values in positions.items()}


# ── SEQUENCES ──
# List of (pose_name, duration_in_seconds) tuples
sequences = {   "coucou": [("coucou1", 2.0), ("coucou2", 1.0), ("coucou3", 1.0)],
                "squat_workout": [("squat", 3.0), ("stand", 1.0)]}

# ── STIFFNESS GAINS (Kp) ──

Kp = np.array([60, 60, 60, 10, 40, 40,
               60, 60, 60, 10, 40, 40,
               60, 40, 40,
               40, 40, 40, 40, 40, 40, 40,
               40, 40, 40, 40, 40, 40, 40])

# ── DAMPING GAINS (Kd) ──

Kd = np.array([  1, 1, 1, 2, 1, 1,
        1, 1, 1, 2, 1, 1,
        1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1])

# ── GLOBAL STATE ──

low_state: LowState_ = None
crc = CRC()

# ── UTILITY FUNCTIONS ──

# To smoothen the joint control for joints lists
def interpolate_cubic_vectorized(start_array, end_array, time, duration): # (radians, radians, seconds, seconds)
    t = np.clip(time / duration, 0.0, 1.0)
    phase = 3 * (t**2) - 2 * (t**3)
    return (1 - phase) * start_array + phase * end_array

# To smoothen the joint control for single joints
# (Not used because done by incrementation but could be to give directly positions to joints)
def limit_step(current, target, max_step):
    return current + np.clip(target - current, -max_step, max_step)

def play_sequence(sequence, step_index, step_time, current_position, dt_val, forced_duration=None):
    if sequence is None or step_index >= len(sequence):
        return current_position, step_index, step_time, current_position, True

    pose_name, duration = sequence[step_index]
    if forced_duration is not None:
        duration = forced_duration
    else:
        sequence[step_index][1]
    is_finished = False

    if pose_name in positions_rad:
        target_pose = positions_rad[pose_name]
        step_time += dt_val
        
        target_list = interpolate_cubic_vectorized(current_position, target_pose, step_time, duration)
        
        # Step complete: advance to the next step
        if step_time >= duration:
            step_index += 1
            step_time = 0.0
            current_position = np.copy(target_list)
            
            if step_index >= len(sequence):
                is_finished = True
    else:
        # Unknown pose name: abort the sequence
        is_finished = True
        target_list = current_position
        
    return target_list, step_index, step_time, current_position, is_finished

def get_rpy_from_quat(quat): # Convert a quaternion to Roll-Pitch-Yaw angles (in degrees)
    if np.linalg.norm(quat) < 0.1:
        return None
    rotation = R.from_quat(quat)
    rpy = rotation.as_euler('xyz', degrees=True)
    roll, pitch, yaw = rpy
    return (roll, pitch, yaw)

def evaluate_stability_pitch(pitch, seuil_perturbation=5.0, seuil_chute=30.0): # Evaluate robot stability based on the pitch angle   
    if pitch > seuil_chute:
        return "CRITICAL"
    elif pitch > seuil_perturbation:
        return "DISTURBED"
    else:
        return "STABLE"