#!/usr/bin/env python3
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

# ── CONFIGURATION GÉNÉRALE ──

nbr_joints = 29
dt = 0.02   # Boucle de contrôle à 50 Hz

# ── CARTOGRAPHIE DES ARTICULATIONS (29 DOF - ORDRE SDK / ROS 2) ──

dico_joints = {
    "hip_pitch_l": 0, "hip_roll_l": 1, "hip_yaw_l": 2, "knee_pitch_l": 3, "ankle_pitch_l": 4, "ankle_roll_l": 5,
    "hip_pitch_r": 6, "hip_roll_r": 7, "hip_yaw_r": 8, "knee_pitch_r": 9, "ankle_pitch_r": 10, "ankle_roll_r": 11,
    "waist_yaw": 12, "waist_roll": 13, "waist_pitch": 14,
    "should_pitch_l": 15, "should_roll_l": 16, "should_yaw_l": 17, "elbow_pitch_l": 18,
    "wrist_roll_l": 19, "wrist_pitch_l": 20, "wrist_yaw_l": 21,
    "should_pitch_r": 22, "should_roll_r": 23, "should_yaw_r": 24, "elbow_pitch_r": 25,
    "wrist_roll_r": 26, "wrist_pitch_r": 27, "wrist_yaw_r": 28
}
dico_joints_inv = {v: k for k, v in dico_joints.items()}

# ── LIMITES ARTICULAIRES (ORDRE SDK) ──

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

# ── POSES EN DEGRÉS (ORDRE SDK) ──

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

DEFAULT_DOF_POS = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0,
    -0.312, 0, 0, 0.669, -0.363, 0,
    0, 0, 0,
    0.2, 0.2, 0, 0.6, 0, 0, 0,
    0.2, -0.2, 0, 0.6, 0, 0, 0
], dtype=np.float32)

# Poses converties en radians
positions_rad = {key: np.array([math.radians(angle) for angle in values], dtype=np.float32) for key, values in positions.items()}

# ── SÉQUENCES D'ANIMATION ──

sequences = {
    "coucou": [("coucou1", 2.0), ("coucou2", 1.0), ("coucou3", 1.0)],
    "squat_workout": [("squat", 3.0), ("stand", 1.0)]
}

# ── GAINS ASSERVISSEMENT (Kp / Kd) ──

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

# ── SCALAIRES OBSERVATIONS RL ──

OBS_SCALE_ANG_VEL = 0.2
OBS_SCALE_DOF_VEL = 0.05
ACTION_SCALE = np.array([   0.55,0.35,0.55,0.35,0.44,0.44,
                            0.55,0.35,0.55,0.35,0.44,0.44,
                            0.55,0.44,0.44,                         # waist_roll/pitch=0.0 for waist lock (were 0.44/0.44)
                            0.44,0.44,0.44,0.44,0.44,0.07,0.07,
                            0.44,0.44,0.44,0.44,0.44,0.07,0.07
                            ],dtype=np.float32)

# ── ÉTAT GLOBAL SDK ──

low_state: LowState_ = None
crc = CRC()

# ── FONCTIONS UTILITAIRES ──

def interpolate_cubic_vectorized(start_array, end_array, time, duration):
    """Interpolation cubique lissée entre deux positions."""
    t = np.clip(time / duration, 0.0, 1.0)
    phase = 3 * (t**2) - 2 * (t**3)
    return (1.0 - phase) * start_array + phase * end_array

def limit_step(current, target, max_step):
    """Limite la variation maximale par pas de temps."""
    return current + np.clip(target - current, -max_step, max_step)

def play_sequence(sequence, step_index, step_time, current_position, dt_val, forced_duration=None):
    """Joue une séquence de poses prédéfinies."""
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

def get_rpy_from_quat(quat):
    """
    Convertit un quaternion [w, x, y, z] en angles d'Euler Roll-Pitch-Yaw (en RADIANS).
    """
    if quat is None or len(quat) < 4:
        return 0.0, 0.0, 0.0

    q = np.array(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return 0.0, 0.0, 0.0
    q = q / norm

    try:
        # Repère SciPy attend [x, y, z, w]
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        return float(roll), float(pitch), float(yaw)
    except Exception:
        return 0.0, 0.0, 0.0

def get_projected_gravity(quat):
    """
    Calcule le vecteur de gravité projeté [0, 0, -1] dans le repère local du robot.
    quat: [w, x, y, z]
    """
    if quat is None or len(quat) < 4:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)

    q = np.array(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    q = q / norm

    # Transformation du vecteur [0, 0, -1] vers le repère local
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    grav_body = r.inv().apply([0.0, 0.0, -1.0])
    return grav_body.astype(np.float32)

def evaluate_stability_pitch(pitch_rad, seuil_perturbation_deg=5.0, seuil_chute_deg=30.0):
    """
    Évalue la stabilité du robot basée sur l'angle de pitch (en radians).
    """
    pitch_deg = abs(math.degrees(pitch_rad))
    if pitch_deg > seuil_chute_deg:
        return "CRITICAL"
    elif pitch_deg > seuil_perturbation_deg:
        return "DISTURBED"
    else:
        return "STABLE"

def get_local_lin_vel(quat, vel_world):
    """Convertit une vitesse linéaire du repère monde vers le repère local du corps."""
    q = np.array(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    r = R.from_quat([q[1], q[2], q[3], q[0]])  # SDK: [w,x,y,z] -> SciPy: [x,y,z,w]
    return r.inv().apply(vel_world).astype(np.float32)