"""
G1 Unitree - Configuration, constantes et fonctions utilitaires
===============================================================
"""

import math
import numpy as np
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

# CONFIGURATION

NUM_MOTORS = 29
JOINTS = [  0, 0, 0, 0, 0, 0,           # Jambe gauche (6)
            0, 0, 0, 0, 0, 0,           # Jambe droite (6)
            0, 0, 0,                    # Torse (3)
            0 ,0, 0, 0, 0, 0, 0,        # Bras gauche (7)
            0, 0, 0, 0, 0, 0, 0]        # Bras droit (7)

TARGETS = [ 0, 0, 0, 0, 0, 0,           # Jambe gauche (6)
            0, 0, 0, 0, 0, 0,           # Jambe droite (6)
            0, 0, 0,                    # Torse (3)
            0 ,0, 0, 0, 0, 0, 0,        # Bras gauche (7)
            0, 0, 0, 0, 0, 0, 0]        # Bras droit (7)

MOVE_DURATION   = 3.0     # secondes
dt              = 0.002   # 500 Hz

# Gains officiels redimensionnés pour les 29 moteurs du G1 29 DOF

Kp = [  60, 60, 60, 100, 40, 40,        # Jambe gauche (6)
        60, 60, 60, 100, 40, 40,        # Jambe droite (6)
        60, 40, 40,                     # Torse (3)
        40, 40, 40, 40, 40, 40, 40,     # Bras gauche (7)
        40, 40, 40, 40, 40, 40, 40]     # Bras droit (7)

# Amortissements redimensionnés pour les 29 moteurs

Kd = [  1, 1, 1, 2, 1, 1,               # Jambe gauche (6)
        1, 1, 1, 2, 1, 1,               # Jambe droite (6)
        1, 1, 1,                        # Torse (3)
        1, 1, 1, 1, 1, 1, 1,            # Bras gauche (7)
        1, 1, 1, 1, 1, 1, 1]            # Bras droit (7)

# ÉTAT GLOBAL

low_state: LowState_ = None
crc = CRC()

# FONCTIONS UTILITAIRES

def state_handler(msg: LowState_):
    global low_state
    low_state = msg


def compute_target(running_time: float, start_pos: list) -> list:
    "Calcule la position cible pour chaque joint actif."
    phase = np.tanh(running_time / MOVE_DURATION)
    targets = []
    for i in range(NUM_MOTORS):
        if JOINTS[i] == 1:
            # Joint actif : interpolation vers TARGETS[i]
            target_rad = math.radians(TARGETS[i])
            targets.append(phase * target_rad + (1 - phase) * start_pos[i])
        else:
            # Joint passif : reste à la position de départ
            targets.append(start_pos[i])
    return targets

def get_gravity_tau():
    "Lit le couple estimé (compensation gravité) depuis lowstate."
    if low_state is None:
        return [0.0] * NUM_MOTORS
    return [low_state.motor_state[i].tau_est for i in range(NUM_MOTORS)]

def reset():
    for i in range(NUM_MOTORS):
        JOINTS[i]  = 0
        TARGETS[i] = 0

def marche_jambe_gauche():
    reset()
    JOINTS[0]  = 1 ;  TARGETS[0]  = 20   # left_hip_roll
    JOINTS[1]  = 1 ;  TARGETS[1]  = 30   # left_hip_pitch
    JOINTS[3]  = 1 ;  TARGETS[3]  = 40   # left_knee
 
def marche_jambe_droite():
    reset()
    JOINTS[6]  = 1 ;  TARGETS[6]  = 20   # right_hip_roll
    JOINTS[7]  = 1 ;  TARGETS[7]  = 30   # right_hip_pitch
    JOINTS[9]  = 1 ;  TARGETS[9]  = 40   # right_knee