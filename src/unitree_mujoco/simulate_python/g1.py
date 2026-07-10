"""
G1 Unitree - Point d'entrée : initialisation et boucle de contrôle
==================================================================

Usage:
    python3 g1.py          # simulation (lo for loopback)
    python3 g1.py enp1s0     # vrai robot
"""

import time
import sys
import math
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import g1_config
from g1_config import (
    NUM_MOTORS, dt,
    state_handler, compute_target, get_gravity_tau, crc,
    marche_jambe_gauche
)

# ─────────────────────────────────────────────
# AFFICHAGE INITIAL
# ─────────────────────────────────────────────
print(f"\n=== G1 Test ===")
marche_jambe_gauche()
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':

    # ── Init réseau DDS ──────────────────────
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo") # simulation
    else:
        ChannelFactoryInitialize(0, sys.argv[1]) # réel

    # ── Libérer le robot du mode automatique ──
    if len(sys.argv) >= 2:
        # Vrai robot uniquement
        print("Libération des services automatiques Unitree...")
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        while result is not None and result.get('name'):
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(0.5)
        print("-> Mode Low-Level activé.")
    else:
        print("-> Mode simulation, MotionSwitcher ignoré.")

    # ── Subscriber : lire l'état du robot ────
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(state_handler, 10)

    # ── Publisher : envoyer les commandes ────
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    # ── Attendre le premier état valide ──────
    print("En attente du lowstate...", end="", flush=True)
    t0 = time.time()
    while g1_config.low_state is None:
        time.sleep(0.01)
        if time.time() - t0 > 5.0:
            print("\nERREUR : pas de lowstate reçu après 5s. Robot connecté ?")
            sys.exit(1)
    print(" OK")

    # ── Position de départ = position actuelle réelle ──
    start_pos = np.array([g1_config.low_state.motor_state[i].q for i in range(NUM_MOTORS)])

    # ── Init message de commande ──────────────
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = 0 # 0 pour sim, 1 pour reel
    cmd.mode_machine = 0 # 0 pour commandes Low Level # Autre pour mode avec commandes High Level

    for i in range(NUM_MOTORS):
        cmd.motor_cmd[i].mode = 1 # 0:Disable 1:Enable
        cmd.motor_cmd[i].q   = start_pos[i]
        cmd.motor_cmd[i].kp  = g1_config.Kp[i]
        cmd.motor_cmd[i].dq  = 0.0
        cmd.motor_cmd[i].kd  = g1_config.Kd[i]
        cmd.motor_cmd[i].tau = 0.0

    # ── Boucle de contrôle ────────────────────
    running_time = 0.0
    print(f"\nDémarrage du mouvement...")

    # ── Encapsulation de ta boucle dans le RecurrentThread ──
    def control_loop_callback():
        global running_time
        running_time += dt
        
        cmd.mode_pr = 0
        cmd.mode_machine = g1_config.low_state.mode_machine

        targets = compute_target(running_time, start_pos)

        for i in range(NUM_MOTORS):
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q   = targets[i]
            cmd.motor_cmd[i].kp  = g1_config.Kp[i]
            cmd.motor_cmd[i].kd  = g1_config.Kd[i]
            cmd.motor_cmd[i].dq  = 0.0
            cmd.motor_cmd[i].tau = 0.0

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

    # Activation de l'horloge système Unitree à 500Hz
    low_control_thread = RecurrentThread(interval=dt, target=control_loop_callback, name="control")
    low_control_thread.Start()

    # Bloque le script principal pour laisser tourner le thread
    while True:
        time.sleep(1)