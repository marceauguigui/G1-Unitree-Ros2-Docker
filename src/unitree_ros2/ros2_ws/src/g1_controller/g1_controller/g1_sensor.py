import time
import sys
import math

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# Liste complète des 29 articulations du G1 29 DOF
JOINT_NAMES = [
    # Jambe Gauche (0-5)
    "L_HipPitch", "L_HipRoll", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
    # Jambe Droite (6-11)
    "R_HipPitch", "R_HipRoll", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll",
    # Taille / Bust (12-14)
    "WaistYaw", "WaistRoll", "WaistPitch",
    # Bras Gauche (15-21)
    "L_ShoulderPitch", "L_ShoulderRoll", "L_ShoulderYaw", "L_Elbow", "L_WristRoll", "L_WristPitch", "L_WristYaw",
    # Bras Droit (22-28)
    "R_ShoulderPitch", "R_ShoulderRoll", "R_ShoulderYaw", "R_Elbow", "R_WristRoll", "R_WristPitch", "R_WristYaw"
]

class G1StateReader29:
    def __init__(self):
        self.counter = 0

    def LowStateHandler(self, msg: LowState_):
        self.counter += 1
        
        # Affichage toutes les 4 secondes (2000 cycles à 500Hz)
        if self.counter % 2000 == 0:
            self.counter = 0
            
            # Efface l'écran du terminal pour figer l'affichage au même endroit
            print("\033[H\033[J") 
            print("=== POSITIONS ACTUELLES DES MOTEURS G1 (29 DOF) ===")
            print("-" * 55)
            print(f"{'Index':<6} | {'Nom du Joint':<20} | {'Position (Degrés)':<18}")
            print("-" * 55)
            
            # Boucle sur les 29 moteurs
            for i in range(29):
                pos_rad = msg.motor_state[i].q
                pos_deg = math.degrees(pos_rad)
                print(f"{i:<6} | {JOINT_NAMES[i]:<20} | {pos_deg:>8.2f} °")
            
            print("-" * 55)
            print("Appuie sur Ctrl+C pour quitter.")

if __name__ == '__main__':
    print("Démarrage du lecteur d'état 29 DOF...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    reader = G1StateReader29()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(reader.LowStateHandler, 10)

    print("Abonnement au flux rt/lowstate OK.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nArrêt du script.")