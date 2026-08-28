import time
import sys
import math

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# Full list of the 29 joints of the G1 29 DOF
JOINT_NAMES = [
    # Left leg (0-5)
    "L_HipPitch", "L_HipRoll", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
    # Right leg (6-11)
    "R_HipPitch", "R_HipRoll", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll",
    # Waist / torso (12-14)
    "WaistYaw", "WaistRoll", "WaistPitch",
    # Left arm (15-21)
    "L_ShoulderPitch", "L_ShoulderRoll", "L_ShoulderYaw", "L_Elbow", "L_WristRoll", "L_WristPitch", "L_WristYaw",
    # Right arm (22-28)
    "R_ShoulderPitch", "R_ShoulderRoll", "R_ShoulderYaw", "R_Elbow", "R_WristRoll", "R_WristPitch", "R_WristYaw"
]

class G1StateReader29:
    def __init__(self):
        self.counter = 0

    def LowStateHandler(self, msg: LowState_):
        self.counter += 1
        # Display every 4 seconds (2000 cycles at 500Hz)
        if self.counter % 2000 == 0:
            self.counter = 0
            # Clear the terminal screen to keep the display in the same place
            print("\033[H\033[J")
            print("=== CURRENT G1 MOTOR POSITIONS (29 DOF) ===")
            print("-" * 55)
            print(f"{'Index':<6} | {'Joint Name':<20} | {'Position (Degrees)':<18}")
            print("-" * 55)
            # Loop over the 29 motors
            for i in range(29):
                pos_rad = msg.motor_state[i].q
                pos_deg = math.degrees(pos_rad)
                print(f"{i:<6} | {JOINT_NAMES[i]:<20} | {pos_deg:>8.2f} °")
            print("-" * 55)
            print("Press Ctrl+C to quit.")

if __name__ == '__main__':
    print("Starting the 29 DOF state reader...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    reader = G1StateReader29()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(reader.LowStateHandler, 10)

    print("Subscribed to rt/lowstate stream OK.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping the script.")