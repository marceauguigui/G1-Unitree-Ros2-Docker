#!/usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node

# ROS 2 messages
from unitree_hg.msg import LowCmd, LowState

# SDK Unitree (DDS)
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, MotorCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.utils.crc import CRC


class RealG1Bridge(Node):
    def __init__(self, interface):
        super().__init__('real_g1_bridge')

        # DDS init
        ChannelFactoryInitialize(0, interface)
        self.get_logger().info(f"DDS Init on interface : {interface}")

        # Deactivation of the intern controller of G1
        self.release_internal_controller()
        self.crc = CRC()

        # ROS 2 -> DDS
        self.ros_sub = self.create_subscription(LowCmd, '/lowcmd', self.ros_cmd_callback, 10)
        self.dds_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.dds_pub.Init()

        # DDS -> ROS 2
        self.ros_pub = self.create_publisher(LowState, '/lowstate', 10)
        self.dds_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.dds_sub.Init(self.dds_state_callback, 10)

        self.motor_cmds = [
            MotorCmd_(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=0)
            for _ in range(35)
        ]

        self.get_logger().info("Bridge active !")

    def release_internal_controller(self):
        self.get_logger().info("Wait for control")
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        while result is not None and result.get('name'):
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(0.5)
        self.get_logger().info("Ready !")

    def ros_cmd_callback(self, ros_msg: LowCmd):
        dds_cmd = LowCmd_(  mode_pr=0,
                            mode_machine=ros_msg.mode_machine,
                            motor_cmd=self.motor_cmds,
                            reserve=[0] * 4,
                            crc=0)

        for i in range(35):
            dds_cmd.motor_cmd[i].mode  = ros_msg.motor_cmd[i].mode
            dds_cmd.motor_cmd[i].q     = ros_msg.motor_cmd[i].q
            dds_cmd.motor_cmd[i].dq    = ros_msg.motor_cmd[i].dq
            dds_cmd.motor_cmd[i].tau   = ros_msg.motor_cmd[i].tau
            dds_cmd.motor_cmd[i].kp    = ros_msg.motor_cmd[i].kp
            dds_cmd.motor_cmd[i].kd    = ros_msg.motor_cmd[i].kd

        dds_cmd.crc = self.crc.Crc(dds_cmd)
        self.dds_pub.Write(dds_cmd)

    def dds_state_callback(self, dds_msg: LowState_):
        if not rclpy.ok():
            return

        ros_msg = LowState()
        ros_msg.mode_machine = dds_msg.mode_machine

        # IMU mapping
        ros_msg.imu_state.quaternion    = list(dds_msg.imu_state.quaternion)
        ros_msg.imu_state.gyroscope     = list(dds_msg.imu_state.gyroscope)
        ros_msg.imu_state.accelerometer = list(dds_msg.imu_state.accelerometer)

        # Motor state mapping
        for i in range(29):
            ros_msg.motor_state[i].q       = dds_msg.motor_state[i].q
            ros_msg.motor_state[i].dq      = dds_msg.motor_state[i].dq
            ros_msg.motor_state[i].tau_est = dds_msg.motor_state[i].tau_est

        try:
            self.ros_pub.publish(ros_msg)
        except Exception:
            pass


def main():
    rclpy.init()
    interface = sys.argv[1] if len(sys.argv) > 1 else "enp1s0"
    node = RealG1Bridge(interface)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()