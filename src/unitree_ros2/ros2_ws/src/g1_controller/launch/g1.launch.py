from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
import os

def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='sim',
        description='Launch mode : sim or real'
    )

    mode = LaunchConfiguration('mode')

    is_sim = IfCondition(PythonExpression(["'", mode, "' == 'sim'"]))
    is_real = IfCondition(PythonExpression(["'", mode, "' == 'real'"]))
    
    return LaunchDescription([
        mode_arg,
        
        ExecuteProcess(
            cmd=['python3', 'unitree_mujoco.py'],
            cwd=os.path.expanduser('/ros2_ws/src/unitree_mujoco/simulate_python'),
            output='screen',
            condition=is_sim
        ),

        Node(
            package='g1_controller',
            executable='real_g1_bridge',
            name='real_g1_bridge',
            arguments=['enp1s0'],
            output='screen',
            condition=is_real
        ),
        
        Node(
            package='g1_controller',
            executable='g1_ros2_controller',
            name='g1_ros2_controller',
            output='screen'
        ),

        Node(
            package='g1_controller',
            executable='ps5_bridge',
            name='ps5_bridge',
            output='screen'
        ),

        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            parameters=[{'dev': '/dev/input/js0'}]
        )
    ])