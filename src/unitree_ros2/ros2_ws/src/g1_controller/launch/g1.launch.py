from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable
import os

def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='sim',
        description='Launch mode : sim or real'
    )

    config = LaunchConfiguration('config')

    is_sim = IfCondition(PythonExpression(["'", config, "' == 'sim'"]))
    is_real = IfCondition(PythonExpression(["'", config, "' == 'real'"]))
    SetEnvironmentVariable('ROS_DOMAIN_ID', '1', condition=is_sim),
    SetEnvironmentVariable('ROS_DOMAIN_ID', '0', condition=is_real),
    
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
            arguments=['eno1'],
            output='screen',
            condition=is_real
        ),
        
        Node(
            package='g1_controller',
            executable='g1_ros2_main',
            name='g1_ros2_main',
            output='screen',
            parameters=[{'config': config}]
        ),

        Node(
            package='g1_controller',
            executable='controller_bridge',
            name='controller_bridge',
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