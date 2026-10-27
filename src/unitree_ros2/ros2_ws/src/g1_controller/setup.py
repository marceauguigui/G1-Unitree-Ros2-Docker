import os
from glob import glob
from setuptools import setup

package_name = 'g1_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='noar',
    maintainer_email='marceauguigui@gmail.com',
    description='G1 ROS 2 Controller Node',
    license='ME',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'g1_ros2_main = g1_controller.g1_ros2_main:main',
            'controller_bridge = g1_controller.controller_bridge:main',
            'real_g1_bridge = g1_controller.real_g1_bridge:main',
            'keyboard_bridge = g1_controller.keyboard_bridge:main'
        ],
    },
)
