# DOCKER COMMANDS

# Start the environment in the background (None -> Created -> Running):
docker compose up -d

# Rebuild the image (Dockerfile modified -> Rebuild -> Running):
xhost +local:docker
docker compose up -d --build

# Enter the Docker container (Running -> Open Terminal):
docker compose exec ros2_foxy bash

# Fully stop the container (Running -> Exited -> Removed):
docker compose down

# WARNING: THE CONTROLLER MUST BE CONNECTED WHEN LAUNCHING DOCKER FOR IT TO BE DETECTED

______________________________________________________________________________________________

# ROS2 COMMANDS

cb

# LAUNCH FILE

ros2 launch g1_controller g1.launch.py config:=sim
ros2 launch g1_controller g1.launch.py config:=real

# TERMINAL KEYBOARD
ros2 run g1_controller keyboard_bridge

# TERMINAL 1
ros2 run joy joy_node
# TERMINAL 2
ros2 run g1_controller controller_bridge
# TERMINAL 3
ros2 run g1_controller g1_ros2_main
# TERMINAL 4 SIM
cd /ros2_ws/src/unitree_mujoco/simulate_python
python3 unitree_mujoco.py
# TERMINAL 4 REAL
ros2 run g1_controller real_g1_bridge
______________________________________________________________________________________________

# MJLAB RAW COMMANDS

#TRAIN
python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs=4096 --agent.max-iterations 50000

python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs=4096 --agent.resume True --agent.max-iterations 50000

#PLAY
python scripts/play.py Mjlab-Velocity-Flat-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -n 1)

#TELEOP
python scripts/teleop.py Mjlab-Velocity-Flat-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -n 1)

#WORKING
______________________________________________________________________________________________

# ISAACLAB RAW COMMANDS

#TRAIN
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless

python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless --resume

#PLAY
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --num_envs 1

#TELEOP
python scripts/rsl_rl/teleop.py --task Unitree-G1-29dof-Velocity --num_envs 1

#WORKING
python scripts/rsl_rl/teleop.py --task Unitree-G1-29dof-Velocity --checkpoint ${workspaceFolder}/src/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-07-24_15-55-15/model_89999.pt --num_envs 1

# Data from RL

tensorboard --logdir=logs/