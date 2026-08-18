# DOCKER COMMANDS

# Démarrer l'environnement en arrière-plan (None -> Created -> Running) :
docker compose up -d

# Reconstruire l'image (Dockerfile modif -> Rebuild -> Running) :
xhost +local:docker
docker compose up -d --build

# Entrer dans le conteneur Docker (Running -> Open Terminal) :
docker compose exec ros2_foxy bash

# Arrêter complètement le conteneur (Running -> Exited -> Removed) :
docker compose down

# ATTENTION LA MANETTE DOIT ETRE CONNECTE AU MOMENT DE LANCER LE DOCKER POUR QU'IL LA DETECTE

______________________________________________________________________________________________

# ROS2 COMMANDS

# TERMINAL 1
ros2 run joy joy_node
# TERMINAL 2
ros2 run g1_controller ps5_bridge
# TERMINAL 3
ros2 run g1_controller g1_ros2_main
# TERMINAL 4 SIM
cd /ros2_ws/src/unitree_mujoco/simulate_python
python3 unitree_mujoco.py
# TERMINAL 4 REAL
ros2 run g1_controller real_g1_bridge

# Exemples of publishers

ros2 topic pub --once /command std_msgs/msg/String "data: 'mode LOW'"
ros2 topic pub --once /command std_msgs/msg/String "data: 'joint knee_pitch_l 3.0'"
ros2 topic pub --once /command std_msgs/msg/String "data: 'joint 0 40.0'"

ros2 topic pub --once /command std_msgs/msg/String "data: 'mode HIGH'"
ros2 topic pub --once /command std_msgs/msg/String "data: 'stand'"

# LAUNCH FILE

ros2 launch g1_controller g1.launch.py mode:=sim
ros2 launch g1_controller g1.launch.py mode:=real

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


tensorboard --logdir=logs/

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

tensorboard --logdir=logs/