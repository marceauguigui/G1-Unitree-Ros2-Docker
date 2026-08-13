# Docker commands

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

# Ros2 commands

# TERMINAL 1
ros2 run joy joy_node
# TERMINAL 2
ros2 run g1_controller ps5_bridge
# TERMINAL 3
ros2 run g1_controller g1_ros2_main
# TERMINAL 4
mujocopython //  mujocopython_real

# Exemples of publishers

ros2 topic pub --once /command std_msgs/msg/String "data: 'mode LOW'"
ros2 topic pub --once /command std_msgs/msg/String "data: 'joint knee_pitch_l 3.0'"
ros2 topic pub --once /command std_msgs/msg/String "data: 'joint 0 40.0'"

ros2 topic pub --once /command std_msgs/msg/String "data: 'mode HIGH'"
ros2 topic pub --once /command std_msgs/msg/String "data: 'stand'"

# LAUNCH FILE

ros2 launch g1_controller g1.launch.py mode:=sim
ros2 launch g1_controller g1.launch.py mode:=real