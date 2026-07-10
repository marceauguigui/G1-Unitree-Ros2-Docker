FROM osrf/ros:foxy-desktop

ENV DEBIAN_FRONTEND=noninteractive

# Outils indispensables pour ROS2, le dev et les dépendances Unitree/MuJoCo
RUN apt-get update && apt-get install -y \
    tmux \
    git \
    wget \
    nano \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    python3-scipy \
    ros-foxy-rosidl-generator-dds-idl \
    ros-foxy-rmw-cyclonedds-cpp \
    libglfw3-dev \
    libyaml-cpp-dev \
    libfmt-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install scipy cyclonedds==0.10.2 opencv-python mujoco
RUN mkdir -p /root && \
    ln -s /ros2_ws/src/unitree_mujoco /root/unitree_mujoco
RUN pip3 install scipy cyclonedds==0.10.2 opencv-python mujoco pygame

WORKDIR /ros2_ws

# Téléchargement et installation automatique de MuJoCo 3.2.3 à l'emplacement attendu
RUN mkdir -p /ros2_ws/src/unitree_mujoco/simulate && \
    wget -q https://github.com/google-deepmind/mujoco/releases/download/3.2.3/mujoco-3.2.3-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz && \
    tar -xf /tmp/mujoco.tar.gz -C /ros2_ws/src/unitree_mujoco/simulate && \
    mv /ros2_ws/src/unitree_mujoco/simulate/mujoco-3.2.3 /ros2_ws/src/unitree_mujoco/simulate/mujoco && \
    rm /tmp/mujoco.tar.gz

# Sourcing automatique de l'environnement ROS2 de base
RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc

# Injection des variables d'environnement et alias propres au Docker de manière sécurisée
RUN echo 'export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"' >> /root/.bashrc && \
    echo 'export ROS_DOMAIN_ID=1' >> /root/.bashrc && \
    echo 'export ROS_LOCALHOST_ONLY=1' >> /root/.bashrc && \
    echo 'alias brc="nano /root/.bashrc"' >> /root/.bashrc && \
    echo 'alias sbrc="source /root/.bashrc"' >> /root/.bashrc

# Raccourci de chargement adapté aux dossiers du Docker (si le dossier install existe)
RUN echo 'g1_load() { cd /ros2_ws && [ -f install/setup.bash ] && source install/setup.bash && echo "Environnement G1 chargé."; }' >> /root/.bashrc

# Le fameux raccourci CB adapté aux dossiers du Docker
RUN echo 'cb() { cd /ros2_ws && colcon build --symlink-install "$@" && [ -f install/setup.bash ] && source install/setup.bash && echo "Compilation terminée et environnement sourcé."; }' >> /root/.bashrc

# Configuration de la communication FastDDS
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

CMD ["bash"]