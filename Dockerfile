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
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libyaml-cpp-dev \
    libfmt-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install scipy cyclonedds==0.10.2 opencv-python mujoco pygame

WORKDIR /ros2_ws

# Sourcing automatique de l'environnement ROS2 de base
RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc

# Injection des variables d'environnement et alias propres au Docker
RUN echo 'export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"' >> /root/.bashrc && \
    echo 'export ROS_DOMAIN_ID=1' >> /root/.bashrc && \
    echo 'export ROS_LOCALHOST_ONLY=1' >> /root/.bashrc && \
    echo 'alias brc="nano /root/.bashrc"' >> /root/.bashrc && \
    echo 'alias sbrc="source /root/.bashrc"' >> /root/.bashrc

# Raccourcis ROS 2
RUN echo 'g1_load() { cd /ros2_ws && [ -f install/setup.bash ] && source install/setup.bash && echo "Environnement G1 chargé."; }' >> /root/.bashrc
RUN echo 'cb() { cd /ros2_ws && colcon build --symlink-install "$@" && [ -f install/setup.bash ] && source install/setup.bash && echo "Compilation terminée et environnement sourcé."; }' >> /root/.bashrc

# Rétablir CycloneDDS sur la boucle locale (remplace FastDDS)
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI="<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress></General></Domain></CycloneDDS>"

CMD ["bash"]