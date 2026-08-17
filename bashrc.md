export MUJOCO_LIB_PATH=~/.mujoco/mujoco-3.2.3/lib
export LD_LIBRARY_PATH=/opt/ros/foxy/lib:/opt/ros/foxy/lib/x86_64-linux-gnu:$MUJOCO_LIB_PATH

alias brc="nano ~/.bashrc"
alias sbrc="source ~/.bashrc"

# >>> conda initialize >>>
__conda_setup="$('/home/noar/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/noar/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/noar/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/noar/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Isaac lab conda environment
alias isaaclab="conda activate isaaclab"

# Lancer un nouvel entraînement
alias g1train="python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless"

# Reprendre le dernier entraînement
alias g1resume="python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless --resume"

# --- PLAY & TELEOP ---
# Regarder l'IA évoluer (1 env)
alias g1play="python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --num_envs 1"

# Prendre le contrôle au clavier (1 env)
alias g1teleop="python scripts/rsl_rl/teleop.py --task Unitree-G1-29dof-Velocity --num_envs 1"

export ISAACSIM_PATH=~/isaacsim/_build/linux-x86_64/release

#Mjlab conda environment
alias mjlab="conda activate mjlab"

# Lancer un nouvel entraînement
alias g1mjtrain="python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs=4096 --agent.max-iterations 50000"

# Reprendre le dernier entraînement
alias g1mjresume="python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs=4096 --agent.resume True --agent.max-iterations 50000"

# --- PLAY & TELEOP ---
# Regarder l'IA évoluer (1 env)
alias g1mjplay='python scripts/play.py Mjlab-Velocity-Flat-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -n 1)'

# Prendre le contrôle au clavier (1 env)
alias g1mjteleop='python scripts/teleop.py Mjlab-Velocity-Flat-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -n 1)'

# SAME FOR ROUGH TERRAIN
alias g1mjtrainrough="python scripts/train.py Mjlab-Velocity-Rough-Unitree-G1 --env.scene.num-envs=1024 --agent.max-iterations 50000"

alias g1mjresumerough="python scripts/train.py Mjlab-Velocity-Rough-Unitree-G1 --env.scene.num-envs=1024 --agent.resume True --agent.max-iterations 50000"

alias g1mjplayrough='python scripts/play.py Mjlab-Velocity-Rough-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -n 1)'

alias g1mjteleoprough='python scripts/teleop.py Mjlab-Velocity-Rough-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -n 1)'