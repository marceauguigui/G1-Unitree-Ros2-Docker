#!/bin/bash

# ── MuJoCo / ROS2 ──
export MUJOCO_LIB_PATH="$HOME/.mujoco/mujoco-3.2.3/lib"
export LD_LIBRARY_PATH="/opt/ros/foxy/lib:/opt/ros/foxy/lib/x86_64-linux-gnu:$MUJOCO_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── General aliases ──
alias brc="nano ~/.bashrc"
alias sbrc="source ~/.bashrc"

# >>> conda initialize >>>
__conda_setup="$("$HOME/miniconda3/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# ══════════════════════════════════════════════════════════════════════════
# Isaac Lab (unitree_rl_lab)
# ══════════════════════════════════════════════════════════════════════════

export ISAACSIM_PATH="$HOME/isaacsim/_build/linux-x86_64/release"
export UNITREE_RL_LAB_DIR="$HOME/G1-Unitree-Ros2-Docker/src/unitree_rl_lab"

alias isaaclab="conda activate isaaclab"

# Start new training
alias g1train="cd $UNITREE_RL_LAB_DIR && python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless"

# Resume training
alias g1resume="cd $UNITREE_RL_LAB_DIR && python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --headless --resume"

# Observe training (1 env)
alias g1play="cd $UNITREE_RL_LAB_DIR && python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --num_envs 1"

# Control training (1 env)
alias g1teleop="cd $UNITREE_RL_LAB_DIR && python scripts/rsl_rl/teleop.py --task Unitree-G1-29dof-Velocity --num_envs 1"

# ══════════════════════════════════════════════════════════════════════════
# MJlab (unitree_rl_mjlab)
# ══════════════════════════════════════════════════════════════════════════

export UNITREE_RL_MJLAB_DIR="$HOME/G1-Unitree-Ros2-Docker/src/unitree_rl_mjlab"

alias mjlab="conda activate mjlab"

# Start new training
# (flat)
alias g1mjtrain="cd $UNITREE_RL_MJLAB_DIR && python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs=4096 --agent.max-iterations 50000"
# (rough)
alias g1mjtrainrough="cd $UNITREE_RL_MJLAB_DIR && python scripts/train.py Mjlab-Velocity-Rough-Unitree-G1 --env.scene.num-envs=1024 --agent.max-iterations 50000"
#(crawl flat)
alias g1mjtraincrawl="cd \$UNITREE_RL_MJLAB_DIR && python scripts/train.py Mjlab-Crawling-Unitree-G1 --env.scene.num-envs=4096 --agent.max-iterations 50000"

# Resume training
# (flat)
alias g1mjresume="cd $UNITREE_RL_MJLAB_DIR && python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs=4096 --agent.resume True --agent.max-iterations 50000"
# (rough)
alias g1mjresumerough="cd $UNITREE_RL_MJLAB_DIR && python scripts/train.py Mjlab-Velocity-Rough-Unitree-G1 --env.scene.num-envs=1024 --agent.resume True --agent.max-iterations 50000"
# (crawl flat)
alias g1mjresumecrawl="cd \$UNITREE_RL_MJLAB_DIR && python scripts/train.py Mjlab-Crawling-Unitree-G1 --env.scene.num-envs=4096 --agent.resume True --agent.max-iterations 50000"

# Observe training (1 env)
# (flat)
alias g1mjplay='cd $UNITREE_RL_MJLAB_DIR && python scripts/play.py Mjlab-Velocity-Flat-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -n 1)'
# (rough)
alias g1mjplayrough='cd $UNITREE_RL_MJLAB_DIR && python scripts/play.py Mjlab-Velocity-Rough-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -n 1)'
# (crawl flat)
alias g1mjplaycrawl='cd $UNITREE_RL_MJLAB_DIR && python scripts/play.py Mjlab-Crawling-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_crawling/*/model_*.pt | head -n 1)'


# Control training (1 env)
#(flat)
alias g1mjteleop='cd $UNITREE_RL_MJLAB_DIR && python scripts/teleop.py Mjlab-Velocity-Flat-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -n 1)'
# (rough)
alias g1mjteleoprough='cd $UNITREE_RL_MJLAB_DIR && python scripts/teleop.py Mjlab-Velocity-Rough-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_velocity/*/model_*.pt | head -n 1)'
# (crawl flat)
alias g1mjteleopcrawl='cd $UNITREE_RL_MJLAB_DIR && python scripts/teleop.py Mjlab-Crawling-Unitree-G1 --checkpoint-file $(ls -t logs/rsl_rl/g1_crawling/*/model_*.pt | head -n 1)'