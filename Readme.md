# G1-Unitree-Ros2-Docker

RL locomotion pipeline for the Unitree G1 humanoid: training (mjlab and/or
Isaac Lab), sim2sim validation, and sim2real deployment via ROS2.

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Training](#training)
- [Sim2sim testing](#sim2sim-testing)
- [Exporting a policy for deployment](#exporting-a-policy-for-deployment)
- [Real robot deployment](#real-robot-deployment)
- [Safety protocol](#safety-protocol)
- [Known issues / open items](#known-issues--open-items)
- [Troubleshooting](#troubleshooting)

## Overview

Two independent training stacks are supported, each with its own conda
environment and its own extension package:

| Stack | Framework | Extension package | Conda env |
|---|---|---|---|
| mjlab | [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp, GPU) | `unitree_rl_mjlab` | `mjlab` |
| Isaac Lab | [IsaacLab](https://github.com/isaac-sim/IsaacLab) (Isaac Sim) | `unitree_rl_lab` | `isaaclab` |

Both produce a TorchScript `.pt` policy that gets deployed the same way: via
`g1_ros2_main.py`, a ROS2 node that talks to the robot (real or simulated)
over `/lowstate` and `/lowcmd`.

Two ways to validate a policy before the real robot:
1. **Native sim2sim** (`teleop.py`, mjlab only) — runs the exact training
   environment, guaranteed-correct observations, fastest to iterate on.
2. **ROS2 sim2sim** (`unitree_mujoco.py` + `g1_ros2_main.py`) — runs the same
   ROS2 pipeline used on the real robot, against a simulated G1 in MuJoCo.
   Slower to set up, but validates the actual deployment code path.

## Repository structure

```
G1-Unitree-Ros2-Docker/
├── setup_env.sh                  # source this for aliases + env vars
├── src/
│   ├── unitree_rl_mjlab/         # mjlab extension: G1 task definitions
│   │   ├── scripts/               train.py, play.py, teleop.py
│   │   ├── src/tasks/              velocity env configs
│   │   ├── src/assets/robots/      g1_constants.py (Kp/Kd/action scale/actuator groups)
│   │   └── logs/rsl_rl/            training runs + checkpoints
│   ├── unitree_rl_lab/            # Isaac Lab extension: G1 task definitions
│   │   ├── scripts/rsl_rl/         train.py, play.py, teleop.py
│   │   └── source/unitree_rl_lab/  the actual installable package (pip install -e . HERE)
│   ├── unitree_model/             # USD assets for Isaac Lab (G1/H1/Go2/...)
│   ├── unitree_mujoco/            # MuJoCo sim2sim bridge (unitree_sdk2py_bridge.py, unitree_mujoco.py)
│   │   └── simulate_python/
│   ├── unitree_ros2/              # ROS2 workspace
│   │   └── ros2_ws/src/g1_controller/
│   │       ├── g1_controller/
│   │       │   ├── g1_ros2_main.py       # main real-time controller (LOW/HIGH/RL/STOP)
│   │       │   ├── g1_config.py          # Kp/Kd/DEFAULT_DOF_POS/ACTION_SCALE/poses/sequences
│   │       │   ├── real_g1_bridge.py     # ROS2<->DDS bridge, real robot only
│   │       │   └── ps5_bridge.py         # PS5 controller -> /command
│   │       └── launch/g1.launch.py       # mode:=sim | mode:=real
│   ├── unitree_sdk2/              # Unitree SDK (C++)
│   └── unitree_sdk2_python/       # Unitree SDK (Python) — unitree_sdk2py
```

## Installation

### 1. Miniconda

Install miniconda if not already present. The conda init block lives in
`setup_env.sh` — do not let the installer add a second one to `~/.bashrc`
(see the warning at the top of that file).

### 2. mjlab stack

```bash
conda create -n mjlab python=3.11
conda activate mjlab
cd src/unitree_rl_mjlab
pip install -e .
```

⚠️ `mujoco` / `mujoco_warp` versions must match what `unitree_rl_mjlab`'s
`setup.py` pins exactly (currently `mujoco-warp==3.5.0`, which requires
`mujoco<3.8.0` — newer `mujoco` removed the `mjENBL_MULTICCD` enum that this
`mujoco_warp` version expects). Check `setup.py` for the current pin before
assuming versions — see [Troubleshooting](#troubleshooting).

### 3. Isaac Lab stack

```bash
conda create -n isaaclab python=3.11
conda activate isaaclab
# Install IsaacLab itself first (separate large repo, see IsaacLab's own docs)
cd src/unitree_rl_lab/source/unitree_rl_lab
pip install -e .
```

USD assets (`src/unitree_model/`) must additionally be reachable at
`~/unitree_model` (a path currently hardcoded in the task config — see
[Known issues](#known-issues--open-items)). Either symlink it:
```bash
ln -s ~/G1-Unitree-Ros2-Docker/src/unitree_model ~/unitree_model
```
or patch the hardcoded path directly in the config.

### 4. ROS2 workspace (Docker)

```bash
cd docker && docker compose up -d
docker compose exec ros2_foxy bash
cd /ros2_ws
colcon build --packages-select g1_controller
source install/setup.bash
```
⚠️ The controller (Ethernet) connecting to the real G1 needs a static IP on
the correct interface (e.g. `192.168.123.x` range) — see
[Real robot deployment](#real-robot-deployment). Confirm the interface name
with `ls /sys/class/net/` — it is **not** guaranteed to be the same across
machines (e.g. `eno1` vs a USB adapter's `enp1s0`).

### 5. Environment script

```bash
echo 'source ~/G1-Unitree-Ros2-Docker/setup_env.sh' >> ~/.bashrc
source ~/.bashrc
```
Gives you: `mjlab` / `isaaclab` (conda activate shortcuts), `g1train` /
`g1mjtrain` and friends (see `setup_env.sh` for the full alias list).

## Training

### mjlab (recommended — lighter, faster to iterate)

```bash
mjlab   # conda activate mjlab
g1mjtrain          # flat terrain
g1mjtrainrough     # rough terrain (adds height_scan — see note below)
g1mjresume         # resume the latest run
```

⚠️ **Rough terrain requires `height_scan` (187-dim terrain raycast), which
has no real-hardware equivalent.** Do not deploy a rough-trained policy to
the real robot. Train `Mjlab-Velocity-Flat-Unitree-G1` for anything intended
for real deployment.

### Isaac Lab

```bash
isaaclab   # conda activate isaaclab
g1train
g1resume
```

## Sim2sim testing

### Native mjlab (`teleop.py`) — fastest, most reliable

```bash
mjlab
g1mjteleop   # or g1mjteleoprough
```
Runs the exact training environment — observations are guaranteed correct.
Controls (gamepad, PS5 mapping):
- **PS**: STOP | **OPTIONS/SHARE**: cycle mode (LOW↔HIGH↔RL)
- **HIGH**: Cross=squat_workout, Circle=coucou, Triangle=stand, Square=chair, R2=squat
- **LOW**: L1/R1=select joint, right stick vertical=nudge
- **RL**: left stick=vx/vy, right stick horizontal=yaw

Run with `TELEOP_CALIBRATE=1 g1mjteleop` once on a new controller to verify
button indices match (SDL button numbering isn't guaranteed to match ROS2's
`joy_node`).

### ROS2 + unitree_mujoco — validates the actual deployment code path

```bash
# Terminal 1 (inside the ROS2 Docker container):
cd ~/G1-Unitree-Ros2-Docker/src/unitree_ros2/ros2_ws
source install/setup.bash
ros2 launch g1_controller g1.launch.py mode:=sim
```
This launches `unitree_mujoco.py` (simulator, publishes `/base_lin_vel` from
a native local-frame velocimeter), `g1_ros2_main.py`, `ps5_bridge.py`, and
`joy_node`. Control via `/command` (String topic), same PS5 mapping as above.

Send commands manually if needed:
```bash
ros2 topic pub /command std_msgs/String "data: 'mode RL'" --once
ros2 topic pub /command std_msgs/String "data: 'vel 0.3 0 0'" --once
```

## Exporting a policy for deployment

⚠️ **Critical**: the export script MUST bake in observation normalization
(`obs_mean`/`obs_std` from the training checkpoint's `EmpiricalNormalization`
layer). The original export script only kept 2D tensors with `"weight"` in
their name, silently dropping the 1D normalizer buffers — this produced a
policy that looked structurally fine but output wildly extreme, erratic
actions when fed real-scale (unnormalized) observations. Confirmed root
cause of multiple real-robot instability episodes. Verify the export script
wraps the MLP with `(x - obs_mean) / obs_std` before the forward pass, and
that `input_dim` matches expectations:
- Flat, no waist lock: 99 dims
- Rough (height_scan): 286 dims — **do not deploy to real hardware**

Exported policies go to `g1_exported_policies/mjpolicy_N.pt`, referenced by
`policy_path` in `g1_ros2_main.py` (update manually after each export).

## Real robot deployment

### Architecture

```
G1 firmware (DDS, rt/lowstate, rt/lowcmd)
       │  (ROS2 <-> DDS wire-compatible via unitree_hg — no explicit bridge needed for these two)
       ▼
real_g1_bridge.py  ──publishes──▶  /lowstate (ROS2)
       │  also: releases the robot's internal "sport mode" controller
       │  (MotionSwitcherClient.ReleaseMode()) before g1_ros2_main takes over
       ▼
g1_ros2_main.py  ──publishes──▶  /lowcmd (ROS2) ──▶ rt/lowcmd (DDS, wire-compatible)
       ▲
ps5_bridge.py  ──publishes──▶  /command (String)
```

Launch:
```bash
ros2 launch g1_controller g1.launch.py mode:=real
```

### Control modes (`/command` topic, String)

| Command | Effect |
|---|---|
| `stop` | Emergency: zero stiffness, pure damping |
| `mode STOP\|LOW\|HIGH\|RL` | Switch control mode |
| `joint <name/idx> <delta>` | Nudge one joint (LOW mode only) |
| `<pose name>` (e.g. `stand`, `squat`, `chair`) | Move to a fixed pose (HIGH mode) |
| `<sequence name>` (e.g. `coucou`, `squat_workout`) | Play a predefined sequence (HIGH mode) |
| `vel <vx> <vy> <vyaw>` | Set velocity command (RL mode) |

Node always boots in a safe mode (currently `LOW` — see
[Known issues](#known-issues--open-items) regarding switching this to `STOP`
before real-hardware use).

## Safety protocol

**Never skip steps. Always have the e-stop in hand, never set down.**

1. Robot on the gantry, weight fully supported (not just "feet grazing the
   floor" — that is not representative of training conditions and produces
   misleading test results).
2. `mode HIGH` → verify smooth interpolation to `stand`.
3. `mode RL`, `vel 0 0 0`, still suspended → verify no violent oscillation.
4. Progressively lower the gantry to transfer weight.
5. Only then, small velocity commands.
6. Fall detection (`roll`/`pitch` > 30°) currently only **logs** — the line
   that actually switches to `STOP` is commented out in
   `g1_ros2_main.py::control_loop_callback`. **Uncomment it before any
   unsupervised real-hardware test.**

## Known issues / open items

- **`waist_roll` / `waist_pitch` locked at the firmware level.** Mechanical
  unlock done, remote-controller 3-DOF gesture done, but low-level `LowCmd`
  commands to these two joints are silently ignored. `LocoClient.GetFsmId()`
  consistently returns error `3102` regardless of call sequence tested
  (`Damp`/`Start`/`WaitLeaseApplied`, before/after `ReleaseMode()`). No
  physical movement observed from any `LocoClient` call. **Reported to
  Unitree support, unresolved.** Current policies are trained with these two
  DOF's action scale forced to `0.0` (`g1_constants.py`,
  `G1_ACTION_SCALE["waist_roll_joint"/"waist_pitch_joint"] = 0.0`) — armature/
  stiffness/damping are left intact so the joint stays physically rigid in
  simulation, matching the real lock.
- **`rt/sportmodestate` never publishes on this robot.** Confirmed via raw
  DDS subscription inside `real_g1_bridge.py`'s own process (same process
  that successfully handles `rt/lowstate`), across multiple `LocoClient`
  call sequences. `base_lin_vel` currently has no real-robot source and
  reads as `[0, 0, 0]` in `mode:=real` — the `/sportstate` relay path was
  removed from `g1_ros2_main.py` pending a resolution from Unitree support
  or an alternative estimation source.
- **`~/unitree_model` hardcoded path** in Isaac Lab task config, doesn't
  match the actual repo location — symlink or patch required (see
  installation step 3).
- **Default boot mode is `LOW`, not `STOP`**, in `g1_ros2_main.py` — a
  leftover from debugging the mode-switch phantom-target bug. Switch back to
  `STOP` before treating any real-hardware session as routine (not just
  first contact).

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'` / `'unitree_rl_lab'` /
`'unitree_rl_mjlab'`** — the extension package wasn't installed in the
active conda env. Find its `setup.py`/`pyproject.toml` (may be nested, e.g.
`source/unitree_rl_lab/` for Isaac Lab) and `pip install -e .` from there,
in the correct conda environment.

**`AttributeError: 'NoneType' object has no attribute '_ref'` (DDS)** —
CycloneDDS's participant isn't ready yet. A fixed `time.sleep()` after
`ChannelFactoryInitialize()` is not reliable; use a retry loop instead (see
`real_g1_bridge.py::_create_dds_client_with_retry`).

**`AttributeError: ... has no attribute 'mjENBL_MULTICCD'`** — `mujoco` /
`mujoco_warp` version mismatch. `mujoco>=3.8.0` removed this enable-bit
(multiccd became default-on, moved to `mjDSBL_MULTICCD`). Match the exact
pin in `unitree_rl_mjlab/setup.py`.

**`does not match an available interface` (DDS init)** — the network
interface has no IPv4 assigned. `ifconfig <iface>` to check; assign a static
IP in the robot's subnet if missing.

**`FileNotFoundError: USD file not found`** (Isaac Lab) — see the
`~/unitree_model` known issue above.

**Real robot "does something wrong / jerks" in RL mode** — check, in order:
1. Is `ACTION_SCALE` in `g1_config.py` (deployment) actually consistent with
   `G1_ACTION_SCALE` in `g1_constants.py` (training)? These are two separate
   files that must be kept in sync manually.
2. Was the exported policy normalized correctly (see
   [Exporting a policy](#exporting-a-policy-for-deployment))?
3. Is fall detection silently letting an unstable episode run to completion
   (log-only, not enforcing STOP)?
4. Compare `[obs]` debug logs (enable `PRINT_OBS_DEBUG` in `g1_ros2_main.py`)
   against a known-good `teleop.py` run, term by term.