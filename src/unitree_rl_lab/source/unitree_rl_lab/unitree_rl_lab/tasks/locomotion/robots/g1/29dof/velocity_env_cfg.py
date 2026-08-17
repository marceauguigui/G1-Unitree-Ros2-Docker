import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.5, noise_range=(-0.03, 0.03), noise_step=0.01),
    },
)

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=True,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),    
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0), 
                "y": (-1.0, 1.0), 
                "z": (0.0, 0.05),       # Le robot spawn entre 0 et 5 cm au-dessus du sol (petite chute)
                "roll": (-0.2, 0.2),    # Inclinaison latérale aléatoire (environ -11° à +11°)
                "pitch": (-0.2, 0.2),   # Inclinaison avant/arrière aléatoire
                "yaw": (-3.14, 3.14)    # Orientation de départ totalement aléatoire
            },
            "velocity_range": {
                "x": (-0.1, 0.1),       # Petite vitesse linéaire initiale pour le déstabiliser
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),    # Petite vitesse angulaire initiale
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    # Jambes + bassin : on garde une randomisation modérée -> point de départ stable pour marcher
    reset_robot_joints_lower = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", "waist_.*",
            ]),
            "position_range": (0.9, 1.1),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # Haut du corps : randomisation large, quasi pleine amplitude articulaire
    reset_robot_joints_upper = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*",
            ]),
            "position_range": (-1.8, 1.8),   # radians, offset absolu par rapport au default
            "velocity_range": (-0.2, 0.2),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 4.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.3, 0.3)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.6, 0.6)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # Plus le robot suit fidèlement la vitesse demandée, plus il gagne de points
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Récompense le suivi de la consigne de rotation sur lui-même
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.2) # Bonus constant pour l'inciter à ne pas abandonner/tomber

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-3.0) # Pénalise les sauts/secousses verticaux
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) # Pénalise le roulis et le tangage du torse
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.002) # Evite les mouvements sacadés
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) # Pénalise les accélérations brusques (vibrations)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01) # Sanctionne les changements d'ordres trop violents
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0) # Très forte pénalité si les moteurs butent en fin de course
    energy = RewTerm(func=mdp.energy, weight=-2e-5) # Pénalise la surconsommation électrique

    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_.*_joint",
    #                 ".*_elbow_joint",
    #                 ".*_wrist_.*",
    #             ],
    #         )
    #     },
    # )
    # Axe de swing avant/arrière : on laisse beaucoup de liberté ici
    joint_deviation_arms_swing = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,  # poids faible : juste un léger rappel, pas une contrainte
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint"])},
    )

    # Axes latéraux/rotation + poignets : on garde une contrainte plus stricte
    # pour éviter que le bras parte n'importe où ou tremble bizarrement
    joint_deviation_arms_other = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_.*",
        ])},
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    shoulder_pitch_range = RewTerm(
        func=mdp.joint_range_penalty, # Remarque le "mdp."
        weight=-5.0,  
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint"]),
            "bound": 0.5
        }
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=1.0,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # Pénalise l'effet "patinoire" (le pied ne doit pas glisser quand il est au sol)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    # Ajusté pour équilibrer : pousse le robot à lever le pied proprement pendant son pas
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    # Sanctionne si des parties non désirables du robot touchent le sol
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True) # 20s
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2}) # Bassin descend sous hauteur min
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7}) # Robot s'incline à plus de 45°


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0 # Durée d'une simu
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation # Le rendu graphique s'aligne sur les décisions du RL (50 Hz)
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        # En mode Play, on pousse directement le robot dans ses retranchements physiques maximaux (limit_ranges)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges