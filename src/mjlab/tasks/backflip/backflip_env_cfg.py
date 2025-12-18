import math
from copy import deepcopy

from mjlab.entity.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.backflip import mdp
from mjlab.tasks.backflip.mdp import BackflipPhaseCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Scene configuration for backflip (flat ground, no terrain)
SCENE_CFG = SceneCfg(
    terrain=None,
    num_envs=1,
    extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="",  # Override in robot cfg
    distance=4.0,
    elevation=-15.0,
    azimuth=90.0,
)

SIM_CFG = SimulationCfg(
    nconmax=50,
    njmax=400,
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
    ),
)


def create_backflip_env_cfg(
    robot_cfg: EntityCfg,
    action_scale: float | dict[str, float],
    viewer_body_name: str,
    feet_sensor_cfg: ContactSensorCfg,
    self_collision_sensor_cfg: ContactSensorCfg,
    standing_height: float = 0.35,
    peak_height: float = 0.7,
    flip_duration: float = 0.8,
) -> ManagerBasedRlEnvCfg:
    """Create a backflip task configuration.

    Args:
      robot_cfg: Robot configuration.
      action_scale: Action scaling factor(s).
      viewer_body_name: Body for camera tracking.
      feet_sensor_cfg: Contact sensor config for feet-ground contact.
      self_collision_sensor_cfg: Contact sensor for body collision detection.
      standing_height: Robot standing height (m).
      peak_height: Target peak height during flip (m).
      flip_duration: Duration of the backflip (s).

    Returns:
      Complete ManagerBasedRlEnvCfg for backflip task.
    """
    scene = deepcopy(SCENE_CFG)

    scene.entities = {"robot": robot_cfg}

    scene.sensors = (
        feet_sensor_cfg,
        self_collision_sensor_cfg,
    )

    # Enable curriculum mode for terrain generator.
    if scene.terrain is not None and scene.terrain.terrain_generator is not None:
        scene.terrain.terrain_generator.curriculum = True

    viewer = deepcopy(VIEWER_CONFIG)
    viewer.body_name = viewer_body_name

    # Actions: joint position control
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=action_scale,
            use_default_offset=True,
        )
    }

    # Commands: phase-based backflip trajectory
    commands: dict[str, CommandTermCfg] = {
        "backflip": BackflipPhaseCommandCfg(
            asset_name="robot",
            # No resampling during episode
            resampling_time_range=(100.0, 100.0),
            flip_duration=flip_duration,
            standing_height=standing_height,
            peak_height=peak_height,
        )
    }

    # ---------------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------------
    policy_terms: dict[str, ObservationTermCfg] = {
        # Phase and target information
        "backflip_phase": ObservationTermCfg(
            func=mdp.backflip_phase,
            params={"command_name": "backflip"},
        ),
        "target_height": ObservationTermCfg(
            func=mdp.backflip_target_height,
            params={"command_name": "backflip"},
        ),
        "target_pitch_vel": ObservationTermCfg(
            func=mdp.backflip_target_pitch_vel,
            params={"command_name": "backflip"},
        ),
        # Current state
        "base_height": ObservationTermCfg(
            func=mdp.base_height,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "last_action": ObservationTermCfg(
            func=mdp.last_action,
        ),
    }

    critic_terms = {
        **policy_terms,
        # Critic gets clean target pitch for better value estimation
        "target_pitch": ObservationTermCfg(
            func=mdp.backflip_target_pitch,
            params={"command_name": "backflip"},
        ),
        "base_orientation_quat": ObservationTermCfg(
            func=mdp.base_orientation_quat,
        ),
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # ---------------------------------------------------------------------------
    # Events
    # ---------------------------------------------------------------------------
    events = {
        # Reset to standing pose
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    # ---------------------------------------------------------------------------
    # Rewards
    # ---------------------------------------------------------------------------
    rewards = {
        # Primary tracking rewards
        "track_height": RewardTermCfg(
            func=mdp.track_height,
            weight=2.0,
            params={
                "command_name": "backflip",
                "std": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "track_pitch_velocity": RewardTermCfg(
            func=mdp.track_pitch_velocity,
            weight=3.0,
            params={
                "command_name": "backflip",
                "std": 2.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Phase progress
        "phase_progress": RewardTermCfg(
            func=mdp.phase_progress,
            weight=1.0,
            params={"command_name": "backflip"},
        ),
        # Completion bonus
        "completion_bonus": RewardTermCfg(
            func=mdp.backflip_completion_bonus,
            weight=1.0,
            params={"command_name": "backflip", "bonus": 20.0},
        ),
        # Landing orientation
        "upright_at_landing": RewardTermCfg(
            func=mdp.upright_at_landing,
            weight=2.0,
            params={
                "command_name": "backflip",
                "std": 0.3,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Takeoff reward
        "takeoff_impulse": RewardTermCfg(
            func=mdp.takeoff_impulse,
            weight=1.5,
            params={
                "command_name": "backflip",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Rotation consistency
        "rotation_consistency": RewardTermCfg(
            func=mdp.rotation_consistency,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        # Regularization
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_penalty,
            weight=-0.01,
        ),
        "joint_velocity": RewardTermCfg(
            func=mdp.joint_velocity_penalty,
            weight=-0.0001,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }

    # ---------------------------------------------------------------------------
    # Terminations
    # ---------------------------------------------------------------------------
    terminations = {
        # Success
        "backflip_completed": TerminationTermCfg(
            func=mdp.backflip_completed,
            time_out=True,  # Treat as success (not failure)
            params={
                "command_name": "backflip",
                "uprightness_threshold": 0.4,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Timeout
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        # Failures
        "height_too_low": TerminationTermCfg(
            func=mdp.height_too_low,
            time_out=False,
            params={
                "command_name": "backflip",
                "min_height": 0.12,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "bad_landing": TerminationTermCfg(
            func=mdp.bad_landing_orientation,
            time_out=False,
            params={
                "command_name": "backflip",
                "limit_angle": 1.3,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "excessive_rotation": TerminationTermCfg(
            func=mdp.excessive_rotation,
            time_out=False,
            params={
                "max_roll": 1.2,
                "max_yaw_rate": 6.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=SIM_CFG,
        viewer=viewer,
        decimation=2,  # Lower decimation for faster control during dynamic motion
        episode_length_s=3.0,  # Short episodes for the flip
    )
