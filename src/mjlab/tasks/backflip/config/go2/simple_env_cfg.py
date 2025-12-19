"""Simplified Go2 backflip environment - minimal rewards following PDF guidance.

Key design choices (from PDF Section 3b):
1. Command outputs simple reference quantities (height + orientation)
2. Coarse, hand-crafted trajectory shapes are sufficient
3. Small number of straightforward reward terms
4. Phase variable parameterizes progress through the flip
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)

# Import simple components
from mjlab.tasks.backflip.mdp.simple_backflip_command import SimpleBackflipCommandCfg
from mjlab.tasks.backflip.mdp import simple_rewards
from mjlab.tasks.backflip import mdp  # For standard observations and events


SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(terrain_type="plane"),
  num_envs=1,
  extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="trunk",
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


# Simple env config
scene = deepcopy(SCENE_CFG)
scene.entities = {"robot": get_go2_robot_cfg()}

actions = {
  "joint_pos": JointPositionActionCfg(
    asset_name="robot",
    actuator_names=(".*",),
    scale=GO2_ACTION_SCALE,
    use_default_offset=True,
  )
}

# Simple time-based command with crouch-then-jump trajectory
commands = {
  "backflip": SimpleBackflipCommandCfg(
    asset_name="robot",
    resampling_time_range=(100.0, 100.0),  # Never resample
    flip_duration=1.0,  # 1 second total (including crouch)
    standing_height=0.35,
    crouch_height=0.18,  # Deeper crouch for more explosive power
    peak_height=0.90,  # Higher peak = more air time (~0.72s, needs ~8.7 rad/s)
  )
}

# Observations following PDF Section 1f guidance
# Policy sees: command info + proprioception (with noise)
policy_terms = {
  # Command information (what to track)
  "phase": ObservationTermCfg(
    func=mdp.backflip_phase,
    params={"command_name": "backflip"},
  ),
  "target_height": ObservationTermCfg(
    func=mdp.backflip_target_height,
    params={"command_name": "backflip"},
  ),
  "target_grav_x": ObservationTermCfg(
    func=mdp.backflip_target_grav_x,
    params={"command_name": "backflip"},
  ),
  "target_grav_z": ObservationTermCfg(
    func=mdp.backflip_target_grav_z,
    params={"command_name": "backflip"},
  ),
  # Proprioception (with noise for robustness)
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

# Critic gets clean observations (asymmetric critic training)
critic_terms = {
  **policy_terms,
  "base_orientation_quat": ObservationTermCfg(
    func=mdp.base_orientation_quat,
  ),
}
# Remove noise from critic observations
for key in critic_terms:
  if hasattr(critic_terms[key], 'noise'):
    critic_terms[key] = ObservationTermCfg(
      func=critic_terms[key].func,
      params=critic_terms[key].params if hasattr(critic_terms[key], 'params') else {},
    )

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

# Simple reset events
events = {
  "reset_base": EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (0.0, 0.0)},
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

# MINIMAL REWARDS - following PDF "small number of straightforward reward terms"
rewards = {
  # Track height trajectory (Gaussian)
  "track_height": RewardTermCfg(
    func=simple_rewards.track_height_simple,
    weight=2.0,
    params={
      "command_name": "backflip",
      "std": 0.1,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  ),
  # Track orientation via projected gravity (Gaussian)
  "track_orientation": RewardTermCfg(
    func=simple_rewards.track_orientation_simple,
    weight=3.0,  # Slightly higher to encourage rotation
    params={
      "command_name": "backflip",
      "std": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  ),
  # Penalize off-axis rotation (we want pure pitch, not roll/yaw)
  "off_axis": RewardTermCfg(
    func=simple_rewards.off_axis_simple,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot")},
  ),
  # Small action smoothness penalty
  "action_rate": RewardTermCfg(
    func=simple_rewards.action_rate_simple,
    weight=-0.01,
  ),
}

# Simple termination - just timeout
terminations = {
  "time_out": TerminationTermCfg(
    func=mdp.time_out,
    time_out=True,
  ),
}

# Create the config
SIMPLE_GO2_BACKFLIP_ENV_CFG = ManagerBasedRlEnvCfg(
  scene=scene,
  observations=observations,
  actions=actions,
  commands=commands,
  rewards=rewards,
  terminations=terminations,
  events=events,
  sim=SIM_CFG,
  viewer=VIEWER_CONFIG,
  decimation=2,
  episode_length_s=1.5,  # 1.5s episodes (1s flip + buffer)
)
