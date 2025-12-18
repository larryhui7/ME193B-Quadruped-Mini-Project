"""Termination conditions for the backflip task.

Implements success and failure conditions:
- Success: Completed the flip and landed upright
- Failure: Fell over, body contact, or exceeded time limit
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def backflip_completed(
  env: ManagerBasedRlEnv,
  command_name: str,
  uprightness_threshold: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Success termination: backflip completed and robot is upright.

  Triggers when:
  - Phase has reached 1.0
  - Robot is approximately upright (gravity aligned with -z)
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  phase = command[:, 0]
  projected_gravity = asset.data.projected_gravity_b

  # Check if phase is complete
  phase_complete = phase >= 0.99

  # Check if robot is upright (projected gravity close to [0, 0, -1])
  uprightness_error = torch.sum(projected_gravity[:, :2] ** 2, dim=1)
  is_upright = uprightness_error < uprightness_threshold

  return phase_complete & is_upright


def body_contact_during_flip(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  """Failure termination: body (non-foot) contact during mid-flip.

  Triggers when the robot's body touches the ground during the
  airborne phase of the flip (φ ∈ [0.2, 0.8]).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  phase = command[:, 0]
  assert sensor.data.found is not None

  # Check for any self-collision or body contact
  has_contact = torch.any(sensor.data.found > 0, dim=1)

  # Only terminate during mid-flip phase
  mid_flip = (phase > 0.2) & (phase < 0.8)

  return mid_flip & has_contact


def height_too_low(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_height: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Failure termination: robot too low during mid-flip.

  Triggers when the robot's base height falls below minimum
  during the airborne phase (φ ∈ [0.15, 0.75]).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  phase = command[:, 0]
  height = asset.data.root_link_pos_w[:, 2]

  # Height check only during mid-flip
  mid_flip = (phase > 0.15) & (phase < 0.75)
  too_low = height < min_height

  return mid_flip & too_low


def bad_landing_orientation(
  env: ManagerBasedRlEnv,
  command_name: str,
  limit_angle: float = 1.2,  # ~70 degrees
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Failure termination: robot not upright at end of flip.

  Triggers when the robot has completed the phase but is not
  properly oriented for landing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  phase = command[:, 0]
  projected_gravity = asset.data.projected_gravity_b

  # Orientation angle from upright
  # When upright, projected_gravity_b should be [0, 0, -1]
  # so -projected_gravity_b[:, 2] should be close to 1
  orientation_angle = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))

  # Only check during landing phase
  landing_phase = phase > 0.85
  bad_orientation = orientation_angle > limit_angle

  return landing_phase & bad_orientation


def no_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_phase_by_time: float = 0.5,
  time_threshold: float = 1.0,
) -> torch.Tensor:
  """Failure termination: not making enough progress.

  Triggers when the robot hasn't advanced far enough in the flip
  after a certain amount of time has elapsed.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None

  phase = command[:, 0]
  episode_length = env.episode_length_buf * env.step_dt

  # Expected phase based on time elapsed
  expected_phase = episode_length / time_threshold * min_phase_by_time

  # Fail if significantly behind expected progress
  return (episode_length > time_threshold) & (phase < expected_phase * 0.5)


def excessive_rotation(
  env: ManagerBasedRlEnv,
  max_roll: float = 1.0,  # ~57 degrees
  max_yaw_rate: float = 5.0,  # rad/s
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Failure termination: excessive roll or yaw during flip.

  The backflip should be primarily pitch rotation. Excessive roll
  or yaw indicates loss of control.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Check roll from projected gravity
  proj_grav = asset.data.projected_gravity_b
  roll_error = torch.abs(proj_grav[:, 1])  # y-component indicates roll

  # Check yaw rate
  yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])

  excessive_roll = roll_error > max_roll
  excessive_yaw = yaw_rate > max_yaw_rate

  return excessive_roll | excessive_yaw

