"""Reward functions for the backflip task.

Reward terms are designed to encourage:
1. Following the phase-based reference trajectory
2. Completing the full rotation
3. Safe takeoff and landing
4. Smooth, controlled motion
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# =============================================================================
# Primary Backflip Rewards
# =============================================================================


def track_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the target base height.

  Uses exponential reward shaping: exp(-error^2 / std^2)
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  current_height = asset.data.root_link_pos_w[:, 2]
  target_height = command[:, 1]
  height_error = torch.square(current_height - target_height)

  return torch.exp(-height_error / (std**2))


def track_pitch_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the target pitch angular velocity.

  Encourages the robot to rotate at the correct rate for the backflip.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Pitch velocity is rotation about y-axis in body frame
  current_pitch_vel = asset.data.root_link_ang_vel_b[:, 1]
  target_pitch_vel = command[:, 3]

  pitch_vel_error = torch.square(current_pitch_vel - target_pitch_vel)
  return torch.exp(-pitch_vel_error / (std**2))


def phase_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Reward for making progress through the flip phases.

  Returns the current phase value, encouraging advancement.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, 0]  # phase


def backflip_completion_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  bonus: float = 10.0,
) -> torch.Tensor:
  """Large bonus reward for completing the backflip.

  Awarded when phase reaches 1.0 and robot is approximately upright.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None

  phase = command[:, 0]
  completed = (phase >= 0.99).float()

  return completed * bonus


# =============================================================================
# Orientation Rewards
# =============================================================================


def upright_at_landing(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for being upright during the landing phase (φ > 0.8).

  Only active during the final landing phase to ensure proper recovery.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  phase = command[:, 0]
  projected_gravity = asset.data.projected_gravity_b

  # Uprightness: how aligned is gravity with the -z axis in body frame
  # When upright, projected_gravity_b should be [0, 0, -1]
  uprightness_error = torch.sum(projected_gravity[:, :2] ** 2, dim=1)

  # Only apply during landing phase
  landing_mask = (phase > 0.8).float()

  reward = torch.exp(-uprightness_error / (std**2))
  return reward * landing_mask


def rotation_consistency(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for consistent backward rotation (negative pitch velocity).

  Encourages the robot to rotate in the correct direction during the flip.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # Pitch velocity (rotation about y-axis)
  # For a backflip, we want negative pitch velocity (rotating backward)
  pitch_vel = asset.data.root_link_ang_vel_b[:, 1]

  # Reward for negative (backward) pitch velocity, clamped
  return torch.clamp(-pitch_vel / 10.0, 0.0, 1.0)


# =============================================================================
# Safety and Regularization Rewards
# =============================================================================


def air_time_during_flip(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  """Reward for being airborne during the middle phase of the flip.

  The robot should be in the air during φ ∈ [0.2, 0.8].
  """
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  phase = command[:, 0]
  assert sensor.data.found is not None
  in_contact = torch.any(sensor.data.found > 0, dim=1)

  # During mid-flip, reward being airborne
  mid_flip_mask = ((phase > 0.2) & (phase < 0.8)).float()
  airborne = (~in_contact).float()

  return mid_flip_mask * airborne


def ground_contact_at_landing(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  """Reward for having ground contact during landing phase.

  The robot should land on its feet during φ > 0.85.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  phase = command[:, 0]
  assert sensor.data.found is not None
  in_contact = torch.any(sensor.data.found > 0, dim=1)

  # During landing, reward ground contact
  landing_mask = (phase > 0.85).float()

  return landing_mask * in_contact.float()


def takeoff_impulse(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for upward velocity during takeoff phase.

  Encourages a strong jump at the beginning of the flip.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  phase = command[:, 0]

  # Vertical velocity in world frame
  vertical_vel = asset.data.root_link_lin_vel_w[:, 2]

  # During takeoff phase, reward upward velocity
  takeoff_mask = ((phase > 0.05) & (phase < 0.3)).float()
  upward_reward = torch.clamp(vertical_vel / 5.0, 0.0, 1.0)

  return takeoff_mask * upward_reward


def smooth_actions(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Penalize large action magnitudes for smoother motion."""
  action = env.action_manager.action
  return torch.sum(torch.square(action), dim=1)


def action_rate_penalty(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Penalize rapid changes in actions."""
  action = env.action_manager.action
  last_action = env.action_manager.prev_action

  action_diff = action - last_action
  return torch.sum(torch.square(action_diff), dim=1)


def joint_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive joint velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel
  return torch.sum(torch.square(joint_vel), dim=1)


def joint_torque_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive joint torques."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_torque = asset.data.applied_torque
  if joint_torque is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.sum(torch.square(joint_torque), dim=1)

