"""Observations for the backflip task.

Provides observations that allow the robot to understand:
- Its current motion state (velocities, orientation)
- Its progress through the backflip (phase)
- Relevant proprioceptive information
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def backflip_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Current phase of the backflip φ ∈ [0, 1]."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  # Return just the phase (first element)
  return command[:, 0:1]


def backflip_target_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Target base height for current phase."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, 1:2]


def backflip_target_pitch(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Target pitch angle for current phase."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, 2:3]


def backflip_target_pitch_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Target pitch angular velocity for current phase."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, 3:4]


def base_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Current base height above ground."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def base_pitch(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Current base pitch angle (rotation about y-axis in body frame).

  Computed from the projected gravity vector.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # Projected gravity gives us orientation relative to world
  proj_grav = asset.data.projected_gravity_b  # [num_envs, 3]

  # Pitch is the angle between gravity and vertical in the x-z plane
  # sin(pitch) = -proj_grav_x (forward component when tilted back)
  pitch = torch.asin(-proj_grav[:, 0:1])
  return pitch


def base_pitch_rate(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Current pitch angular velocity (rotation rate about y-axis)."""
  asset: Entity = env.scene[asset_cfg.name]
  # Angular velocity in body frame, y-component is pitch rate
  return asset.data.root_link_ang_vel_b[:, 1:2]


def base_orientation_quat(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Base orientation as quaternion (w, x, y, z)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_quat_w


def base_lin_vel_vertical(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Vertical component of base linear velocity in world frame."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w[:, 2:3]


def height_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Error between current height and target height."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  current_height = asset.data.root_link_pos_w[:, 2:3]
  target_height = command[:, 1:2]
  return target_height - current_height


def cumulative_pitch(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Estimate of cumulative pitch rotation (for tracking full rotation).

  This is a simple approximation using angular velocity integration.
  For a proper implementation, you'd track the unwrapped angle.
  """
  asset: Entity = env.scene[asset_cfg.name]
  # Return pitch angular velocity as a proxy (actual cumulative tracking
  # would require state in the command term)
  return asset.data.root_link_ang_vel_b[:, 1:2]

