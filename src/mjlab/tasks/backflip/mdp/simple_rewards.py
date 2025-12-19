"""Simple rewards for backflip task - just tracking height and orientation."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_height_simple(env, command_name, std=0.1, asset_cfg=_DEFAULT_ASSET_CFG):
  """Gaussian reward for tracking target height."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  current_height = asset.data.root_link_pos_w[:, 2]
  target_height = command[:, 1]

  height_error = torch.square(current_height - target_height)
  return torch.exp(-height_error / (std**2))


def track_orientation_simple(env, command_name, std=0.3, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Gaussian reward for tracking target orientation via projected gravity.

  This avoids angle wraparound issues by tracking the gravity vector directly.
  The projected gravity naturally traces a circle during rotation.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  # Current projected gravity (body frame)
  proj_grav = asset.data.projected_gravity_b
  current_grav_x = proj_grav[:, 0]
  current_grav_z = proj_grav[:, 2]

  # Target gravity from command
  target_grav_x = command[:, 2]
  target_grav_z = command[:, 3]

  # Compute error (Euclidean distance in gravity space)
  grav_error = torch.square(current_grav_x - target_grav_x) + torch.square(current_grav_z - target_grav_z)

  return torch.exp(-grav_error / (std**2))


def action_rate_simple(env):
  """Penalize rapid changes in actions."""
  action = env.action_manager.action
  last_action = env.action_manager.prev_action
  action_diff = action - last_action
  return torch.sum(torch.square(action_diff), dim=1)


def off_axis_simple(env, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Penalize off-axis rotation (roll and yaw).
  We only want pitch rotation for a backflip.
  """
  asset = env.scene[asset_cfg.name]

  # Projected gravity y-component indicates roll
  proj_grav = asset.data.projected_gravity_b
  roll_error = torch.abs(proj_grav[:, 1])

  # Yaw rate
  yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])

  return roll_error + 0.5 * yaw_rate
