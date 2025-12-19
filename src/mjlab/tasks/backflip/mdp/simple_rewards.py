"""Simple rewards for backflip task."""

from __future__ import annotations

import math
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
  """Gaussian reward for tracking target orientation via projected gravity."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  proj_grav = asset.data.projected_gravity_b
  current_grav_x = proj_grav[:, 0]
  current_grav_z = proj_grav[:, 2]

  target_grav_x = command[:, 2]
  target_grav_z = command[:, 3]

  grav_error = torch.square(current_grav_x - target_grav_x) + torch.square(current_grav_z - target_grav_z)
  return torch.exp(-grav_error / (std**2))


def action_rate_simple(env):
  """Penalize rapid action changes."""
  action = env.action_manager.action
  last_action = env.action_manager.prev_action
  return torch.sum(torch.square(action - last_action), dim=1)


def off_axis_simple(env, asset_cfg=_DEFAULT_ASSET_CFG):
  """Penalize roll and yaw (we want pure pitch rotation)."""
  asset = env.scene[asset_cfg.name]
  proj_grav = asset.data.projected_gravity_b
  roll_error = torch.abs(proj_grav[:, 1])
  yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])
  return roll_error + 0.5 * yaw_rate


def backflip_progress_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward phase-based backflip progress (must achieve phases in sequence)."""
  command_term = env.command_manager.get_term(command_name)

  # Get phase progress from command metrics (0.0 -> 0.1 -> 0.25 -> 0.5 -> 0.75 -> 1.0)
  phase_progress = command_term.metrics["max_rotation_progress"]

  # Reward is simply the phase progress value
  # 0.0 = nothing, 0.1 = crouched, 0.25 = took off, 0.5 = past vertical, 0.75 = inverted, 1.0 = landing
  return phase_progress


def max_height_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward for max height achieved during episode (encourages higher jumps)."""
  command_term = env.command_manager.get_term(command_name)
  max_height = command_term.metrics["max_height"]
  standing = command_term.cfg.standing_height
  peak = command_term.cfg.peak_height
  # Normalize: 0 at standing, 1 at peak
  return torch.clamp((max_height - standing) / (peak - standing), 0.0, 1.0)
