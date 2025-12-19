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
  """Reward actual backflip progress with milestone bonuses."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  time_phase = command[:, 0]

  proj_grav = asset.data.projected_gravity_b
  grav_x = proj_grav[:, 0]
  grav_z = proj_grav[:, 2]

  # Rotation angle from upright (0 at start, π at inverted, 2π back to upright)
  rotation_angle = torch.atan2(grav_x, -grav_z)
  rotation_progress = torch.where(
    rotation_angle >= 0,
    rotation_angle / (2 * math.pi),
    (rotation_angle + 2 * math.pi) / (2 * math.pi)
  )

  current_height = asset.data.root_link_pos_w[:, 2]

  # Milestones
  crouch_achieved = (current_height < 0.25).float() * (time_phase < 0.2).float()
  jump_achieved = (current_height > 0.5).float() * (time_phase > 0.2).float()
  past_vertical = (rotation_progress > 0.25).float()
  inverted = ((rotation_progress > 0.4) & (rotation_progress < 0.6)).float()
  past_inverted = (rotation_progress > 0.5).float()

  milestone_bonus = (
    0.1 * crouch_achieved +
    0.2 * jump_achieved +
    0.3 * past_vertical +
    0.5 * inverted +
    1.0 * past_inverted
  )

  return rotation_progress + milestone_bonus
