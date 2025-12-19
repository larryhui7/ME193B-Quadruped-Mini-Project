"""Simple rewards for backflip task - just tracking + basic regularization."""

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


def track_pitch_simple(env, command_name, std=0.5, asset_cfg=_DEFAULT_ASSET_CFG):
  """Gaussian reward for tracking target pitch angle."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  # Get current pitch from quaternion
  quat = asset.data.root_link_quat_w
  sinp = 2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1])
  sinp = torch.clamp(sinp, -1.0, 1.0)
  current_pitch = torch.asin(sinp)

  # Target pitch (wrapped to [-π, π] for comparison)
  target_pitch = command[:, 2]
  # Wrap target pitch to [-π, π]
  target_pitch_wrapped = torch.remainder(target_pitch + math.pi, 2 * math.pi) - math.pi

  # Compute angular error (handle wraparound)
  pitch_error = current_pitch - target_pitch_wrapped
  pitch_error = torch.remainder(pitch_error + math.pi, 2 * math.pi) - math.pi

  return torch.exp(-torch.square(pitch_error) / (std**2))


def action_rate_simple(env):
  """Penalize rapid changes in actions."""
  action = env.action_manager.action
  last_action = env.action_manager.prev_action
  action_diff = action - last_action
  return torch.sum(torch.square(action_diff), dim=1)


def upright_bonus(env, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward being upright (gravity aligned with -z in body frame)."""
  asset = env.scene[asset_cfg.name]
  proj_grav = asset.data.projected_gravity_b

  # Perfect upright: proj_grav = [0, 0, -1]
  # Reward based on how close z-component is to -1
  upright_reward = torch.clamp(-proj_grav[:, 2], 0.0, 1.0)
  return upright_reward
