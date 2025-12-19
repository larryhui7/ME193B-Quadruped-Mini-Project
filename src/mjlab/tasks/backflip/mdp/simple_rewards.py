"""Simple rewards for backflip task."""

from __future__ import annotations

import math
import os
import numpy as np
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Load reference trajectory for motion imitation
_REF_DATA_PATH = os.path.join(os.path.dirname(__file__), "backflip_reference_phase.npz")
_REF_DATA = None
_REF_POSITIONS = None
_REF_POSITIONS_DEVICE = {}


def _load_reference():
  global _REF_DATA, _REF_POSITIONS
  if _REF_DATA is None and os.path.exists(_REF_DATA_PATH):
    _REF_DATA = np.load(_REF_DATA_PATH)
    _REF_POSITIONS = torch.from_numpy(_REF_DATA["positions"]).float()


def _get_ref_positions(device):
  """Get reference positions on the correct device."""
  _load_reference()
  if _REF_POSITIONS is None:
    return None
  if device not in _REF_POSITIONS_DEVICE:
    _REF_POSITIONS_DEVICE[device] = _REF_POSITIONS.to(device)
  return _REF_POSITIONS_DEVICE[device]


def track_height_simple(env, command_name, std=0.1, asset_cfg=_DEFAULT_ASSET_CFG):
  """Gaussian reward for tracking target height."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  current_height = asset.data.root_link_pos_w[:, 2]
  target_height = command[:, 1]

  height_error = torch.square(current_height - target_height)
  return torch.exp(-height_error / (std**2))


def track_x_simple(env, command_name, std=0.15, asset_cfg=_DEFAULT_ASSET_CFG):
  """Gaussian reward for tracking target x-position (2D trajectory)."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  current_x = asset.data.root_link_pos_w[:, 0]
  target_x = command[:, 4]

  x_error = torch.square(current_x - target_x)
  return torch.exp(-x_error / (std**2))


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


def pitch_velocity_reward(env, sensor_name, scale=10.0, asset_cfg=_DEFAULT_ASSET_CFG):
  """Reward backward pitch rate while airborne (encourages faster rotation)."""
  asset = env.scene[asset_cfg.name]
  sensor = env.scene[sensor_name]

  # Check if airborne (no foot contacts)
  any_contact = (sensor.data.found > 0).any(dim=-1)
  is_airborne = ~any_contact

  # Pitch rate: Y-axis angular velocity in body frame
  # Positive pitch rate = backward rotation (for backflip)
  pitch_rate = asset.data.root_link_ang_vel_b[:, 1]

  # Only reward positive (backward) pitch rate, and only while airborne
  reward = torch.clamp(pitch_rate / scale, 0.0, 1.0)
  return torch.where(is_airborne, reward, torch.zeros_like(reward))


def motion_imitation_reward(env, command_name, std=0.5, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Reward for imitating reference joint trajectory.

  Compares current joint positions to keyframe-defined reference at current phase.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  device = command.device

  # Get reference positions
  ref_positions = _get_ref_positions(device)
  if ref_positions is None:
    return torch.zeros(env.num_envs, device=device)

  # Get current phase (0 to 1)
  phase = command[:, 0]
  n_phases = ref_positions.shape[0]

  # Find nearest phase index
  phase_idx = (phase * (n_phases - 1)).long()
  phase_idx = torch.clamp(phase_idx, 0, n_phases - 1)

  # Get reference joint positions for current phase
  ref_joint_pos = ref_positions[phase_idx]  # (num_envs, 12)

  # Get current joint positions
  current_joint_pos = asset.data.joint_pos[:, :12]

  # Position error (sum of squared differences)
  pos_error = torch.sum(torch.square(current_joint_pos - ref_joint_pos), dim=1)

  # Gaussian reward
  return torch.exp(-pos_error / (std**2))
