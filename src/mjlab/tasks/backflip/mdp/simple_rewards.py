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
  # NEGATIVE pitch rate = backward rotation (nose up first = backflip)
  # Right-hand rule: Y points left, negative rotation = nose goes up
  pitch_rate = asset.data.root_link_ang_vel_b[:, 1]

  # Reward NEGATIVE pitch rate (backflip direction), only while airborne
  reward = torch.clamp(-pitch_rate / scale, 0.0, 1.0)
  return torch.where(is_airborne, reward, torch.zeros_like(reward))


def upward_velocity_reward(env, command_name, scale=2.0, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Reward upward velocity during takeoff phase (0.15-0.40).

  This provides gradient toward jumping - the robot learns that
  pushing up with its legs leads to reward.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  z_vel = asset.data.root_link_lin_vel_w[:, 2]

  # Only reward during takeoff phase (0.15 to 0.40)
  in_takeoff = (phase >= 0.15) & (phase < 0.40)

  # Reward positive z velocity (going up)
  reward = torch.clamp(z_vel / scale, 0.0, 1.0)
  return torch.where(in_takeoff, reward, torch.zeros_like(reward))


def airborne_bonus(env, sensor_name, min_height=0.40, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Bonus for being properly airborne (feet off ground AND body elevated).

  Requires both:
  1. All feet off the ground
  2. Body height above min_height (prevents sitting-on-butt hack)
  """
  asset = env.scene[asset_cfg.name]
  sensor = env.scene[sensor_name]

  # Check if any foot is in contact
  any_contact = (sensor.data.found > 0).any(dim=-1)
  feet_airborne = ~any_contact

  # Check body height is above threshold (prevents butt-sitting)
  current_height = asset.data.root_link_pos_w[:, 2]
  height_ok = current_height > min_height

  # Must have both conditions
  properly_airborne = feet_airborne & height_ok

  return properly_airborne.float()


def joint_velocity_penalty(env, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Penalize high joint velocities to discourage flailing/oscillating.

  This helps prevent the robot from just swinging legs back and forth.
  """
  asset = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel[:, :12]

  # Sum of squared velocities
  return torch.sum(torch.square(joint_vel), dim=1)


def leg_extension_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Reward leg extension (straightening) during takeoff phase.

  Calf joints: more negative = more bent, less negative = straighter
  Default standing is around -1.8, fully extended is around -0.3

  This shapes the crouch→extend behavior needed for jumping.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]

  # Calf joint indices: 2, 5, 8, 11 (every 3rd joint starting from 2)
  calf_indices = [2, 5, 8, 11]
  calf_positions = asset.data.joint_pos[:, calf_indices]

  # Average calf angle (more negative = bent, less negative = extended)
  avg_calf = calf_positions.mean(dim=1)

  # Extension: -2.6 (crouched) to -0.3 (extended)
  # Normalize: 0 at -2.6, 1 at -0.3
  extension = torch.clamp((avg_calf + 2.6) / 2.3, 0.0, 1.0)

  # Only reward during takeoff phase (0.15 to 0.35)
  in_takeoff = (phase >= 0.15) & (phase < 0.35)

  return torch.where(in_takeoff, extension, torch.zeros_like(extension))


def backward_takeoff_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Reward backward pitch during takeoff phase.

  This encourages the robot to lean back (nose up) during takeoff,
  which is essential for initiating a backflip. Without this, the
  rear legs often push harder and cause a forward pitch.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]

  # During takeoff phase (0.10 to 0.35), reward backward pitch
  in_takeoff = (phase >= 0.10) & (phase < 0.35)

  # Pitch rate: negative = backward (backflip direction)
  pitch_rate = asset.data.root_link_ang_vel_b[:, 1]

  # Reward negative pitch rate (backward rotation)
  reward = torch.clamp(-pitch_rate / 5.0, 0.0, 1.0)

  return torch.where(in_takeoff, reward, torch.zeros_like(reward))


def stand_up_reward(env, command_name, standing_height=0.35, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Reward standing up tall at the end of the flip.

  During landing phase (phase > 0.85), reward:
  1. Being upright (grav_z close to -1)
  2. Being at standing height
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]

  # Only during landing phase
  in_landing = phase >= 0.85

  # Upright reward (grav_z = -1 is upright)
  grav_z = asset.data.projected_gravity_b[:, 2]
  upright_reward = torch.clamp((-grav_z - 0.5) / 0.5, 0.0, 1.0)

  # Height reward (closer to standing height is better)
  current_height = asset.data.root_link_pos_w[:, 2]
  height_reward = torch.clamp(current_height / standing_height, 0.0, 1.0)

  # Combined reward
  reward = upright_reward * height_reward

  return torch.where(in_landing, reward, torch.zeros_like(reward))


def landing_preparation_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Reward extending legs during descent to prepare for landing.

  During descent phase (0.60 to 0.90), reward legs being extended (not tucked).
  This encourages the robot to untuck and prepare to absorb the landing.

  Calf joints: more negative = more bent/tucked, less negative = straighter/extended
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]

  # During descent/pre-landing phase
  in_descent = (phase >= 0.60) & (phase < 0.90)

  # Calf joint indices: 2, 5, 8, 11 (every 3rd joint starting from 2)
  calf_indices = [2, 5, 8, 11]
  calf_positions = asset.data.joint_pos[:, calf_indices]

  # Average calf angle
  # Tucked: around -2.6 (very bent)
  # Standing: around -1.8
  # Extended for landing: around -1.0 to -1.5
  avg_calf = calf_positions.mean(dim=1)

  # Reward less bent legs (higher values = more extended)
  # -2.6 (tucked) -> 0.0, -1.0 (extended) -> 1.0
  extension = torch.clamp((avg_calf + 2.6) / 1.6, 0.0, 1.0)

  return torch.where(in_descent, extension, torch.zeros_like(extension))


def crouch_reward(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  """
  Reward crouching (lowering height) during crouch phase.

  This shapes the preparation for the jump.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  current_height = asset.data.root_link_pos_w[:, 2]

  # Standing ~0.35, crouch target ~0.18
  # Reward for getting lower (closer to 0.18)
  crouch_target = 0.18
  standing = 0.35

  # How much we've crouched (0 at standing, 1 at target)
  crouch_amount = torch.clamp((standing - current_height) / (standing - crouch_target), 0.0, 1.0)

  # Only during crouch phase (0 to 0.15)
  in_crouch = phase < 0.15

  return torch.where(in_crouch, crouch_amount, torch.zeros_like(crouch_amount))


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
