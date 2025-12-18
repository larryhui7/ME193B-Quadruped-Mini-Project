from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def backflip_completed(env, command_name, uprightness_threshold=0.3, min_rotation=1.8 * 3.14159, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  cumulative_rotation = command[:, 4]
  projected_gravity = asset.data.projected_gravity_b

  phase_complete = phase >= 0.9
  has_rotated = cumulative_rotation <= -min_rotation
  uprightness_error = torch.sum(projected_gravity[:, :2] ** 2, dim=1)
  is_upright = uprightness_error < uprightness_threshold

  return phase_complete & has_rotated & is_upright


def body_contact_during_flip(env, command_name, sensor_name):
  sensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  has_contact = torch.any(sensor.data.found > 0, dim=1)
  mid_flip = (phase > 0.2) & (phase < 0.8)

  return mid_flip & has_contact


def height_too_low(env, command_name, min_height=0.15, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  height = asset.data.root_link_pos_w[:, 2]

  mid_flip = (phase > 0.15) & (phase < 0.75)
  too_low = height < min_height

  return mid_flip & too_low


def bad_landing_orientation(env, command_name, limit_angle=1.2, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  projected_gravity = asset.data.projected_gravity_b

  orientation_angle = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))
  landing_phase = phase > 0.85
  bad_orientation = orientation_angle > limit_angle

  return landing_phase & bad_orientation


def no_progress(env, command_name, min_phase_by_time=0.5, time_threshold=1.0):
  command = env.command_manager.get_command(command_name)

  phase = command[:, 0]
  episode_length = env.episode_length_buf * env.step_dt
  expected_phase = episode_length / time_threshold * min_phase_by_time

  return (episode_length > time_threshold) & (phase < expected_phase * 0.5)


def excessive_rotation(env, max_roll=1.0, max_yaw_rate=5.0, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]

  proj_grav = asset.data.projected_gravity_b
  roll_error = torch.abs(proj_grav[:, 1])
  yaw_rate = torch.abs(asset.data.root_link_ang_vel_b[:, 2])

  excessive_roll = roll_error > max_roll
  excessive_yaw = yaw_rate > max_yaw_rate

  return excessive_roll | excessive_yaw
