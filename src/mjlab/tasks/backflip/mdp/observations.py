from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def backflip_phase(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 0:1]


def backflip_target_height(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 1:2]


def backflip_target_pitch(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 2:3]


def backflip_target_pitch_vel(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 3:4]


def backflip_target_x(env, command_name):
  command = env.command_manager.get_command(command_name)
  return command[:, 5:6]


def base_height(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def base_pitch(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  proj_grav = asset.data.projected_gravity_b
  pitch = torch.asin(-proj_grav[:, 0:1])
  return pitch


def base_pitch_rate(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b[:, 1:2]


def base_orientation_quat(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_quat_w


def base_lin_vel_vertical(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w[:, 2:3]


def height_error(env, command_name, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  current_height = asset.data.root_link_pos_w[:, 2:3]
  target_height = command[:, 1:2]
  return target_height - current_height


def cumulative_pitch(env, asset_cfg=_DEFAULT_ASSET_CFG):
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b[:, 1:2]
