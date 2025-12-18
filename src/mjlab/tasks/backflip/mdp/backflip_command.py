from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg


class BackflipPhaseCommand(CommandTerm):

  cfg: BackflipPhaseCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)

    self.robot = env.scene[cfg.asset_name]
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self._command = torch.zeros(self.num_envs, 5, device=self.device)
    self.cumulative_rotation = torch.zeros(self.num_envs, device=self.device)
    self.prev_pitch = torch.zeros(self.num_envs, device=self.device)
    self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    self.metrics["max_phase_reached"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["completed"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["cumulative_rotation"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self):
    return self._command

  def _update_metrics(self):
    self.metrics["max_phase_reached"] = torch.maximum(
      self.metrics["max_phase_reached"], self.phase
    )
    self.metrics["completed"] = self.completed.float()
    self.metrics["cumulative_rotation"] = self.cumulative_rotation.abs()

  def _resample_command(self, env_ids):
    self.phase[env_ids] = 0.0
    self.cumulative_rotation[env_ids] = 0.0
    self.prev_pitch[env_ids] = 0.0
    self.completed[env_ids] = False
    self._update_reference(env_ids)

  def _update_command(self):
    quat = self.robot.data.root_link_quat_w
    siny_cosp = 2.0 * (quat[:, 0] * quat[:, 2] + quat[:, 3] * quat[:, 1])
    cosy_cosp = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    current_pitch = torch.atan2(siny_cosp, cosy_cosp)

    pitch_delta = current_pitch - self.prev_pitch
    pitch_delta = torch.where(pitch_delta > math.pi, pitch_delta - 2 * math.pi, pitch_delta)
    pitch_delta = torch.where(pitch_delta < -math.pi, pitch_delta + 2 * math.pi, pitch_delta)
    self.cumulative_rotation += pitch_delta
    self.prev_pitch = current_pitch

    self.phase = torch.clamp(-self.cumulative_rotation / (2 * math.pi), 0.0, 1.0)
    self.completed = self.phase >= 0.95

    all_envs = torch.arange(self.num_envs, device=self.device)
    self._update_reference(all_envs)

  def _update_reference(self, env_ids):
    phi = self.phase[env_ids]

    standing_height = self.cfg.standing_height
    peak_height = self.cfg.peak_height
    target_height = standing_height + (peak_height - standing_height) * torch.sin(math.pi * phi)

    target_pitch = -2.0 * math.pi * phi
    target_pitch_vel = torch.full_like(phi, -2.0 * math.pi / self.cfg.flip_duration)

    landing_mask = phi > 0.8
    target_pitch_vel[landing_mask] *= 0.5

    self._command[env_ids, 0] = phi
    self._command[env_ids, 1] = target_height
    self._command[env_ids, 2] = target_pitch
    self._command[env_ids, 3] = target_pitch_vel
    self._command[env_ids, 4] = self.cumulative_rotation[env_ids]

  def _debug_vis_impl(self, visualizer):
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    target_height = self._command[batch, 1].item()
    base_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    target_pos = base_pos.copy()
    target_pos[2] = target_height

    visualizer.add_sphere(target_pos, radius=0.05, color=(0.2, 0.8, 0.2, 0.5))


@dataclass(kw_only=True)
class BackflipPhaseCommandCfg(CommandTermCfg):
  asset_name: str
  flip_duration: float = 0.8
  standing_height: float = 0.35
  peak_height: float = 0.7
  class_type: type = BackflipPhaseCommand

  @dataclass
  class VizCfg:
    enabled: bool = True

  viz: VizCfg = field(default_factory=VizCfg)
