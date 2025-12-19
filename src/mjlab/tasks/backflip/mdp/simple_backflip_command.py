"""Time-based backflip command generator with crouch-then-jump trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg


class SimpleBackflipCommand(CommandTerm):
  """
  Outputs: [phase, target_height, target_grav_x, target_grav_z]

  Phases: crouch (0-0.15), takeoff (0.15-0.30), airborne (0.30-0.70), land (0.70-1.0)
  """

  cfg: SimpleBackflipCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.asset_name]
    self.env = env
    self._command = torch.zeros(self.num_envs, 4, device=self.device)
    self.dt = env.step_dt
    self.flip_steps = int(cfg.flip_duration / self.dt)
    self.metrics["max_phase_reached"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self):
    return self._command

  def _update_metrics(self):
    phase = self._command[:, 0]
    self.metrics["max_phase_reached"] = torch.maximum(self.metrics["max_phase_reached"], phase)

  def _resample_command(self, env_ids):
    self._update_reference(env_ids)

  def _update_command(self):
    all_envs = torch.arange(self.num_envs, device=self.device)
    self._update_reference(all_envs)

  def _update_reference(self, env_ids):
    time_steps = self.env.episode_length_buf[env_ids].float()
    phase = torch.clamp(time_steps / self.flip_steps, 0.0, 1.0)

    target_height = self._compute_height_trajectory(phase)
    target_grav_x, target_grav_z = self._compute_orientation_trajectory(phase)

    self._command[env_ids, 0] = phase
    self._command[env_ids, 1] = target_height
    self._command[env_ids, 2] = target_grav_x
    self._command[env_ids, 3] = target_grav_z

  def _compute_height_trajectory(self, phase):
    standing = self.cfg.standing_height
    crouch = self.cfg.crouch_height
    peak = self.cfg.peak_height

    height = torch.zeros_like(phase)

    # Crouch (0 -> 0.15)
    mask1 = phase < 0.15
    t1 = phase / 0.15
    height = torch.where(mask1, standing - (standing - crouch) * t1, height)

    # Takeoff (0.15 -> 0.30)
    mask2 = (phase >= 0.15) & (phase < 0.30)
    t2 = (phase - 0.15) / 0.15
    takeoff_height = crouch + (peak - crouch) * 0.8 * t2
    height = torch.where(mask2, takeoff_height, height)

    # Airborne (0.30 -> 0.70)
    mask3 = (phase >= 0.30) & (phase < 0.70)
    t3 = (phase - 0.30) / 0.40
    airborne_base = crouch + (peak - crouch) * 0.8
    airborne_height = airborne_base + (peak - airborne_base) * torch.sin(math.pi * t3)
    height = torch.where(mask3, airborne_height, height)

    # Landing (0.70 -> 1.00)
    mask4 = phase >= 0.70
    t4 = (phase - 0.70) / 0.30
    landing_height = airborne_base * (1 - t4) + standing * t4
    height = torch.where(mask4, landing_height, height)

    return height

  def _compute_orientation_trajectory(self, phase):
    # Rotation starts after crouch (phase > 0.15)
    rotation_phase = torch.clamp((phase - 0.15) / 0.85, 0.0, 1.0)
    rotation_angle = 2.0 * math.pi * rotation_phase

    target_grav_x = torch.sin(rotation_angle)
    target_grav_z = -torch.cos(rotation_angle)

    return target_grav_x, target_grav_z

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
class SimpleBackflipCommandCfg(CommandTermCfg):
  asset_name: str
  flip_duration: float = 1.0
  standing_height: float = 0.35
  crouch_height: float = 0.18
  peak_height: float = 0.90
  class_type: type = SimpleBackflipCommand

  @dataclass
  class VizCfg:
    enabled: bool = True

  viz: VizCfg = field(default_factory=VizCfg)
