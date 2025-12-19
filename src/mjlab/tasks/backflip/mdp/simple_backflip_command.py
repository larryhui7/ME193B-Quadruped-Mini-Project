"""Simple time-based backflip command generator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg


class SimpleBackflipCommand(CommandTerm):
  """
  Simple time-based backflip command generator.

  Outputs:
    - phase (0-1): time-based progress through flip
    - target_height: sinusoidal height trajectory
    - target_pitch: linear pitch target from 0 to -2π
  """

  cfg: SimpleBackflipCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)

    self.robot = env.scene[cfg.asset_name]
    self.env = env
    self._command = torch.zeros(self.num_envs, 3, device=self.device)

    self.dt = env.step_dt
    self.flip_steps = int(cfg.flip_duration / self.dt)

    self.metrics["max_phase_reached"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self):
    return self._command

  def _update_metrics(self):
    phase = self._command[:, 0]
    self.metrics["max_phase_reached"] = torch.maximum(
      self.metrics["max_phase_reached"], phase
    )

  def _resample_command(self, env_ids):
    self._update_reference(env_ids)

  def _update_command(self):
    all_envs = torch.arange(self.num_envs, device=self.device)
    self._update_reference(all_envs)

  def _update_reference(self, env_ids):
    # Simple time-based phase: phi = t / T
    time_steps = self.env.episode_length_buf[env_ids].float()
    phase = torch.clamp(time_steps / self.flip_steps, 0.0, 1.0)

    standing_height = self.cfg.standing_height
    peak_height = self.cfg.peak_height

    # Height trajectory: sine curve (standing -> peak -> standing)
    target_height = standing_height + (peak_height - standing_height) * torch.sin(math.pi * phase)

    # Pitch trajectory: linear from 0 to -2π
    target_pitch = -2.0 * math.pi * phase

    self._command[env_ids, 0] = phase
    self._command[env_ids, 1] = target_height
    self._command[env_ids, 2] = target_pitch

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
  flip_duration: float = 0.8  # Total duration of the backflip
  standing_height: float = 0.35
  peak_height: float = 0.6
  class_type: type = SimpleBackflipCommand

  @dataclass
  class VizCfg:
    enabled: bool = True

  viz: VizCfg = field(default_factory=VizCfg)
