"""Simple time-based backflip command generator with crouch-then-jump trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg


class SimpleBackflipCommand(CommandTerm):
  """
  Simple time-based backflip command generator.

  Outputs a 4-element command:
    [0] phase (0-1): time-based progress through flip
    [1] target_height: height trajectory with crouch-then-jump
    [2] target_grav_x: x-component of target projected gravity
    [3] target_grav_z: z-component of target projected gravity

  Trajectory phases:
    φ = 0.00 → 0.15: Crouch (lower height, stay upright)
    φ = 0.15 → 0.30: Takeoff (rapid height increase, start rotation)
    φ = 0.30 → 0.70: Airborne (peak height, rotate through inverted)
    φ = 0.70 → 1.00: Landing (descend, complete rotation)
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
    self.metrics["max_phase_reached"] = torch.maximum(
      self.metrics["max_phase_reached"], phase
    )

  def _resample_command(self, env_ids):
    self._update_reference(env_ids)

  def _update_command(self):
    all_envs = torch.arange(self.num_envs, device=self.device)
    self._update_reference(all_envs)

  def _update_reference(self, env_ids):
    # Time-based phase: phi = t / T
    time_steps = self.env.episode_length_buf[env_ids].float()
    phase = torch.clamp(time_steps / self.flip_steps, 0.0, 1.0)

    # Compute height and orientation trajectories
    target_height = self._compute_height_trajectory(phase)
    target_grav_x, target_grav_z = self._compute_orientation_trajectory(phase)

    self._command[env_ids, 0] = phase
    self._command[env_ids, 1] = target_height
    self._command[env_ids, 2] = target_grav_x
    self._command[env_ids, 3] = target_grav_z

  def _compute_height_trajectory(self, phase):
    """
    Height trajectory with crouch-then-jump profile.

    φ = 0.00 → 0.15: Crouch down (0.35 → 0.20)
    φ = 0.15 → 0.30: Explosive jump (0.20 → 0.55)
    φ = 0.30 → 0.50: Rise to peak (0.55 → 0.60)
    φ = 0.50 → 0.80: Descend (0.60 → 0.40)
    φ = 0.80 → 1.00: Land (0.40 → 0.35)
    """
    standing = self.cfg.standing_height
    crouch = self.cfg.crouch_height
    peak = self.cfg.peak_height

    height = torch.zeros_like(phase)

    # Phase 1: Crouch (0 → 0.15)
    mask1 = phase < 0.15
    t1 = phase / 0.15  # normalize to [0, 1]
    height = torch.where(mask1, standing - (standing - crouch) * t1, height)

    # Phase 2: Explosive takeoff (0.15 → 0.30)
    mask2 = (phase >= 0.15) & (phase < 0.30)
    t2 = (phase - 0.15) / 0.15
    # Rapid rise from crouch to above standing
    takeoff_height = crouch + (peak - crouch) * 0.8 * t2
    height = torch.where(mask2, takeoff_height, height)

    # Phase 3: Airborne - sinusoidal peak (0.30 → 0.70)
    mask3 = (phase >= 0.30) & (phase < 0.70)
    t3 = (phase - 0.30) / 0.40
    # Smooth sine curve peaking at t3=0.5 (i.e., phase=0.50)
    airborne_base = crouch + (peak - crouch) * 0.8
    airborne_height = airborne_base + (peak - airborne_base) * torch.sin(math.pi * t3)
    height = torch.where(mask3, airborne_height, height)

    # Phase 4: Landing (0.70 → 1.00)
    mask4 = phase >= 0.70
    t4 = (phase - 0.70) / 0.30
    # Descend from ~peak back to standing
    land_start = airborne_base + (peak - airborne_base) * torch.sin(math.pi * 1.0)  # = airborne_base
    height = torch.where(mask4, peak * 0.8 * (1 - t4) + standing * t4, height)

    return height

  def _compute_orientation_trajectory(self, phase):
    """
    Orientation trajectory using projected gravity.

    For backward rotation (pitch negative):
    - grav_x = sin(rotation_angle)
    - grav_z = -cos(rotation_angle)

    where rotation_angle goes from 0 to 2π over the flip.

    Key points:
    - φ = 0.00: upright (0, -1)
    - φ = 0.15: still upright (0, -1) - crouching, no rotation yet
    - φ = 0.50: upside-down (0, +1)
    - φ = 1.00: upright again (0, -1)
    """
    # Rotation only starts after crouch phase (φ > 0.15)
    # Map φ ∈ [0.15, 1.0] to rotation ∈ [0, 2π]
    rotation_phase = torch.clamp((phase - 0.15) / 0.85, 0.0, 1.0)
    rotation_angle = 2.0 * math.pi * rotation_phase

    # Projected gravity for backward rotation
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
  flip_duration: float = 1.0  # Total duration including crouch
  standing_height: float = 0.35
  crouch_height: float = 0.18  # Deeper crouch for more power
  peak_height: float = 0.90  # Higher peak for more air time (~0.72s airborne, ~8.7 rad/s needed)
  class_type: type = SimpleBackflipCommand

  @dataclass
  class VizCfg:
    enabled: bool = True

  viz: VizCfg = field(default_factory=VizCfg)
