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
    self.metrics["max_rotation_progress"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["max_height"] = torch.zeros(self.num_envs, device=self.device)
    # Optional feet sensor for airborne detection
    self.feet_sensor = None
    if cfg.feet_sensor_name is not None:
      self.feet_sensor = env.scene[cfg.feet_sensor_name]

  @property
  def command(self):
    return self._command

  def _update_metrics(self):
    import math
    proj_grav = self.robot.data.projected_gravity_b
    grav_x = proj_grav[:, 0]
    grav_z = proj_grav[:, 2]

    # Rotation: 0 at upright, 0.5 at inverted, 1.0 back to upright
    rotation_angle = torch.atan2(grav_x, -grav_z)
    rotation_frac = torch.where(
      rotation_angle >= 0,
      rotation_angle / (2 * math.pi),
      (rotation_angle + 2 * math.pi) / (2 * math.pi)
    )

    current_height = self.robot.data.root_link_pos_w[:, 2]
    self.metrics["max_height"] = torch.maximum(self.metrics["max_height"], current_height)

    # Phase-based progress with smooth interpolation within phases
    # Phases: 0=start, 0.1=crouch, 0.25=takeoff, 0.5=vertical, 0.75=inverted, 1.0=land
    current_progress = self.metrics["max_rotation_progress"]
    new_progress = current_progress.clone()

    standing = self.cfg.standing_height
    crouch = self.cfg.crouch_height

    # Phase 0->0.1: Crouching (height goes from standing to crouch)
    in_crouch_phase = current_progress < 0.1
    crouch_frac = torch.clamp((standing - current_height) / (standing - crouch), 0.0, 1.0)
    crouch_progress = 0.1 * crouch_frac
    new_progress = torch.where(in_crouch_phase, torch.maximum(new_progress, crouch_progress), new_progress)

    # Check if airborne (using sensor if available, else height estimate)
    if self.feet_sensor is not None:
      any_contact = (self.feet_sensor.data.found > 0).any(dim=-1)
      is_airborne = ~any_contact
    else:
      # Fallback: estimate from height
      is_airborne = current_height > standing + 0.15

    # Phase 0.1->0.25: Takeoff (height toward peak, must become airborne)
    crouch_done = new_progress >= 0.1
    takeoff_target = standing + 0.5 * (self.cfg.peak_height - standing)
    in_takeoff_phase = crouch_done & (current_progress < 0.25)
    takeoff_frac = torch.clamp((current_height - crouch) / (takeoff_target - crouch), 0.0, 1.0)
    takeoff_progress = 0.1 + 0.15 * takeoff_frac
    new_progress = torch.where(in_takeoff_phase, torch.maximum(new_progress, takeoff_progress), new_progress)

    # Phase 0.25->0.5: Rotating to inverted (rotation 0 -> 0.5 = 0° -> 180°, must be airborne)
    takeoff_done = new_progress >= 0.25
    in_rotation_phase = takeoff_done & (current_progress < 0.5)
    # rotation_frac 0->0.5 maps to phase 0.25->0.5
    rotation_to_inverted = torch.clamp(rotation_frac / 0.5, 0.0, 1.0)
    height_toward_peak = torch.clamp((current_height - takeoff_target) / (self.cfg.peak_height - takeoff_target), 0.0, 1.0)
    # Combine height and rotation progress
    rotation_combined = 0.3 * height_toward_peak + 0.7 * rotation_to_inverted
    rotation_progress = 0.25 + 0.25 * rotation_combined
    # Only count if actually airborne
    new_progress = torch.where(in_rotation_phase & is_airborne, torch.maximum(new_progress, rotation_progress), new_progress)

    # Phase 0.5->0.75: Past inverted (rotation 0.5 -> 0.75 = 180° -> 270°, must be airborne)
    at_inverted = new_progress >= 0.5
    in_descent_phase = at_inverted & (current_progress < 0.75)
    descent_frac = torch.clamp((rotation_frac - 0.5) / 0.25, 0.0, 1.0)
    descent_progress = 0.5 + 0.25 * descent_frac
    # Must be airborne
    new_progress = torch.where(in_descent_phase & is_airborne, torch.maximum(new_progress, descent_progress), new_progress)

    # Phase 0.75->1.0: Landing (rotation 0.75 -> 1.0 = 270° -> 360°)
    past_descent = new_progress >= 0.75
    in_landing_phase = past_descent & (current_progress < 1.0)
    landing_frac = torch.clamp((rotation_frac - 0.75) / 0.25, 0.0, 1.0)
    landing_progress = 0.75 + 0.25 * landing_frac
    new_progress = torch.where(in_landing_phase, torch.maximum(new_progress, landing_progress), new_progress)

    self.metrics["max_rotation_progress"] = new_progress

  def _resample_command(self, env_ids):
    # Reset metrics on episode reset
    self.metrics["max_rotation_progress"][env_ids] = 0.0
    self.metrics["max_height"][env_ids] = 0.0
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
  feet_sensor_name: str | None = None
  flip_duration: float = 1.0
  standing_height: float = 0.35
  crouch_height: float = 0.18
  peak_height: float = 0.90
  class_type: type = SimpleBackflipCommand

  @dataclass
  class VizCfg:
    enabled: bool = True

  viz: VizCfg = field(default_factory=VizCfg)
