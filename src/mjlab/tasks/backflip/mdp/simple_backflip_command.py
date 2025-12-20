"""Time-based backflip command generator with crouch-then-jump trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg


class SimpleBackflipCommand(CommandTerm):
  """
  Outputs: [phase, target_height, target_grav_x, target_grav_z, target_x]

  Trajectory phases (time-based):
    crouch (0-0.15), takeoff (0.15-0.30), airborne (0.30-0.70), land (0.70-1.0)

  Progress metric phases (achievement-based):
    0.0=start, 0.1=crouched, 0.25=airborne, 0.5=inverted(180°), 0.75=270°, 1.0=landed upright

  2D trajectory: Robot moves backward (negative x) during flip, landing near start position.
  """

  cfg: SimpleBackflipCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.asset_name]
    self.env = env
    self._command = torch.zeros(self.num_envs, 5, device=self.device)  # Added target_x
    self.dt = env.step_dt
    self.flip_steps = int(cfg.flip_duration / self.dt)
    self.metrics["max_rotation_progress"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["max_height"] = torch.zeros(self.num_envs, device=self.device)
    # Store starting x position for relative trajectory
    self.start_x = torch.zeros(self.num_envs, device=self.device)
    # Track cumulative pitch rotation (negative = backflip, positive = frontflip)
    self.cumulative_pitch = torch.zeros(self.num_envs, device=self.device)
    # Optional feet sensor for airborne detection
    self.feet_sensor = None
    if cfg.feet_sensor_name is not None:
      self.feet_sensor = env.scene[cfg.feet_sensor_name]

  @property
  def command(self):
    return self._command

  def _update_metrics(self):
    import math

    # Simple approach: integrate pitch rate to track cumulative rotation
    # Pitch rate is Y-axis angular velocity in body frame
    # NEGATIVE pitch rate = backflip (nose up first)
    # POSITIVE pitch rate = frontflip (nose down first)
    pitch_rate = self.robot.data.root_link_ang_vel_b[:, 1]
    self.cumulative_pitch = self.cumulative_pitch + pitch_rate * self.dt

    # For backflip, cumulative_pitch goes negative (full backflip = -2π)
    # Convert to progress: 0 at start, 1 at full rotation
    # Only count negative rotation (backflip direction)
    backflip_rotation = torch.clamp(-self.cumulative_pitch, 0.0, 2.0 * math.pi)
    rotation_frac = backflip_rotation / (2.0 * math.pi)

    current_height = self.robot.data.root_link_pos_w[:, 2]
    self.metrics["max_height"] = torch.maximum(self.metrics["max_height"], current_height)

    # Phase-based progress with smooth interpolation within phases
    # Phases: 0=start, 0.1=crouch, 0.25=takeoff, 0.5=inverted(180°), 0.75=270°, 1.0=land
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

    # Phase 0.1->0.25: Takeoff (height toward peak)
    # Progress based on height, but must be airborne to complete (reach 0.25)
    crouch_done = new_progress >= 0.1
    takeoff_target = standing + 0.5 * (self.cfg.peak_height - standing)
    in_takeoff_phase = crouch_done & (current_progress < 0.25)
    takeoff_frac = torch.clamp((current_height - crouch) / (takeoff_target - crouch), 0.0, 1.0)
    # Can get up to 0.24 from height alone, need airborne to hit 0.25
    takeoff_progress_from_height = 0.1 + 0.14 * takeoff_frac
    takeoff_progress_airborne = torch.where(is_airborne, torch.tensor(0.25, device=self.device), takeoff_progress_from_height)
    new_progress = torch.where(in_takeoff_phase, torch.maximum(new_progress, takeoff_progress_airborne), new_progress)

    # Phase 0.25->0.5: Rotating to inverted (rotation 0 -> 0.5 = 0° -> 180°)
    # Must be airborne to get rotation credit; height gives bonus
    takeoff_done = new_progress >= 0.25
    in_rotation_phase = takeoff_done & (current_progress < 0.5)
    # Height bonus: 0.5 base credit + 0.5 bonus for height (so even low jumps get credit)
    height_bonus = torch.clamp((current_height - standing) / (self.cfg.peak_height - standing), 0.0, 1.0)
    credit_scale = 0.5 + 0.5 * height_bonus  # ranges from 0.5 to 1.0
    # Progress based on rotation angle
    rotation_to_inverted = torch.clamp(rotation_frac / 0.5, 0.0, 1.0)
    rotation_progress = 0.25 + 0.25 * rotation_to_inverted * credit_scale
    # Only count if airborne
    new_progress = torch.where(in_rotation_phase & is_airborne, torch.maximum(new_progress, rotation_progress), new_progress)

    # Phase 0.5->0.75: Complete flip (rotation 0.5 -> 1.0 = 180° -> 270°, must be airborne)
    # Note: 270°-360° is handled by landing phase to avoid giving credit for forward pitch
    at_inverted = new_progress >= 0.5
    in_flip_phase = at_inverted & (current_progress < 0.75)
    # rotation_frac 0.5->1.0 maps to phase 0.5->0.75
    # Use max_height for credit_scale so falling doesn't penalize progress
    max_height_bonus = torch.clamp((self.metrics["max_height"] - standing) / (self.cfg.peak_height - standing), 0.0, 1.0)
    flip_credit_scale = 0.5 + 0.5 * max_height_bonus
    flip_frac = torch.clamp((rotation_frac - 0.5) / 0.5, 0.0, 1.0)
    flip_progress = 0.5 + 0.25 * flip_frac * flip_credit_scale
    # Must be airborne
    new_progress = torch.where(in_flip_phase & is_airborne, torch.maximum(new_progress, flip_progress), new_progress)

    # Phase 0.75->1.0: Landing (on ground, upright)
    flip_done = new_progress >= 0.75
    in_landing_phase = flip_done & (current_progress < 1.0)
    # Landing progress: upright (grav_z close to -1) and on ground
    grav_z = self.robot.data.projected_gravity_b[:, 2]
    upright_frac = torch.clamp((-grav_z - 0.5) / 0.5, 0.0, 1.0)  # grav_z: -1=upright, 1=inverted
    on_ground = ~is_airborne
    landing_progress = 0.75 + 0.25 * upright_frac
    # Only count if back on ground
    new_progress = torch.where(in_landing_phase & on_ground, torch.maximum(new_progress, landing_progress), new_progress)

    self.metrics["max_rotation_progress"] = new_progress

  def _resample_command(self, env_ids):
    # Reset metrics on episode reset
    self.metrics["max_rotation_progress"][env_ids] = 0.0
    self.metrics["max_height"][env_ids] = 0.0
    # Reset cumulative pitch tracking
    self.cumulative_pitch[env_ids] = 0.0
    # Store starting x position for relative trajectory
    self.start_x[env_ids] = self.robot.data.root_link_pos_w[env_ids, 0]
    self._update_reference(env_ids)

  def _update_command(self):
    all_envs = torch.arange(self.num_envs, device=self.device)
    self._update_reference(all_envs)

  def _update_reference(self, env_ids):
    time_steps = self.env.episode_length_buf[env_ids].float()
    phase = torch.clamp(time_steps / self.flip_steps, 0.0, 1.0)

    target_height = self._compute_height_trajectory(phase)
    target_grav_x, target_grav_z = self._compute_orientation_trajectory(phase)
    target_x_offset = self._compute_x_trajectory(phase)
    target_x = self.start_x[env_ids] + target_x_offset

    self._command[env_ids, 0] = phase
    self._command[env_ids, 1] = target_height
    self._command[env_ids, 2] = target_grav_x
    self._command[env_ids, 3] = target_grav_z
    self._command[env_ids, 4] = target_x

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
    # Rotation happens between takeoff (0.15) and landing (0.70)
    # Complete full 360° before landing so robot is upright when touching down
    rotation_phase = torch.clamp((phase - 0.15) / 0.55, 0.0, 1.0)
    rotation_angle = 2.0 * math.pi * rotation_phase

    # For backflip (pitch backward), grav_x goes negative first
    target_grav_x = -torch.sin(rotation_angle)
    target_grav_z = -torch.cos(rotation_angle)

    return target_grav_x, target_grav_z

  def _compute_x_trajectory(self, phase):
    """Compute backward displacement during flip.

    The robot moves backward (negative x) during the flip, with peak
    displacement around the apex, then returns near the start position.
    """
    backward = self.cfg.backward_displacement

    x_offset = torch.zeros_like(phase)

    # Crouch (0 -> 0.15): stay in place
    # (x_offset already 0)

    # Takeoff to apex (0.15 -> 0.50): move backward
    mask2 = (phase >= 0.15) & (phase < 0.50)
    t2 = (phase - 0.15) / 0.35
    x_offset = torch.where(mask2, -backward * torch.sin(math.pi / 2 * t2), x_offset)

    # Apex to landing (0.50 -> 1.00): return toward start
    mask3 = phase >= 0.50
    t3 = (phase - 0.50) / 0.50
    x_offset = torch.where(mask3, -backward * torch.cos(math.pi / 2 * t3), x_offset)

    return x_offset

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
  backward_displacement: float = 0.3  # How far back (m) the robot moves at apex
  class_type: type = SimpleBackflipCommand

  @dataclass
  class VizCfg:
    enabled: bool = True

  viz: VizCfg = field(default_factory=VizCfg)
