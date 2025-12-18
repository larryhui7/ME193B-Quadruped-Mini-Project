"""Backflip command generator with phase-based reference trajectory.

The phase variable φ ∈ [0, 1] parameterizes progress through the backflip:
  - φ = 0.00: Initial standing pose
  - φ ≈ 0.25: Takeoff and beginning of rotation
  - φ ≈ 0.50: Midair, upside-down orientation
  - φ ≈ 0.75: Approaching landing
  - φ = 1.00: Final landing and recovery pose
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class BackflipPhaseCommand(CommandTerm):
  """Phase-based command generator for the backflip task.

  Outputs reference quantities that describe the progression of a backflip:
  - Phase variable φ ∈ [0, 1]
  - Target base height
  - Target pitch angle (rotation about y-axis)
  - Target pitch angular velocity
  """

  cfg: BackflipPhaseCommandCfg

  def __init__(self, cfg: BackflipPhaseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]

    # Phase variable: φ ∈ [0, 1]
    self.phase = torch.zeros(self.num_envs, device=self.device)

    # Command buffer: [phase, target_height, target_pitch, target_pitch_vel]
    self._command = torch.zeros(self.num_envs, 4, device=self.device)

    # Track if backflip has been completed
    self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # Metrics
    self.metrics["max_phase_reached"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["completed"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Returns [phase, target_height, target_pitch, target_pitch_vel]."""
    return self._command

  def _update_metrics(self) -> None:
    # Track maximum phase reached
    self.metrics["max_phase_reached"] = torch.maximum(
      self.metrics["max_phase_reached"], self.phase
    )
    self.metrics["completed"] = self.completed.float()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Reset phase to 0 for resampled environments."""
    self.phase[env_ids] = 0.0
    self.completed[env_ids] = False
    self._update_reference(env_ids)

  def _update_command(self) -> None:
    """Advance phase and update reference trajectory."""
    # Advance phase based on time
    # Phase advances from 0 to 1 over the flip duration
    dt = self._env.step_dt
    phase_rate = 1.0 / self.cfg.flip_duration
    self.phase = torch.clamp(self.phase + phase_rate * dt, 0.0, 1.0)

    # Mark as completed when phase reaches 1
    self.completed = self.phase >= 1.0

    # Update reference for all environments
    all_envs = torch.arange(self.num_envs, device=self.device)
    self._update_reference(all_envs)

  def _update_reference(self, env_ids: torch.Tensor) -> None:
    """Compute reference trajectory based on current phase."""
    phi = self.phase[env_ids]

    # Target height profile (smooth arc)
    # Standing height -> jump up -> peak in middle -> come down -> land
    standing_height = self.cfg.standing_height
    peak_height = self.cfg.peak_height

    # Use a smooth function for height: starts at standing, peaks at φ=0.5
    # h(φ) = standing + (peak - standing) * sin(π * φ)
    target_height = standing_height + (peak_height - standing_height) * torch.sin(
      math.pi * phi
    )

    # Target pitch profile (full rotation backward)
    # φ=0: pitch=0 (upright)
    # φ=0.5: pitch=-π (upside down)
    # φ=1: pitch=-2π (completed rotation, upright again)
    # We use -2π for a full backward rotation
    target_pitch = -2.0 * math.pi * phi

    # Target pitch velocity (derivative of pitch w.r.t. time)
    # dθ/dt = dθ/dφ * dφ/dt = -2π * (1/flip_duration)
    target_pitch_vel = torch.full_like(
      phi, -2.0 * math.pi / self.cfg.flip_duration
    )

    # For landing phase (φ > 0.8), reduce target velocity
    landing_mask = phi > 0.8
    target_pitch_vel[landing_mask] *= 0.5

    # Store in command buffer
    self._command[env_ids, 0] = phi
    self._command[env_ids, 1] = target_height
    self._command[env_ids, 2] = target_pitch
    self._command[env_ids, 3] = target_pitch_vel

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Visualize the backflip phase and target."""
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    phi = self.phase[batch].item()
    target_height = self._command[batch, 1].item()

    # Draw target height indicator
    base_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    target_pos = base_pos.copy()
    target_pos[2] = target_height

    visualizer.add_sphere(
      target_pos, radius=0.05, color=(0.2, 0.8, 0.2, 0.5)
    )


@dataclass(kw_only=True)
class BackflipPhaseCommandCfg(CommandTermCfg):
  """Configuration for the backflip phase command generator."""

  asset_name: str
  """Name of the robot asset."""

  flip_duration: float = 0.8
  """Duration of the backflip in seconds."""

  standing_height: float = 0.35
  """Standing height of the robot base (m)."""

  peak_height: float = 0.7
  """Peak height during the flip (m)."""

  class_type: type[CommandTerm] = BackflipPhaseCommand

  @dataclass
  class VizCfg:
    enabled: bool = True

  viz: VizCfg = field(default_factory=VizCfg)

