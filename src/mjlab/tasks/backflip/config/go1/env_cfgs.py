"""Go1 environment configuration for the backflip task."""

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.backflip.backflip_env_cfg import create_backflip_env_cfg

# Contact sensor for feet
GO1_FEET_SENSOR_CFG = ContactSensorCfg(
  name="feet_contact",
  prim_path="robot",
  track_air_time=True,
  body_names=(".*foot",),
)

# Self-collision sensor (body contact detection)
GO1_SELF_COLLISION_SENSOR_CFG = ContactSensorCfg(
  name="self_collision",
  prim_path="robot",
  body_names=(
    ".*thigh",
    ".*calf",
    "trunk",
  ),
)

UNITREE_GO1_BACKFLIP_ENV_CFG = create_backflip_env_cfg(
  robot_cfg=get_go1_robot_cfg(),
  action_scale=GO1_ACTION_SCALE,
  viewer_body_name="trunk",
  feet_sensor_cfg=GO1_FEET_SENSOR_CFG,
  self_collision_sensor_cfg=GO1_SELF_COLLISION_SENSOR_CFG,
  standing_height=0.28,  # Go1 standing height
  peak_height=0.6,       # Target peak height during flip
  flip_duration=0.7,     # Duration of the backflip
)

