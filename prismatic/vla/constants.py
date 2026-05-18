"""
Important constants for VLA training and evaluation.

The active set of constants is selected by the ``ASYNCVLA_PLATFORM`` environment
variable so the same codebase can drive both the original navigation pipeline
(``omnivla``) and arm-manipulation finetuning (``so101``).
"""
import os
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
OMNIVLA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 4,
    "POSE_DIM": 4,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# SO-101 (LeRobot, 6-DOF arm + gripper). Joints + gripper packed into a 6-D action.
SO101_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 6,
    "POSE_DIM": 6,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

PLATFORM_REGISTRY = {
    "omnivla": OMNIVLA_CONSTANTS,
    "so101": SO101_CONSTANTS,
}

ROBOT_PLATFORM = os.environ.get("ASYNCVLA_PLATFORM", "omnivla").lower()
if ROBOT_PLATFORM not in PLATFORM_REGISTRY:
    raise ValueError(
        f"Unknown ASYNCVLA_PLATFORM='{ROBOT_PLATFORM}'. "
        f"Supported: {sorted(PLATFORM_REGISTRY)}"
    )
constants = PLATFORM_REGISTRY[ROBOT_PLATFORM]

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
POSE_DIM = constants["POSE_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  POSE_DIM = {POSE_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("Override by exporting ASYNCVLA_PLATFORM (e.g. `export ASYNCVLA_PLATFORM=so101`).")
