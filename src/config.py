"""
Build and goal configuration for sub-3 marathon training analysis.
Change these to adjust date range, target pace, and ramp rules.
"""
import pandas as pd

# First date to include in run/build datasets (e.g. start of a training block).
BUILD_START = pd.to_datetime("2025-12-29")

# Goal marathon pace: seconds per mile (sub-3 ≈ 6:52/mi → 412 s/mi).
GOAL_MP_SEC = 412.0

# ± seconds per mile around goal pace for "marathon pace" band (stream and whole-activity).
MP_BAND_SEC = 20.0

# Ramp rate above this is flagged (e.g. week-over-week load increase).
# Used for weekly "ramp_over_threshold" and notebook logic.
RAMP_THRESHOLD = 0.15
