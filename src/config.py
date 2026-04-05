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

# Readiness model thresholds
# Weekly mileage that scores 100 on the volume component.
TARGET_WEEKLY_MILES = 50

# Long run distance (miles) that scores 100 on the long run component.
PEAK_LONG_RUN_MI = 20

# Phase classifier thresholds
# Minimum weekly miles to be considered a Build week.
BUILD_MIN_MILES = 40

# Fraction of miles at marathon pace for a week to qualify as Peak.
PEAK_SPECIFICITY = 0.25

# Multiplier: if current miles < (4-week avg × TAPER_FACTOR), label as Taper.
TAPER_FACTOR = 0.80

# Weeks elapsed before Base can graduate to Build or Peak.
BUILD_START_WEEKS = 4
