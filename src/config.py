"""
Build and goal configuration for sub-3 marathon training analysis.
Change these to adjust date range, target pace, and ramp rules.
"""
import pandas as pd

# First date to include in run/build datasets (e.g. start of a training block).
BUILD_START = pd.to_datetime("2025-12-29")

# Last date to include — set to day before race to produce a clean pre-race snapshot.
# Race day: April 19, 2026.
BUILD_END = pd.to_datetime("2026-04-18")

# Goal marathon pace: seconds per mile (sub-3 ≈ 6:52/mi → 412 s/mi).
GOAL_MP_SEC = 412.0

# ± seconds per mile around goal pace for "marathon pace" band (stream and whole-activity).
MP_BAND_SEC = 20.0

# Ramp rate above this is flagged (e.g. week-over-week load increase).
# Used for weekly "ramp_over_threshold" and notebook logic.
RAMP_THRESHOLD = 0.15

# Readiness model thresholds
# Weekly mileage that scores 100 on the volume component.
TARGET_WEEKLY_MILES = 40

# Long run distance (miles) that scores 100 on the long run component.
PEAK_LONG_RUN_MI = 16

# Phase classifier thresholds
# Weekly mileage floor to cross from Base into Build.
BASE_MAX_MILES = 30

# Minimum weekly miles to be considered a Build week.
BUILD_MIN_MILES = 40

# Fraction of miles at marathon pace for a week to qualify as Peak.
PEAK_SPECIFICITY = 0.20

# Multiplier: if current miles < (4-week avg × TAPER_FACTOR), label as Taper.
TAPER_FACTOR = 0.75

# Weeks elapsed before Base can graduate to Build or Peak.
BUILD_START_WEEKS = 4

# Heart rate cap for Zone 2 / easy runs (bpm).
ZONE2_HR_CAP = 150

# Effort zone pace offsets — relative to GOAL_MP_SEC (seconds/mile)
# Easy:     pace > GOAL_MP_SEC + EASY_ZONE_OFFSET     (slower than MP + 90s → 8:22+/mi)
# Moderate: pace > GOAL_MP_SEC + MODERATE_ZONE_OFFSET (7:22–8:22/mi)
# Marathon: pace >= GOAL_MP_SEC - HARD_ZONE_OFFSET    (6:32–7:22/mi)
# Hard:     pace < GOAL_MP_SEC - HARD_ZONE_OFFSET     (faster than 6:32/mi)
EASY_ZONE_OFFSET     = 90
MODERATE_ZONE_OFFSET = 30
HARD_ZONE_OFFSET     = 20

# Training plan presets — used by Streamlit sidebar dropdown
# Format: {display_name: {config_key: value}}
TRAINING_PLAN_PRESETS = {
    "Custom (Pfitzinger 18/55)": {
        "TARGET_WEEKLY_MILES": 55,
        "PEAK_LONG_RUN_MI": 20,
        "PEAK_SPECIFICITY": 0.20,
        "TAPER_FACTOR": 0.75,
        "BASE_MAX_MILES": 40,
    },
    "Pfitzinger 18/70": {
        "TARGET_WEEKLY_MILES": 70,
        "PEAK_LONG_RUN_MI": 22,
        "PEAK_SPECIFICITY": 0.20,
        "TAPER_FACTOR": 0.75,
        "BASE_MAX_MILES": 45,
    },
    "Higdon Intermediate": {
        "TARGET_WEEKLY_MILES": 40,
        "PEAK_LONG_RUN_MI": 20,
        "PEAK_SPECIFICITY": 0.15,
        "TAPER_FACTOR": 0.80,
        "BASE_MAX_MILES": 30,
    },
    "Hansons Marathon": {
        "TARGET_WEEKLY_MILES": 60,
        "PEAK_LONG_RUN_MI": 16,
        "PEAK_SPECIFICITY": 0.25,
        "TAPER_FACTOR": 0.70,
        "BASE_MAX_MILES": 40,
    },
    "Current Plan (40mi / 16mi LR)": {
        "TARGET_WEEKLY_MILES": 40,
        "PEAK_LONG_RUN_MI": 16,
        "PEAK_SPECIFICITY": 0.20,
        "TAPER_FACTOR": 0.75,
        "BASE_MAX_MILES": 30,
    },
}
