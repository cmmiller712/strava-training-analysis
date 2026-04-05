"""
Training phase classification for marathon build.

The classify_training_phase() function accepts a cfg object so thresholds can
be overridden at runtime (e.g. from Streamlit sidebar sliders) without touching
this file or config.py. Any object with the required attributes works — the
config module itself, a types.SimpleNamespace, or a dataclass.

Required cfg attributes:
  BASE_MAX_MILES      — weekly miles below this = Base
  BUILD_MIN_MILES     — weekly miles at or above this = eligible for Build
  PEAK_LONG_RUN_MI    — long run at or above this triggers Peak
  PEAK_SPECIFICITY    — fraction of miles at MP for Peak (e.g. 0.20)
  TAPER_FACTOR        — miles < (4wk_avg × this) = Taper
  BUILD_START_WEEKS   — weeks elapsed before Base can graduate to Build/Peak

Phase priority order (Taper checked first, overrides everything):
  TAPER  → current miles < prior-4-week-avg × TAPER_FACTOR
  PEAK   → long_run_miles >= PEAK_LONG_RUN_MI OR mp_specificity >= PEAK_SPECIFICITY
  BUILD  → miles >= BUILD_MIN_MILES AND week_index >= BUILD_START_WEEKS
  BASE   → everything else
"""
import pandas as pd


def classify_training_phase(
    weekly_df: pd.DataFrame,
    cfg,
) -> tuple[pd.DataFrame, str, dict]:
    """
    Assign a training phase label to each week in the weekly model.

    Args:
        weekly_df: DataFrame with columns: week_start, miles, long_run_miles,
                   mp_specificity. Must not be empty.
        cfg:       Object with phase threshold attributes (see module docstring).
                   Pass the config module for headless use; pass a
                   types.SimpleNamespace from Streamlit sliders for dashboard use.

    Returns:
        weekly_df:     Input DataFrame with 'phase' column added (sorted by week_start).
        current_phase: Phase label for the most recent week (str).
        phase_history: Dict of {phase_label: week_count} for all weeks, e.g.
                       {"Base": 3, "Build": 4, "Peak": 1}. Useful for dashboard display.
    """
    weekly = weekly_df.sort_values("week_start").copy()

    # Rolling 4-week average of prior weeks' mileage. Shifted by 1 so the current
    # week is not in its own reference window (no look-ahead bias). min_periods=2
    # ensures we need at least 2 prior data points before firing the Taper rule —
    # preventing the first week from being labeled Taper just because it follows nothing.
    weekly["_miles_4wk_avg"] = (
        weekly["miles"].shift(1).rolling(4, min_periods=2).mean()
    )

    phases = []
    for week_index, (_, row) in enumerate(weekly.iterrows()):
        miles = row["miles"] if pd.notna(row["miles"]) else 0.0
        long_run = row["long_run_miles"] if pd.notna(row.get("long_run_miles")) else 0.0
        mp_spec = row["mp_specificity"] if pd.notna(row.get("mp_specificity")) else 0.0
        avg4 = row["_miles_4wk_avg"]

        # --- Priority 0: Rest ---
        # A zero-mile week means the athlete took the week off entirely (illness,
        # travel, planned rest). Guard this before the Taper check: without it,
        # 0 miles < (4wk_avg × TAPER_FACTOR) always fires, mislabeling every
        # rest week as Taper regardless of where it falls in the build.
        if miles == 0.0:
            phase = "Rest"

        # --- Priority 1: Taper ---
        # A meaningful mileage drop from the recent baseline signals deliberate
        # fatigue reduction before a race. Overrides all other phase signals.
        elif pd.notna(avg4) and avg4 > 0 and miles < avg4 * cfg.TAPER_FACTOR:
            phase = "Taper"

        # --- Priority 2: Peak ---
        # Race-specific stimulus is present: either the long run is long enough
        # to stress the marathon-specific systems, or enough miles were run at
        # goal pace. The BUILD_START_WEEKS gate still applies here — an athlete
        # who hits a long run in week 2 is still building base fitness.
        elif (
            week_index >= cfg.BUILD_START_WEEKS
            and (long_run >= cfg.PEAK_LONG_RUN_MI or mp_spec >= cfg.PEAK_SPECIFICITY)
        ):
            phase = "Peak"

        # --- Priority 3: Build ---
        # Volume is above the base threshold AND enough time has elapsed to have
        # actually built a base. BUILD_START_WEEKS prevents high-volume early weeks
        # from skipping the Base label — aerobic adaptation takes time regardless
        # of how many miles you run in week 3.
        elif week_index >= cfg.BUILD_START_WEEKS and miles >= cfg.BUILD_MIN_MILES:
            phase = "Build"

        # --- Default: Base ---
        # Low volume, early block, or insufficient time elapsed to graduate.
        else:
            phase = "Base"

        phases.append(phase)

    weekly["phase"] = phases
    weekly = weekly.drop(columns=["_miles_4wk_avg"])

    current_phase = weekly.iloc[-1]["phase"]

    # Count how many weeks were spent in each phase — useful for dashboard context
    # ("You've been in Build phase for 4 of the last 8 weeks").
    phase_history: dict[str, int] = weekly["phase"].value_counts().to_dict()

    return weekly, current_phase, phase_history
