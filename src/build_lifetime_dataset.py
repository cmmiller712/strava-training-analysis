"""
Build the lifetime activities dataset: ALL activities ever recorded, no date filter.
For the "fun" lifetime notebook: activity mix, kudos, rarity, do harder workouts get more kudos, etc.
"""
import os
import numpy as np
import pandas as pd

from io_strava import load_strava_activities_lifetime


PROCESSED_DIR = os.path.join("data", "processed")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_strava_activities_lifetime()

    # Pace (seconds per mile) where distance and time exist and are positive
    t = pd.to_numeric(df["moving_seconds"], errors="coerce")
    d = pd.to_numeric(df["distance_mi"], errors="coerce")
    pace = np.where((d > 0) & (t > 0), t / d, np.nan)
    pace = np.where(np.isfinite(pace), pace, np.nan)
    df["pace_sec_mi"] = pd.Series(pace, index=df.index)
    df["pace_min_mi"] = df["pace_sec_mi"] / 60.0

    # Year / month for grouping
    df["year"] = df["activity_date"].dt.year
    df["month"] = df["activity_date"].dt.to_period("M").astype(str)

    out_path = os.path.join(PROCESSED_DIR, "lifetime_activities.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} activities)")


if __name__ == "__main__":
    main()
