"""
Build weekly and monthly activity summaries from Strava export.
Uses io_strava for a single source of column handling and units.
"""
import os
import pandas as pd

from io_strava import load_strava_activities


PROCESSED_DIR = os.path.join("data", "processed")


def _week_start_monday(dt: pd.Series) -> pd.Series:
    """Monday as start of week (matches build_datasets)."""
    d = pd.to_datetime(dt, errors="coerce")
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_strava_activities()

    # Alias for compatibility with existing summary column names
    df = df.rename(columns={"distance_mi": "distance_raw"})
    df["week_start"] = _week_start_monday(df["activity_date"])
    df["month"] = df["activity_date"].dt.to_period("M").astype(str)

    weekly = df.groupby("week_start").agg(
        activities=("activity_date", "count"),
        total_distance=("distance_raw", "sum"),
        total_moving_hours=("moving_seconds", lambda x: x.dropna().sum() / 3600),
        total_elev_gain=("elev_gain", "sum"),
        avg_hr=("avg_hr", "mean"),
    ).reset_index()

    monthly = df.groupby("month").agg(
        activities=("activity_date", "count"),
        total_distance=("distance_raw", "sum"),
        total_moving_hours=("moving_seconds", lambda x: x.dropna().sum() / 3600),
        total_elev_gain=("elev_gain", "sum"),
        avg_hr=("avg_hr", "mean"),
    ).reset_index()

    weekly.to_csv(os.path.join(PROCESSED_DIR, "weekly_summary.csv"), index=False)
    monthly.to_csv(os.path.join(PROCESSED_DIR, "monthly_summary.csv"), index=False)

    print("Wrote data/processed/weekly_summary.csv and monthly_summary.csv")


if __name__ == "__main__":
    main()
