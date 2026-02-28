import os
import numpy as np
import pandas as pd

from config import BUILD_START, GOAL_MP_SEC, MP_BAND_SEC, RAMP_THRESHOLD
from io_strava import load_strava_activities


PROCESSED_DIR = os.path.join("data", "processed")


def pace_sec_per_mile(distance_mi: pd.Series, moving_seconds: pd.Series) -> pd.Series:
    d = pd.to_numeric(distance_mi, errors="coerce")
    t = pd.to_numeric(moving_seconds, errors="coerce")
    pace = t / d
    return pace.replace([np.inf, -np.inf], np.nan)


def to_week_start(dt: pd.Series) -> pd.Series:
    # Monday as start of week
    d = pd.to_datetime(dt, errors="coerce")
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    from stream_mp import find_fit_files, build_fit_index_by_activity_id, mp_miles_from_fit

    df = load_strava_activities()

    # Runs only (handles "Run", "Virtual Run", "Trail Run" etc)
    runs = df[df["activity_type"].str.contains("run", case=False, na=False)].copy()
    runs = runs[runs["activity_date"] >= BUILD_START].copy()
    runs = runs.dropna(subset=["distance_mi", "moving_seconds"])

    runs["pace_sec_mi"] = pace_sec_per_mile(runs["distance_mi"], runs["moving_seconds"])
    runs = runs.dropna(subset=["pace_sec_mi"])
    runs["pace_min_mi"] = runs["pace_sec_mi"] / 60.0
    runs["week_start"] = to_week_start(runs["activity_date"])
    runs["month"] = runs["activity_date"].dt.to_period("M").astype(str)

    # Aerobic Efficiency Score (AES): pace per bpm (lower is better)
    # Only valid when HR exists
    runs["aes"] = np.where(
        runs["avg_hr"].notna() & (runs["avg_hr"] > 0),
        runs["pace_sec_mi"] / runs["avg_hr"],
        np.nan,
    )

    # Goal marathon pace band (from config)
    runs["is_mp_band"] = runs["pace_sec_mi"].between(
        GOAL_MP_SEC - MP_BAND_SEC, GOAL_MP_SEC + MP_BAND_SEC
    )

    raw_root = os.path.join("data", "raw", "strava_export")
    fit_files = find_fit_files(raw_root)
    fit_index = build_fit_index_by_activity_id(fit_files)

    def compute_mp_miles_stream(row) -> float:
        aid = row.get("activity_id")
        if pd.isna(aid):
            return 0.0
        aid = int(aid)
        fp = fit_index.get(aid)
        if not fp:
            return 0.0
        try:
            return mp_miles_from_fit(
                fp,
                goal_mp_sec=GOAL_MP_SEC,
                band_sec=MP_BAND_SEC,
                require_contiguous_miles=0.0,
                allow_gap_seconds=0,
            )
        except Exception:
            return 0.0

    runs["mp_miles_stream"] = runs.apply(compute_mp_miles_stream, axis=1)

    # Weekly metrics
    weekly = runs.groupby("week_start").agg(
        runs=("activity_date", "count"),
        miles=("distance_mi", "sum"),
        hours=("moving_seconds", lambda x: np.nansum(x) / 3600.0),
        elev_gain=("elev_gain", "sum"),
        avg_hr=("avg_hr", "mean"),
        mp_miles=("distance_mi", lambda x: 0.0),  # placeholder, computed below
        aes_mean=("aes", "mean"),
    ).reset_index()

    # MP miles per week: sum miles where is_mp_band True
    mp_week = runs.groupby("week_start")["mp_miles_stream"].sum().reset_index()
    mp_week = mp_week.rename(columns={"mp_miles_stream": "mp_miles"})
    weekly = weekly.drop(columns=["mp_miles"], errors="ignore").merge(mp_week, on="week_start", how="left")
    weekly["mp_miles"] = weekly["mp_miles"].fillna(0.0)

    # Marathon Pace Specificity Index (avoid div by zero when miles == 0)
    weekly["mp_specificity"] = np.where(weekly["miles"] > 0, weekly["mp_miles"] / weekly["miles"], np.nan)

    # Simple load proxy (distance * avg_hr). HR may be missing, so fill with nan-safe.
    weekly["load"] = weekly["miles"] * weekly["avg_hr"]

    # Ramp rate week over week (avoid div by zero when load_prev is 0 or nan)
    weekly = weekly.sort_values("week_start")
    weekly["load_prev"] = weekly["load"].shift(1)
    valid_prev = weekly["load_prev"].notna() & (weekly["load_prev"] > 0)
    weekly["ramp_rate"] = np.where(
        valid_prev,
        (weekly["load"] - weekly["load_prev"]) / weekly["load_prev"],
        np.nan,
    )
    weekly["ramp_over_threshold"] = weekly["ramp_rate"].gt(RAMP_THRESHOLD)

    # Long run each week
    idx = runs.groupby("week_start")["distance_mi"].idxmax()
    long_runs = runs.loc[idx, ["week_start", "distance_mi", "pace_min_mi", "avg_hr"]].copy()
    long_runs = long_runs.rename(columns={
        "distance_mi": "long_run_miles",
        "pace_min_mi": "long_run_pace_min_mi",
        "avg_hr": "long_run_avg_hr",
    })
    weekly = weekly.merge(long_runs, on="week_start", how="left")

    # 30-day rolling AES trend at activity level
    runs = runs.sort_values("activity_date")
    runs["aes_30d"] = runs["aes"].rolling(window=30, min_periods=10).mean()

    # Save outputs
    runs.to_csv(os.path.join(PROCESSED_DIR, "runs_enriched.csv"), index=False)
    weekly.to_csv(os.path.join(PROCESSED_DIR, "weekly_model.csv"), index=False)

    print("Wrote:")
    print("- data/processed/runs_enriched.csv")
    print("- data/processed/weekly_model.csv")


if __name__ == "__main__":
    main()