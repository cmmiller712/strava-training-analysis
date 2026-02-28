import os
import glob
import pandas as pd
import numpy as np


RAW_DIR = os.path.join("data", "raw", "strava_export")
ACTIVITIES_GLOB = os.path.join(RAW_DIR, "**", "activities.csv")


def find_activities_csv() -> str:
    matches = glob.glob(ACTIVITIES_GLOB, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find activities.csv under {RAW_DIR}. Expected pattern: {ACTIVITIES_GLOB}"
        )
    return matches[0]


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_seconds(series: pd.Series) -> pd.Series:
    # Handles seconds, floats, or HH:MM:SS strings
    if series is None:
        return pd.Series(dtype="float64")
    if series.dtype == "O":
        td = pd.to_timedelta(series, errors="coerce")
        return td.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def _distance_to_miles(dist: pd.Series, moving_seconds: pd.Series | None = None) -> pd.Series:
    """
    Convert distance to miles.
    Handles:
      - meters (common in APIs)
      - kilometers (common in Strava exports)
      - miles (already miles)
    Uses a pace sanity check when moving time is available to distinguish mi vs km.
    """
    d = pd.to_numeric(dist, errors="coerce")
    med = np.nanmedian(d.values) if len(d) else np.nan

    # If typical distances are huge, assume meters
    if np.isfinite(med) and med > 200:
        return d / 1609.344

    # If we have moving time, use pace sanity check to decide mi vs km
    if moving_seconds is not None:
        t = pd.to_numeric(moving_seconds, errors="coerce")
        d_safe = d.replace(0, np.nan)
        pace_sec_assuming_miles = np.nanmedian((t / d_safe).values)
        pace_min_assuming_miles = pace_sec_assuming_miles / 60.0 if np.isfinite(pace_sec_assuming_miles) else np.nan

        if np.isfinite(pace_min_assuming_miles) and pace_min_assuming_miles < 6.0:
            return d / 1.609344  # km -> miles

    # Default assume miles
    return d


def load_strava_activities() -> pd.DataFrame:
    path = find_activities_csv()
    df = pd.read_csv(path)

    id_col = _pick_col(df, ["Activity ID", "id"])

    date_col = _pick_col(df, ["Activity Date", "start_date_local", "start_date"])
    type_col = _pick_col(df, ["Activity Type", "sport_type", "type", "Type"])
    dist_col = _pick_col(df, ["Distance", "distance"])
    moving_col = _pick_col(df, ["Moving Time", "moving_time", "Timer Time", "Timer Time.1", "elapsed_time", "Elapsed Time"])
    hr_col = _pick_col(df, ["Average Heart Rate", "average_heartrate"])
    elev_col = _pick_col(df, ["Elevation Gain", "total_elevation_gain"])

    if date_col is None or type_col is None or dist_col is None:
        raise ValueError(
            "Missing required columns. Need at least: date, activity type, distance."
        )

    out = pd.DataFrame()
    out["activity_date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["activity_type"] = df[type_col].astype(str)
    out["moving_seconds"] = _to_seconds(df[moving_col]) if moving_col else np.nan
    out["distance_mi"] = _distance_to_miles(df[dist_col], out["moving_seconds"])
    out["avg_hr"] = pd.to_numeric(df[hr_col], errors="coerce") if hr_col else np.nan
    out["elev_gain"] = pd.to_numeric(df[elev_col], errors="coerce") if elev_col else np.nan
    out["activity_id"] = pd.to_numeric(df[id_col], errors="coerce") if id_col else np.nan

    # Drop bad rows (require valid date and positive distance for run-focused pipeline)
    out = out.dropna(subset=["activity_date", "distance_mi"])
    out = out[out["distance_mi"] > 0]

    return out


def load_strava_activities_lifetime() -> pd.DataFrame:
    """
    Load ALL activities from the Strava export with no date or distance filter.
    For lifetime/fun insights: every activity type (Run, Ride, Yoga, etc.), including
    zero-distance activities. Optionally includes name, kudos, relative_effort if
    present in the export (Strava bulk export column names vary; API exports may have more).
    """
    path = find_activities_csv()
    df = pd.read_csv(path)

    id_col = _pick_col(df, ["Activity ID", "id"])
    date_col = _pick_col(df, ["Activity Date", "start_date_local", "start_date"])
    type_col = _pick_col(df, ["Activity Type", "sport_type", "type", "Type"])
    dist_col = _pick_col(df, ["Distance", "distance"])
    moving_col = _pick_col(df, ["Moving Time", "moving_time", "Timer Time", "Timer Time.1", "elapsed_time", "Elapsed Time"])
    hr_col = _pick_col(df, ["Average Heart Rate", "average_heartrate"])
    elev_col = _pick_col(df, ["Elevation Gain", "total_elevation_gain"])

    if date_col is None or type_col is None:
        raise ValueError("Missing required columns. Need at least: date, activity type.")

    out = pd.DataFrame()
    out["activity_date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["activity_type"] = df[type_col].astype(str)
    out["moving_seconds"] = _to_seconds(df[moving_col]) if moving_col else np.nan
    # Distance: compute where possible, otherwise 0 (e.g. Yoga, Weight Training)
    if dist_col is not None:
        out["distance_mi"] = _distance_to_miles(df[dist_col], out["moving_seconds"])
        out["distance_mi"] = out["distance_mi"].fillna(0).clip(lower=0)
    else:
        out["distance_mi"] = 0.0
    out["avg_hr"] = pd.to_numeric(df[hr_col], errors="coerce") if hr_col else np.nan
    out["elev_gain"] = pd.to_numeric(df[elev_col], errors="coerce") if elev_col else np.nan
    out["activity_id"] = pd.to_numeric(df[id_col], errors="coerce") if id_col else np.nan

    # Optional columns for fun insights (may not be in all exports)
    name_col = _pick_col(df, ["Activity Name", "name", "title"])
    out["name"] = df[name_col].astype(str).replace("nan", "") if name_col else ""

    kudos_col = _pick_col(df, ["Kudos", "kudos", "kudos_count"])
    out["kudos"] = pd.to_numeric(df[kudos_col], errors="coerce") if kudos_col else np.nan

    effort_col = _pick_col(df, ["Relative Effort", "relative_effort", "suffer_score"])
    out["relative_effort"] = pd.to_numeric(df[effort_col], errors="coerce") if effort_col else np.nan

    # Drop only rows with invalid date
    out = out.dropna(subset=["activity_date"])
    return out