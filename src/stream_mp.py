import gzip
import os
import glob
from typing import Optional, Dict, List, Tuple

import pandas as pd
from fitparse import FitFile

# Constants
METERS_PER_MILE = 1609.344


def find_fit_files(raw_root: str) -> List[str]:
    """
    Find all .fit and .fit.gz files under the Strava export directory.
    Kept for backward compatibility; prefer build_fit_index_from_csv when
    activities.csv is available (Strava bulk exports name FIT files with
    Garmin internal IDs, not Strava Activity IDs).
    """
    patterns = [
        os.path.join(raw_root, "**", "activities", "*.fit"),
        os.path.join(raw_root, "**", "activities", "*.FIT"),
        os.path.join(raw_root, "**", "activities", "*.fit.gz"),
        os.path.join(raw_root, "**", "*.fit"),
        os.path.join(raw_root, "**", "*.FIT"),
        os.path.join(raw_root, "**", "*.fit.gz"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    # De-dupe
    return sorted(list(set(files)))


def build_fit_index_from_csv(activities_csv_path: str) -> Dict[int, str]:
    """
    Build a mapping {strava_activity_id: fit_file_path} using the Filename
    column in activities.csv.

    Strava bulk exports name FIT files with Garmin internal IDs — completely
    different numbers from Strava Activity IDs. The Filename column in
    activities.csv is the only reliable bridge between the two.

    Prefers plain .fit over .fit.gz when both exist (no decompression needed
    at parse time). Falls back to .fit.gz when only the gzipped file is present.

    Args:
        activities_csv_path: Absolute or relative path to activities.csv,
                             e.g. data/raw/strava_export/export_xyz/activities.csv.
                             FIT file paths in the Filename column are resolved
                             relative to the directory containing this CSV.

    Returns:
        Dict mapping int Strava Activity ID → str absolute/relative file path.
        Only includes rows where the file actually exists on disk.
    """
    base_dir = os.path.dirname(activities_csv_path)
    df = pd.read_csv(activities_csv_path)

    id_col = next((c for c in ["Activity ID", "id"] if c in df.columns), None)
    fn_col = "Filename" if "Filename" in df.columns else None

    if id_col is None or fn_col is None:
        return {}

    idx: Dict[int, str] = {}
    for _, row in df.iterrows():
        fn = row.get(fn_col)
        aid_raw = row.get(id_col)
        if pd.isna(fn) or fn == "" or pd.isna(aid_raw):
            continue
        try:
            aid = int(aid_raw)
        except (ValueError, TypeError):
            continue

        full_path = os.path.join(base_dir, str(fn))

        # Prefer plain .fit (no decompression) over .fit.gz when both exist
        if full_path.endswith(".fit.gz"):
            plain = full_path[:-3]  # strip .gz
            if os.path.exists(plain):
                full_path = plain

        if os.path.exists(full_path):
            idx[aid] = full_path

    return idx


def build_fit_index_by_activity_id(fit_files: List[str]) -> Dict[int, str]:
    """
    DEPRECATED — use build_fit_index_from_csv instead.

    Build a mapping {activity_id: filepath} by extracting any long integer
    token in the filename. Only works when Strava names FIT files with the
    Strava Activity ID (rare in bulk exports; common in older API exports).
    """
    idx: Dict[int, str] = {}
    for fp in fit_files:
        base = os.path.basename(fp)
        tokens = []
        cur = ""
        for ch in base:
            if ch.isdigit():
                cur += ch
            else:
                if cur:
                    tokens.append(cur)
                    cur = ""
        if cur:
            tokens.append(cur)
        if not tokens:
            continue
        tok = max(tokens, key=len)
        if len(tok) < 8:
            continue
        try:
            aid = int(tok)
        except ValueError:
            continue
        if aid not in idx:
            idx[aid] = fp
    return idx


def pace_sec_per_mile_from_speed_mps(speed_mps: float) -> Optional[float]:
    """
    Convert speed in meters/second to pace in seconds/mile.
    """
    if speed_mps is None or speed_mps <= 0:
        return None
    return METERS_PER_MILE / float(speed_mps)


def _iter_fit_records(fit_path: str):
    """
    Yield dicts from FIT 'record' messages.

    Handles both plain .fit and gzip-compressed .fit.gz files.
    fitparse cannot open .fit.gz by path (invalid header); decompress
    to bytes first and pass the raw bytes to FitFile instead.
    """
    if fit_path.endswith(".gz"):
        with gzip.open(fit_path, "rb") as fh:
            raw = fh.read()
        fit = FitFile(raw)
    else:
        fit = FitFile(fit_path)
    for msg in fit.get_messages("record"):
        data = {d.name: d.value for d in msg}
        yield data


def mp_miles_from_fit(
    fit_path: str,
    goal_mp_sec: float,
    band_sec: float,
    require_contiguous_miles: float = 0.0,
    allow_gap_seconds: int = 0,
) -> float:
    """
    Compute miles at marathon pace (MP) from FIT stream records using distance deltas.

    - goal_mp_sec: target MP in seconds/mile (sub-3 ≈ 412)
    - band_sec: +/- band around goal pace (e.g., 20)
    - require_contiguous_miles: if > 0, only count segments that accumulate at least this many miles
      continuously in-band (useful to avoid counting tiny blips)
    - allow_gap_seconds: if > 0, allow short gaps (seconds) outside band within a segment.

    Returns: mp_miles (float)
    """
    low = goal_mp_sec - band_sec
    high = goal_mp_sec + band_sec

    mp_meters_total = 0.0

    # Segment tracking
    seg_meters = 0.0
    seg_gap_seconds = 0

    prev_dist = None
    prev_ts = None

    for rec in _iter_fit_records(fit_path):
        dist = rec.get("distance")  # meters
        speed = rec.get("speed")    # m/s
        ts = rec.get("timestamp")   # datetime

        if dist is None or speed is None:
            continue

        if prev_dist is None:
            prev_dist = dist
            prev_ts = ts
            continue

        delta_m = float(dist) - float(prev_dist)
        prev_dist = dist

        if delta_m <= 0:
            prev_ts = ts
            continue

        pace = pace_sec_per_mile_from_speed_mps(float(speed))
        if pace is None:
            prev_ts = ts
            continue

        in_band = (low <= pace <= high)

        # Estimate seconds between records if timestamp exists; else treat as 1 second
        dt_seconds = 1
        if prev_ts is not None and ts is not None:
            try:
                dt_seconds = int((ts - prev_ts).total_seconds())
                if dt_seconds <= 0:
                    dt_seconds = 1
            except Exception:
                dt_seconds = 1
        prev_ts = ts

        if in_band:
            seg_meters += delta_m
            seg_gap_seconds = 0
        else:
            # Out of band, allow small gaps if configured
            if allow_gap_seconds > 0 and seg_meters > 0:
                seg_gap_seconds += dt_seconds
                if seg_gap_seconds <= allow_gap_seconds:
                    # Keep segment alive, do not add meters
                    continue

            # Segment ends; decide whether to count it
            if require_contiguous_miles > 0:
                if seg_meters >= require_contiguous_miles * METERS_PER_MILE:
                    mp_meters_total += seg_meters
            else:
                mp_meters_total += seg_meters

            # Reset segment
            seg_meters = 0.0
            seg_gap_seconds = 0

    # Close final segment
    if seg_meters > 0:
        if require_contiguous_miles > 0:
            if seg_meters >= require_contiguous_miles * METERS_PER_MILE:
                mp_meters_total += seg_meters
        else:
            mp_meters_total += seg_meters

    return mp_meters_total / METERS_PER_MILE


def miles_in_pace_bands_from_fit(
    fit_path: str,
    bands_sec_per_mile: List[Tuple[float, float]],
) -> List[float]:
    """
    Generic utility: compute miles in multiple pace bands from FIT stream.
    bands_sec_per_mile: list of (low_sec, high_sec) inclusive.

    Returns list of miles for each band.
    """
    miles_by_band = [0.0 for _ in bands_sec_per_mile]

    prev_dist = None
    for rec in _iter_fit_records(fit_path):
        dist = rec.get("distance")
        speed = rec.get("speed")
        if dist is None or speed is None:
            continue
        if prev_dist is None:
            prev_dist = dist
            continue

        delta_m = float(dist) - float(prev_dist)
        prev_dist = dist
        if delta_m <= 0:
            continue

        pace = pace_sec_per_mile_from_speed_mps(float(speed))
        if pace is None:
            continue

        for i, (low, high) in enumerate(bands_sec_per_mile):
            if low <= pace <= high:
                miles_by_band[i] += delta_m / METERS_PER_MILE

    return miles_by_band