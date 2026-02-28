import os
import glob
from typing import Optional, Dict, List, Tuple

from fitparse import FitFile

# Constants
METERS_PER_MILE = 1609.344


def find_fit_files(raw_root: str) -> List[str]:
    """
    Find all .fit files under the Strava export directory.
    Strava exports usually contain an 'activities' folder with FIT/GPX/TCX files.
    """
    patterns = [
        os.path.join(raw_root, "**", "activities", "*.fit"),
        os.path.join(raw_root, "**", "activities", "*.FIT"),
        os.path.join(raw_root, "**", "*.fit"),
        os.path.join(raw_root, "**", "*.FIT"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    # De-dupe
    return sorted(list(set(files)))


def build_fit_index_by_activity_id(fit_files: List[str]) -> Dict[int, str]:
    """
    Build a mapping {activity_id: filepath} by extracting any long integer token in the filename.
    This works when Strava names files with the activity id (common). If export naming changes
    or you add non-Strava FITs, consider a fallback (e.g. match by date + duration) or a small
    config mapping activity_id -> path.
    """
    idx: Dict[int, str] = {}
    for fp in fit_files:
        base = os.path.basename(fp)
        # Extract all integer tokens from filename
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

        # Pick the longest numeric token as the likely activity id
        if not tokens:
            continue
        tok = max(tokens, key=len)
        # Heuristic: activity ids are usually 8+ digits
        if len(tok) < 8:
            continue
        try:
            aid = int(tok)
        except ValueError:
            continue

        # Keep first occurrence if duplicates
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
    """
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