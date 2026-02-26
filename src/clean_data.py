import os
import glob
import pandas as pd


RAW_DIR = os.path.join("data", "raw", "strava_export")
PROCESSED_DIR = os.path.join("data", "processed")

# We will look for activities.csv anywhere under RAW_DIR
ACTIVITIES_GLOB = os.path.join(RAW_DIR, "**", "activities.csv")

# Columns that often contain location / route / personally identifying info
SENSITIVE_COLS = {
    "start_latlng",
    "end_latlng",
    "map",
    "summary_polyline",
    "polyline",
    "location_city",
    "location_state",
    "location_country",
    "start_latitude",
    "start_longitude",
    "end_latitude",
    "end_longitude",
    "latitude",
    "longitude",
    "upload_id",
    "external_id",
    "from_accepted_tag",
    "manual",
    "commute",
    "trainer",
}


def find_activities_csv() -> str:
    matches = glob.glob(ACTIVITIES_GLOB, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find activities.csv under {RAW_DIR}. "
            f"Expected something like: {ACTIVITIES_GLOB}"
        )
    # If multiple exist, pick the first (Strava export usually has one)
    return matches[0]


def sanitize_activities(df: pd.DataFrame) -> pd.DataFrame:
    # Drop sensitive columns if present
    cols_to_drop = [c for c in df.columns if c in SENSITIVE_COLS]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Normalize / parse date fields if they exist
    for col in ["start_date", "start_date_local"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Keep a clean, consistent subset if these columns exist
    keep_priority = [
        "id",
        "name",
        "type",
        "sport_type",
        "start_date_local",
        "distance",
        "moving_time",
        "elapsed_time",
        "total_elevation_gain",
        "average_speed",
        "max_speed",
        "average_heartrate",
        "max_heartrate",
        "calories",
        "kudos_count",
        "achievement_count",
        "gear_id",
    ]
    existing_keep = [c for c in keep_priority if c in df.columns]
    # If we found a good subset, use it; otherwise keep everything minus sensitive cols
    if existing_keep:
        df = df[existing_keep]

    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    activities_path = find_activities_csv()
    print(f"Reading: {activities_path}")

    df = pd.read_csv(activities_path)

    print(f"Rows: {len(df):,} | Cols: {len(df.columns)}")
    cleaned = sanitize_activities(df)

    out_path = os.path.join(PROCESSED_DIR, "cleaned_activities.csv")
    cleaned.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Cleaned rows: {len(cleaned):,} | Cleaned cols: {len(cleaned.columns)}")


if __name__ == "__main__":
    main()