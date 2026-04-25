"""
Marathon readiness scoring model.

Produces a 0–100 composite score from five components, each independently
scored 0–100 then weighted. Weights reflect a sub-3 marathon build priority:
volume and long run are equally critical; consistency is the next lever;
AES trend and ramp risk are supporting signals.

Component weights:
  volume_score      25%  — Are you hitting target weekly mileage?
  long_run_score    25%  — Is the long run approaching race-specific distance?
  consistency_score 20%  — Have you been running regularly for the past 8 weeks?
  aes_trend_score   15%  — Is aerobic efficiency trending in the right direction?
  ramp_score        15%  — Is training load increasing at a safe rate?

All thresholds are sourced from config.py.
"""
import numpy as np
import pandas as pd

from config import TARGET_WEEKLY_MILES, PEAK_LONG_RUN_MI, RAMP_THRESHOLD


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, float(value)))


def compute_readiness(weekly: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """
    Compute per-week readiness scores and attach them to the weekly DataFrame.

    Added columns:
      volume_score      — 0–100, miles vs. target
      long_run_score    — 0–100, longest run vs. peak target
      consistency_score — 0–100, fraction of last 8 weeks with ≥1 run
      aes_trend_score   — 0, 50, or 100 based on AES slope over last 4 weeks
      ramp_score        — 0–100, penalty for exceeding RAMP_THRESHOLD
      readiness_score   — weighted composite (0–100, 1 decimal)
      readiness_label   — plain-English label

    Fix 4: During Taper weeks, volume and long run scores use peak-block values
    so the readiness card isn't artificially deflated by intentional cutback.

    Args:
        weekly: DataFrame output from build_datasets.py with at minimum:
                miles, long_run_miles, runs, aes_mean, ramp_rate columns.
                If a 'phase' column is present (added by classify_training_phase),
                taper-context scoring is applied automatically.
        cfg:    Optional config object with TARGET_WEEKLY_MILES, PEAK_LONG_RUN_MI,
                RAMP_THRESHOLD attributes. Defaults to config.py values when None.
                Pass a types.SimpleNamespace from Streamlit sliders to override
                thresholds at runtime without touching config.py.

    Returns:
        Same DataFrame with readiness columns appended.
    """
    # Resolve thresholds: caller-supplied cfg overrides module-level defaults.
    target_miles = getattr(cfg, "TARGET_WEEKLY_MILES", TARGET_WEEKLY_MILES)
    peak_long_run = getattr(cfg, "PEAK_LONG_RUN_MI", PEAK_LONG_RUN_MI)
    ramp_thresh = getattr(cfg, "RAMP_THRESHOLD", RAMP_THRESHOLD)

    weekly = weekly.sort_values("week_start").copy()

    # ------------------------------------------------------------------
    # Fix 4: Taper-adjusted volume and long run scores.
    # During Taper phase, use the peak 4-week average miles (not current
    # week) for the volume score, and the peak long run from the entire
    # block for the long run score. This prevents readiness from collapsing
    # during the intentional cutback that precedes race day.
    # ------------------------------------------------------------------
    has_phase = "phase" in weekly.columns

    # Pre-compute peak values across the entire block
    peak_4w_avg_miles = weekly["miles"].dropna().rolling(4, min_periods=1).mean().max()
    peak_long_run_actual = weekly["long_run_miles"].dropna().max() if weekly["long_run_miles"].notna().any() else 0.0

    # ------------------------------------------------------------------
    # Volume score (25%)
    # Fraction of TARGET_WEEKLY_MILES achieved, capped at 100.
    # During Taper: use peak 4-week average instead of current week miles.
    # ------------------------------------------------------------------
    def _volume_score(row):
        m = row["miles"]
        is_taper = has_phase and row.get("phase") == "Taper"
        effective_miles = peak_4w_avg_miles if (is_taper and pd.notna(peak_4w_avg_miles)) else m
        return _clamp((effective_miles or 0) / target_miles) * 100 if pd.notna(effective_miles) else 0.0

    weekly["volume_score"] = weekly.apply(_volume_score, axis=1)

    # ------------------------------------------------------------------
    # Long run score (25%)
    # Fraction of PEAK_LONG_RUN_THRESHOLD achieved, capped at 100.
    # During Taper: use peak long run from the entire training block.
    # ------------------------------------------------------------------
    def _long_run_score(row):
        lr = row["long_run_miles"]
        is_taper = has_phase and row.get("phase") == "Taper"
        effective_lr = peak_long_run_actual if (is_taper and pd.notna(peak_long_run_actual) and peak_long_run_actual > 0) else lr
        return _clamp((effective_lr or 0) / peak_long_run) * 100 if pd.notna(effective_lr) else 0.0

    weekly["long_run_score"] = weekly.apply(_long_run_score, axis=1)

    # ------------------------------------------------------------------
    # Consistency score (20%)
    # Count of weeks in the trailing 8-week window (inclusive) where the
    # athlete logged at least one run, divided by 8.
    # Uses min_periods=1 so early weeks aren't penalized for lacking history.
    # ------------------------------------------------------------------
    has_run = (weekly["runs"].fillna(0) >= 1).astype(int)
    weekly["consistency_score"] = (
        has_run.rolling(8, min_periods=1).sum() / 8.0 * 100
    )

    # ------------------------------------------------------------------
    # AES trend score (15%)
    # AES = pace_sec_mi / avg_hr — lower means more efficient (faster at same HR).
    # We look at the slope of aes_mean over the last 4 available weekly values:
    #   Improving (slope < 0, ≥1% change): 100  — engine getting cleaner
    #   Flat      (|slope| < 1% change):    50  — holding steady
    #   Declining (slope > 0, ≥1% change):   0  — efficiency dropping
    # Weeks with fewer than 2 valid data points default to 50 (neutral).
    # ------------------------------------------------------------------
    aes_vals = weekly["aes_mean"].values
    aes_scores = []
    for i in range(len(aes_vals)):
        window = aes_vals[max(0, i - 3): i + 1]
        valid = window[~np.isnan(window.astype(float))]
        if len(valid) < 2:
            aes_scores.append(50.0)
            continue
        slope = np.polyfit(range(len(valid)), valid, 1)[0]
        pct_change = abs(slope) / valid[0] if valid[0] != 0 else 0.0
        if slope < 0 and pct_change >= 0.01:   # improving efficiency
            aes_scores.append(100.0)
        elif pct_change < 0.01:                # effectively flat
            aes_scores.append(50.0)
        else:                                  # declining efficiency
            aes_scores.append(0.0)
    weekly["aes_trend_score"] = aes_scores

    # ------------------------------------------------------------------
    # Ramp score (15%)
    # Penalizes weeks where load increased faster than RAMP_THRESHOLD.
    # Formula: 1 - (ramp_rate / RAMP_THRESHOLD), clamped to [0, 1].
    # At ramp_rate = 0%:  score = 100 (no increase)
    # At ramp_rate = 15%: score = 0   (hit the threshold exactly)
    # At ramp_rate > 15%: score = 0   (over threshold, clamped)
    # NaN ramp (first week or missing data) receives 100 — no penalty for
    # unknown, since we can't flag a risk we can't measure.
    # ------------------------------------------------------------------
    weekly["ramp_score"] = weekly["ramp_rate"].apply(
        lambda r: _clamp(1.0 - (r / ramp_thresh)) * 100
        if pd.notna(r) else 100.0
    )

    # ------------------------------------------------------------------
    # Composite readiness score
    # ------------------------------------------------------------------
    weekly["readiness_score"] = (
        0.25 * weekly["volume_score"]
        + 0.25 * weekly["long_run_score"]
        + 0.20 * weekly["consistency_score"]
        + 0.15 * weekly["aes_trend_score"]
        + 0.15 * weekly["ramp_score"]
    ).round(1)

    # ------------------------------------------------------------------
    # Readiness label
    # ------------------------------------------------------------------
    def _label(score: float) -> str:
        if score <= 40:
            return "Not Ready"
        elif score <= 65:
            return "Building"
        elif score <= 80:
            return "On Track"
        else:
            return "Race Ready"

    weekly["readiness_label"] = weekly["readiness_score"].apply(_label)

    return weekly


# ---------------------------------------------------------------------------
# Projected finish time
# ---------------------------------------------------------------------------

def _fmt_hms(minutes: float) -> str:
    """Format decimal minutes as H:MM:SS."""
    total_sec = int(round(minutes * 60))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h}:{m:02d}:{s:02d}"


def compute_projected_time(
    runs_df: pd.DataFrame,
    weekly_df: pd.DataFrame = None,
    cfg=None,
) -> dict:
    """
    Estimate marathon finish time using Riegel's endurance formula with
    tier-based qualifying run selection, a dynamic athlete-calibrated exponent,
    and a taper supercompensation adjustment.

    Steps:
      1. Select qualifying runs by effort zone tier (Fix 1):
           Tier 1: effort_zone in ['Marathon', 'Hard'] AND distance_mi >= 8
           Tier 2: effort_zone == 'Moderate' AND distance_mi >= 8
           Tier 3: any non-Easy effort_zone AND distance_mi >= 8
         Runs from the taper window are excluded to prevent easy taper miles
         from contaminating the pace estimate.
         Up to 5 runs are selected from the best available tier, with combined
         pace-and-distance weighting: w = (1/pace_sec_mi) * distance_mi.

      2. Apply Riegel's formula with dynamic exponent (Fix 3):
           T2 = T1 × (D2/D1)^exponent
         where exponent starts at 1.06 and is reduced by 0.005 for each
         of these athlete-quality signals that are met (floor: 1.045):
           - consistency_score >= 80
           - aes_trend_score   >= 75
           - volume_score      >= 75

      3. AES efficiency adjustment (unchanged):
         ±2% for improving/declining aerobic efficiency over last 4 weeks.

      4. Taper supercompensation (Fix 2):
         If current_phase == "Taper" and within 3 weeks of last training date,
         apply a 3% speed bonus (predicted_sec *= 0.97).

      5. Confidence interval based on data quality flags.

    Returns a dict with predicted_str, lower_str, upper_str, confidence_note.

    Args:
        runs_df:   runs_enriched.csv DataFrame with pace_sec_mi, distance_mi,
                   activity_date, effort_zone columns.
        weekly_df: Optional weekly_model DataFrame. If phase/score columns are
                   present, taper detection and dynamic exponent are applied.
        cfg:       Optional config object (accepted for interface consistency).
    """
    _EMPTY = {
        "predicted_minutes": None,
        "predicted_str": "—",
        "lower_str": "—",
        "upper_str": "—",
        "confidence_note": "Not enough long runs to project",
        "delta_vs_goal_str": None,
        "delta_raw": None,
    }

    # ------------------------------------------------------------------
    # Detect current phase and taper window
    # ------------------------------------------------------------------
    current_phase = None
    taper_start_date = None

    if weekly_df is not None and "phase" in weekly_df.columns:
        wk_sorted = weekly_df.sort_values("week_start")
        current_phase = wk_sorted["phase"].iloc[-1] if len(wk_sorted) > 0 else None

        # Find earliest taper week to exclude those runs from qualifying pool
        taper_weeks = wk_sorted[wk_sorted["phase"] == "Taper"]["week_start"]
        if len(taper_weeks) > 0:
            taper_start_date = pd.to_datetime(taper_weeks.iloc[0])

    # ------------------------------------------------------------------
    # Fix 1: Tier-based qualifying run selection
    # ------------------------------------------------------------------
    has_effort_zone = "effort_zone" in runs_df.columns

    # Base pool: ≥8 miles, valid pace
    pool = (
        runs_df[runs_df["distance_mi"] >= 8]
        .dropna(subset=["pace_sec_mi"])
        .copy()
    )
    pool["activity_date"] = pd.to_datetime(pool["activity_date"], errors="coerce")

    # Exclude taper window runs if we can identify when taper began
    if taper_start_date is not None:
        pre_taper = pool[pool["activity_date"] < taper_start_date]
        # Fall back to full pool only if pre-taper is completely empty
        pool_to_use = pre_taper if len(pre_taper) > 0 else pool
    else:
        pool_to_use = pool

    qualifying = None
    if has_effort_zone and len(pool_to_use) > 0:
        # Tier 1: Marathon or Hard efforts
        t1 = pool_to_use[pool_to_use["effort_zone"].isin(["Marathon", "Hard"])]
        if len(t1) >= 3:
            qualifying = t1.sort_values("activity_date").tail(5)

        # Tier 2: Moderate efforts
        if qualifying is None:
            t2 = pool_to_use[pool_to_use["effort_zone"] == "Moderate"]
            if len(t2) >= 2:
                qualifying = t2.sort_values("activity_date").tail(5)

        # Tier 3: Any non-Easy
        if qualifying is None:
            t3 = pool_to_use[pool_to_use["effort_zone"] != "Easy"]
            if len(t3) > 0:
                qualifying = t3.sort_values("activity_date").tail(5)

    # Fallback: use full pool (original behavior) if no zone data or no matches
    if qualifying is None or len(qualifying) == 0:
        qualifying = pool_to_use.sort_values("activity_date").tail(5)
        if len(qualifying) == 0:
            qualifying = pool.sort_values("activity_date").tail(5)

    n_qual = len(qualifying)
    if n_qual == 0:
        return _EMPTY

    # ------------------------------------------------------------------
    # Fix 1 (cont): Combined pace-and-distance weighting
    # w = (1/pace_sec_mi) × distance_mi  — faster AND longer runs count most
    # ------------------------------------------------------------------
    paces   = qualifying["pace_sec_mi"].values.astype(float)
    dists   = qualifying["distance_mi"].values.astype(float)
    weights = (1.0 / paces) * dists
    weights = weights / weights.sum()   # normalise

    weighted_pace        = float(np.dot(paces, weights))
    avg_ref_distance     = float(np.dot(dists, weights))

    # ------------------------------------------------------------------
    # Fix 3: Dynamic Riegel exponent
    # Start at 1.06; subtract 0.005 per quality signal met; floor at 1.045.
    # ------------------------------------------------------------------
    base_exponent = 1.06
    exponent = base_exponent

    if weekly_df is not None:
        wk_sorted_scores = weekly_df.sort_values("week_start")
        last_week = wk_sorted_scores.iloc[-1] if len(wk_sorted_scores) > 0 else None

        if last_week is not None:
            cons  = last_week.get("consistency_score", None)
            aes_t = last_week.get("aes_trend_score", None)
            vol   = last_week.get("volume_score", None)

            if cons is not None and pd.notna(cons) and float(cons) >= 80:
                exponent -= 0.005
            if aes_t is not None and pd.notna(aes_t) and float(aes_t) >= 75:
                exponent -= 0.005
            if vol is not None and pd.notna(vol) and float(vol) >= 75:
                exponent -= 0.005

    exponent = max(exponent, 1.045)

    # Riegel's formula: T2 = T1 × (D2/D1)^exponent
    # Equivalent form using pace: T2 = weighted_pace × D2 × (D2/D1)^(exponent-1)
    # = weighted_pace × 26.2188 × (26.2188 / avg_ref_distance)^(exponent - 1)
    riegel_exp = exponent - 1.0   # the exponent on (D2/D1) in the pace form
    predicted_sec = weighted_pace * 26.2188 * (26.2188 / avg_ref_distance) ** riegel_exp

    # ------------------------------------------------------------------
    # AES efficiency adjustment (unchanged)
    # AES = pace/HR; a *negative* slope means AES is falling = improving.
    # ------------------------------------------------------------------
    aes_adj  = 1.0
    aes_note = ""
    if weekly_df is not None and "aes_mean" in weekly_df.columns:
        recent_aes = (
            weekly_df.sort_values("week_start")["aes_mean"].dropna().tail(4)
        )
        if len(recent_aes) >= 2:
            slope = float(np.polyfit(range(len(recent_aes)), recent_aes.values, 1)[0])
            pct   = abs(slope) / float(recent_aes.iloc[0]) if recent_aes.iloc[0] != 0 else 0.0
            if slope < 0 and pct >= 0.01:   # improving — predict faster
                aes_adj  = 0.98
                aes_note = "AES trending up"
            elif slope > 0 and pct >= 0.01: # declining — predict slower
                aes_adj  = 1.02
                aes_note = "AES trending down"
            else:
                aes_note = "AES steady"

    predicted_sec *= aes_adj

    # ------------------------------------------------------------------
    # Fix 2: Taper supercompensation
    # When in Taper phase and within 3 weeks of last training date, apply
    # a 3% speed bonus to account for rest-induced fitness gains.
    # ------------------------------------------------------------------
    taper_bonus_applied = False
    if current_phase == "Taper" and weekly_df is not None:
        last_training_date = weekly_df.sort_values("week_start")["week_start"].iloc[-1]
        last_training_date = pd.to_datetime(last_training_date)
        # Check if we're within 3 weeks of the end of the training data
        # (proxy for "close to race day")
        today = pd.Timestamp.today().normalize()
        days_to_end = (last_training_date - today).days + 7  # +7 for week duration
        if days_to_end <= 21:  # within 3 weeks
            predicted_sec *= 0.97
            taper_bonus_applied = True

    predicted_minutes  = predicted_sec / 60.0

    # ------------------------------------------------------------------
    # Confidence interval
    # ------------------------------------------------------------------
    uncertainty = 3.0  # base ±3 min

    if n_qual < 4:
        uncertainty += 1.0

    if weekly_df is not None and "aes_mean" in weekly_df.columns:
        aes_missing_frac = float(weekly_df["aes_mean"].isna().mean())
        if aes_missing_frac > 0.30:
            uncertainty += 1.0

    if weekly_df is not None and "ramp_score" in weekly_df.columns:
        last_ramp = weekly_df.sort_values("week_start")["ramp_score"].dropna()
        if len(last_ramp) > 0 and float(last_ramp.iloc[-1]) < 30:
            uncertainty += 1.0

    if weekly_df is not None and "consistency_score" in weekly_df.columns:
        last_cons = weekly_df.sort_values("week_start")["consistency_score"].dropna()
        if len(last_cons) > 0 and float(last_cons.iloc[-1]) > 85:
            uncertainty -= 0.5

    lower_minutes = predicted_minutes - uncertainty
    upper_minutes = predicted_minutes + uncertainty

    # Confidence note
    tier_label = ""
    if has_effort_zone and qualifying is not None and len(qualifying) > 0:
        zones = qualifying["effort_zone"].unique().tolist() if "effort_zone" in qualifying.columns else []
        if any(z in zones for z in ["Marathon", "Hard"]):
            tier_label = "quality effort tier"
        elif "Moderate" in zones:
            tier_label = "moderate effort tier"

    parts = [f"Based on {n_qual} run{'s' if n_qual != 1 else ''}"]
    if tier_label:
        parts.append(tier_label)
    if aes_note:
        parts.append(aes_note)
    if taper_bonus_applied:
        parts.append("taper bonus applied")
    if exponent < base_exponent:
        parts.append(f"exponent {exponent:.3f}")

    return {
        "predicted_minutes": predicted_minutes,
        "predicted_str":     _fmt_hms(predicted_minutes),
        "lower_str":         _fmt_hms(lower_minutes),
        "upper_str":         _fmt_hms(upper_minutes),
        "confidence_note":   " · ".join(parts),
        "delta_vs_goal_str": None,   # caller sets this after knowing goal_minutes
        "delta_raw":         None,
    }
