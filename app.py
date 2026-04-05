"""
Strava Training Intelligence — Streamlit Dashboard
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make src/ importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config
from phases import classify_training_phase
from readiness import compute_readiness, compute_projected_time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strava Training Intelligence",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Visual constants ───────────────────────────────────────────────────────────
PHASE_COLORS = {
    "Base":  "#4299e1",
    "Build": "#ed8936",
    "Peak":  "#e53e3e",
    "Taper": "#805ad5",
    "Rest":  "#a0aec0",
}
PHASE_EMOJIS = {
    "Base":  "🏃",
    "Build": "🔨",
    "Peak":  "🔥",
    "Taper": "🌀",
    "Rest":  "😴",
}

# Columns recomputed live from sidebar sliders — dropped from cached base data
# so classify_training_phase and compute_readiness always run fresh.
_RECOMPUTE_COLS = [
    "phase", "readiness_score", "readiness_label",
    "volume_score", "long_run_score", "consistency_score",
    "aes_trend_score", "ramp_score",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def readiness_color(score: float) -> str:
    if score <= 40:   return "#e53e3e"
    elif score <= 65: return "#dd6b20"
    elif score <= 80: return "#d69e2e"
    else:             return "#38a169"


def fmt_hms(minutes: float) -> str:
    """Format decimal minutes as H:MM:SS."""
    total_sec = int(round(minutes * 60))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h}:{m:02d}:{s:02d}"


def fmt_pace(sec_per_mile: float) -> str:
    """Format seconds/mile as M:SS pace string (e.g. 412 → '6:52')."""
    sec_per_mile = int(round(sec_per_mile))
    return f"{sec_per_mile // 60}:{sec_per_mile % 60:02d}"


def round_half(x: float) -> float:
    """Round to nearest 0.5."""
    return round(x * 2) / 2


def safe_float(val, default: float = 0.0) -> float:
    """Return float(val) or default if val is None/NaN."""
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


# ── Data loading ───────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")


@st.cache_data
def _load_runs() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(_DATA_DIR, "runs_enriched.csv"),
        parse_dates=["activity_date"],
    )


@st.cache_data
def _load_weekly_base() -> pd.DataFrame:
    """Load weekly model, stripping pre-computed phase/readiness columns.

    Stripping at load time means the cached result is always the 'raw'
    pipeline output. The app recomputes phase + readiness live from sliders.
    """
    df = pd.read_csv(
        os.path.join(_DATA_DIR, "weekly_model.csv"),
        parse_dates=["week_start"],
    )
    return df.drop(columns=[c for c in _RECOMPUTE_COLS if c in df.columns])


try:
    runs_raw    = _load_runs()
    weekly_base = _load_weekly_base()
except FileNotFoundError:
    st.error("⚠️ Run `python src/build_datasets.py` first to generate your training data.")
    st.stop()

# One-time HR data quality banner
if runs_raw["avg_hr"].isna().mean() > 0.5:
    st.info(
        "ℹ️ Heart rate data is limited. AES scores and ramp signals "
        "are estimated from volume only for affected weeks."
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Athlete Settings")

    # ── Athlete profile ──
    st.subheader("👤 Athlete Profile")
    athlete_name   = st.text_input("Athlete Name", value="Athlete")
    goal_race      = st.text_input("Goal Race",    value="Sub-3 Marathon")
    goal_race_date = st.date_input("Goal Race Date", value=None)
    goal_minutes   = float(st.number_input(
        "Goal time (minutes)", min_value=120, max_value=360, value=180, step=1
    ))

    # ── Training thresholds ──
    st.subheader("📊 Training Thresholds")
    target_miles  = st.slider("Weekly mileage target",   30, 70, int(config.TARGET_WEEKLY_MILES), step=5)
    base_build    = st.slider("Base → Build (mpw)",      25, 60, int(config.BASE_MAX_MILES),      step=5)
    peak_lr       = st.slider("Peak long run (mi)",      14, 22, int(config.PEAK_LONG_RUN_MI),    step=1)
    peak_spec_pct = st.slider("Peak MP specificity (%)", 10, 35, int(config.PEAK_SPECIFICITY * 100), step=5)
    taper_pct     = st.slider("Taper trigger (% drop)",  60, 85, int(config.TAPER_FACTOR * 100),  step=5)

    # SimpleNamespace carries slider values as cfg — passed to pipeline functions
    # so all thresholds update live without touching config.py.
    cfg = SimpleNamespace(
        TARGET_WEEKLY_MILES = target_miles,
        BASE_MAX_MILES      = base_build,
        BUILD_MIN_MILES     = base_build,
        PEAK_LONG_RUN_MI    = peak_lr,
        PEAK_SPECIFICITY    = peak_spec_pct / 100.0,
        TAPER_FACTOR        = taper_pct / 100.0,
        BUILD_START_WEEKS   = config.BUILD_START_WEEKS,
        RAMP_THRESHOLD      = config.RAMP_THRESHOLD,
    )

    # ── Data info ──
    st.divider()
    st.caption(
        f"Data: {runs_raw['activity_date'].min():%b %d} → "
        f"{runs_raw['activity_date'].max():%b %d, %Y}"
    )
    st.caption(f"{len(runs_raw)} runs · {len(weekly_base)} weeks")

# ── Recompute phase + readiness from sidebar cfg ───────────────────────────────
weekly, current_phase, phase_history = classify_training_phase(weekly_base.copy(), cfg)
weekly = compute_readiness(weekly, cfg)
weekly = weekly.sort_values("week_start").reset_index(drop=True)

cur  = weekly.iloc[-1]                                    # most recent week
prev = weekly.iloc[-2] if len(weekly) >= 2 else None      # previous week

# ── Projected finish time ──────────────────────────────────────────────────────
proj = compute_projected_time(runs_raw, weekly_df=weekly, cfg=cfg)
if proj["predicted_minutes"] is not None:
    delta_raw = proj["predicted_minutes"] - goal_minutes
    proj["delta_raw"]         = delta_raw
    proj["delta_vs_goal_str"] = f"{delta_raw:+.0f} min vs goal"
else:
    proj["delta_raw"]         = None
    proj["delta_vs_goal_str"] = None

goal_time_str = fmt_hms(goal_minutes)

# ── Header ─────────────────────────────────────────────────────────────────────
title_col, _ = st.columns([7, 3])
with title_col:
    st.title(f"🏃 {athlete_name}'s Training Intelligence")
    if goal_race_date:
        days_to_race  = (pd.Timestamp(goal_race_date) - pd.Timestamp.today()).days
        weeks_to_race = max(0, days_to_race // 7)
        st.caption(
            f"Goal: **{goal_race}** · Race in {days_to_race} days "
            f"({weeks_to_race} weeks) · Target: {goal_time_str}"
        )
    else:
        st.caption(f"Goal: **{goal_race}** · Set your race date in the sidebar")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Hero metrics
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)

with c1:
    emoji = PHASE_EMOJIS.get(current_phase, "🏃")
    st.metric("Training Phase", f"{emoji} {current_phase}")
    badges = " · ".join(f"{p}: {n}wk" for p, n in sorted(phase_history.items()))
    st.caption(badges)

with c2:
    score = safe_float(cur.get("readiness_score"), 0.0)
    delta_score = (score - safe_float(prev.get("readiness_score"), score)) if prev is not None else None
    delta_str   = f"{delta_score:+.1f} pts vs last week" if delta_score is not None else None
    st.metric("Readiness Score", f"{score:.0f} / 100", delta=delta_str)
    color = readiness_color(score)
    label = cur.get("readiness_label", "—")
    st.markdown(f'<span style="color:{color}; font-weight:600">{label}</span>', unsafe_allow_html=True)

with c3:
    delta_color_arg = "inverse" if proj["delta_raw"] is not None else "off"
    st.metric(
        "Projected Finish",
        proj["predicted_str"],
        delta=proj["delta_vs_goal_str"],
        delta_color=delta_color_arg,
    )
    st.caption(f"{proj['lower_str']} – {proj['upper_str']} · {proj['confidence_note']}")

with c4:
    if goal_race_date:
        days_to_race  = (pd.Timestamp(goal_race_date) - pd.Timestamp.today()).days
        weeks_to_race = max(0, days_to_race // 7)
        st.metric("Race Countdown", f"{weeks_to_race} weeks")
        if current_phase == "Peak" and weeks_to_race <= 4:
            st.warning("🌀 Taper should begin this week.")
        elif current_phase == "Taper":
            st.success("✅ Taper is active. Protect your legs.")
    else:
        st.metric("Race Countdown", "—")
        st.caption("Add your race date →")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Milestone Tracker
# ══════════════════════════════════════════════════════════════════════════════
st.subheader(f"🏁 {goal_race} Milestone Tracker")

wk = weekly.sort_values("week_start").copy()

# Helper: first week meeting a mask condition
def _first_week(mask_series, date_col="week_start") -> str | None:
    rows = wk[mask_series]
    return rows.iloc[0][date_col].strftime("%b %d") if not rows.empty else None

# M1: First week >= BASE_MAX_MILES
m1_date = _first_week(wk["miles"] >= cfg.BASE_MAX_MILES)

# M2: First long run >= PEAK_LONG_RUN_MI
m2_date = _first_week(wk["long_run_miles"].fillna(0) >= cfg.PEAK_LONG_RUN_MI)

# M3: Three consecutive weeks >= BASE_MAX_MILES
m3_date = None
above = (wk["miles"] >= cfg.BASE_MAX_MILES).tolist()
for i in range(2, len(above)):
    if above[i] and above[i - 1] and above[i - 2]:
        m3_date = wk.iloc[i]["week_start"].strftime("%b %d")
        break

# M4: First 20-mile long run + trajectory if not yet achieved
m4_date = _first_week(wk["long_run_miles"].fillna(0) >= 20)
m4_pending = None
if not m4_date:
    lr_valid = wk.dropna(subset=["long_run_miles"])
    cur_lr   = safe_float(lr_valid["long_run_miles"].iloc[-1] if not lr_valid.empty else None, 0.0)
    if len(lr_valid) >= 2:
        lr_slope = float(np.polyfit(range(len(lr_valid)), lr_valid["long_run_miles"].values, 1)[0])
        if lr_slope > 0:
            wks = int(np.ceil((20.0 - cur_lr) / lr_slope))
            est = (wk.iloc[-1]["week_start"] + pd.Timedelta(weeks=max(1, wks))).strftime("%b %d")
            m4_pending = f"~{est}"
        else:
            m4_pending = "— add 1–2 mi/week"
    else:
        m4_pending = "— building"

# M5: Peak week (miles >= TARGET_WEEKLY_MILES)
m5_date = _first_week(wk["miles"] >= cfg.TARGET_WEEKLY_MILES)
m5_pending = None
if not m5_date:
    cur_mi   = safe_float(wk["miles"].iloc[-1], 0.0)
    mi_slope = float(np.polyfit(range(len(wk)), wk["miles"].values, 1)[0]) if len(wk) >= 2 else 0.0
    if mi_slope > 0:
        wks = int(np.ceil((cfg.TARGET_WEEKLY_MILES - cur_mi) / mi_slope))
        est = (wk.iloc[-1]["week_start"] + pd.Timedelta(weeks=max(1, wks))).strftime("%b %d")
        m5_pending = f"~{est}"
    else:
        m5_pending = "— increase volume steadily"

# M6: Race Ready score >= 80
m6_date = _first_week(wk["readiness_score"].fillna(0) >= 80)

milestones = [
    {
        "label": f"First {int(cfg.BASE_MAX_MILES)}-mile week",
        "done":  bool(m1_date),
        "value": m1_date or "Not yet",
    },
    {
        "label": f"First {int(cfg.PEAK_LONG_RUN_MI)}-mile long run",
        "done":  bool(m2_date),
        "value": m2_date or "Not yet",
    },
    {
        "label": f"3× {int(cfg.BASE_MAX_MILES)}+ mi consecutive",
        "done":  bool(m3_date),
        "value": m3_date or "Not yet",
    },
    {
        "label": "First 20-mile long run",
        "done":  bool(m4_date),
        "value": m4_date or m4_pending or "Not yet",
    },
    {
        "label": f"Peak week ({int(cfg.TARGET_WEEKLY_MILES)} mi)",
        "done":  bool(m5_date),
        "value": m5_date or m5_pending or "Not yet",
    },
    {
        "label": "Race Ready (score ≥ 80)",
        "done":  bool(m6_date),
        "value": m6_date or f"Currently {score:.0f}",
    },
]

ms_cols = st.columns(6)
for col, ms in zip(ms_cols, milestones):
    with col:
        if ms["done"]:
            st.markdown(
                f"""<div style="background:#c6f6d5;border-radius:8px;padding:10px 8px;
                    text-align:center;min-height:88px">
                    <div style="font-size:20px">✅</div>
                    <div style="font-size:11px;font-weight:600;color:#276749;margin-top:4px">
                        {ms['label']}</div>
                    <div style="font-size:12px;font-weight:700;color:#22543d;margin-top:2px">
                        {ms['value']}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div style="background:#edf2f7;border-radius:8px;padding:10px 8px;
                    text-align:center;min-height:88px">
                    <div style="font-size:20px">⏳</div>
                    <div style="font-size:11px;color:#4a5568;margin-top:4px">
                        {ms['label']}</div>
                    <div style="font-size:12px;font-style:italic;color:#718096;margin-top:2px">
                        {ms['value']}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Weekly Mileage by Phase
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Weekly Mileage by Phase")

fig_miles = go.Figure()

# One bar trace per phase so each phase gets its legend color
for phase_name, ph_color in PHASE_COLORS.items():
    sub = wk[wk["phase"] == phase_name]
    if sub.empty:
        continue
    hover = [
        (
            f"<b>Week of {r['week_start'].strftime('%b %d')}</b><br>"
            f"Miles: {r['miles']:.1f}<br>"
            f"Phase: {r['phase']}<br>"
            + (f"Long run: {r['long_run_miles']:.1f} mi" if pd.notna(r.get("long_run_miles")) else "")
        )
        for _, r in sub.iterrows()
    ]
    fig_miles.add_trace(go.Bar(
        x=sub["week_start"], y=sub["miles"],
        name=phase_name,
        marker_color=ph_color,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))

# Long run diamonds, colored by phase
for phase_name, ph_color in PHASE_COLORS.items():
    sub = wk[(wk["phase"] == phase_name) & wk["long_run_miles"].notna()]
    if sub.empty:
        continue
    fig_miles.add_trace(go.Scatter(
        x=sub["week_start"], y=sub["long_run_miles"],
        name="Long Run" if phase_name == list(PHASE_COLORS)[0] else None,
        showlegend=(phase_name == list(PHASE_COLORS)[0]),
        mode="markers",
        marker=dict(symbol="diamond", size=9, color=ph_color,
                    line=dict(width=1.5, color="white")),
        hovertemplate="<b>Long run</b> %{x|%b %d}: %{y:.1f} mi<extra></extra>",
    ))

# Target mileage reference line
fig_miles.add_hline(
    y=cfg.TARGET_WEEKLY_MILES,
    line_dash="dash", line_color="#a0aec0",
    annotation_text=f"Target ({cfg.TARGET_WEEKLY_MILES} mi)",
    annotation_position="top right",
    annotation_font_color="#718096",
)

fig_miles.update_layout(
    template="plotly_white",
    barmode="overlay",
    xaxis=dict(title="", tickformat="%b %d"),
    yaxis_title="Miles per Week",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=40, r=20),
    height=350,
)
st.plotly_chart(fig_miles, use_container_width=True)

sc1, sc2 = st.columns(2)
with sc1:
    lr_max_row = wk.dropna(subset=["long_run_miles"]).sort_values("long_run_miles").iloc[-1]
    st.caption(
        f"Longest run this block: **{lr_max_row['long_run_miles']:.1f} mi** "
        f"({lr_max_row['week_start'].strftime('%b %d')})"
    )
with sc2:
    avg4 = wk.tail(4)["miles"].mean()
    st.caption(f"Avg weekly miles (last 4 weeks): **{avg4:.1f} mi**")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Readiness Trajectory
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Readiness Trajectory")

rd = wk.dropna(subset=["readiness_score"]).copy()

fig_rd = go.Figure()

# Reference bands
for y_val, band_color, band_label in [
    (40, "#e53e3e", "Not Ready"),
    (65, "#dd6b20", "Building"),
    (80, "#38a169", "Race Ready"),
]:
    fig_rd.add_hline(
        y=y_val, line_dash="dash", line_color=band_color, line_width=1,
        annotation_text=band_label,
        annotation_position="right",
        annotation_font_color=band_color,
    )

# Readiness line with per-point marker colors
if not rd.empty:
    fig_rd.add_trace(go.Scatter(
        x=rd["week_start"], y=rd["readiness_score"],
        mode="lines+markers",
        name="Readiness",
        line=dict(color="#4299e1", width=2),
        marker=dict(
            size=10,
            color=[readiness_color(float(s)) for s in rd["readiness_score"]],
            line=dict(width=1.5, color="white"),
        ),
        hovertemplate="<b>%{x|%b %d}</b><br>Readiness: %{y:.1f}<extra></extra>",
    ))

    # Linear trendline
    if len(rd) >= 3:
        xs = list(range(len(rd)))
        slope, intercept = np.polyfit(xs, rd["readiness_score"].values, 1)
        y_trend = [slope * x + intercept for x in xs]
        t_color = "#38a169" if slope > 0 else "#e53e3e"
        fig_rd.add_trace(go.Scatter(
            x=rd["week_start"], y=y_trend,
            mode="lines", name="Trend",
            line=dict(color=t_color, width=1.5, dash="dot"),
            showlegend=False,
        ))

    # Annotate most recent point
    last_rd = rd.iloc[-1]
    fig_rd.add_annotation(
        x=last_rd["week_start"],
        y=float(last_rd["readiness_score"]),
        text=f"  {float(last_rd['readiness_score']):.0f} — {last_rd['readiness_label']}",
        showarrow=False, xanchor="left",
        font=dict(size=11, color=readiness_color(float(last_rd["readiness_score"]))),
    )

    # 4-week delta annotation
    if len(rd) >= 4:
        delta_4 = float(rd.iloc[-1]["readiness_score"]) - float(rd.iloc[-4]["readiness_score"])
        arrow   = "↑" if delta_4 > 0 else "↓"
        d_color = "#38a169" if delta_4 > 0 else "#e53e3e"
        fig_rd.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.97,
            text=f"{arrow} {delta_4:+.0f} pts over last 4 weeks",
            showarrow=False, xanchor="left",
            font=dict(size=11, color=d_color),
        )

fig_rd.update_layout(
    template="plotly_white",
    xaxis=dict(title="", tickformat="%b %d"),
    yaxis=dict(title="Readiness Score (0–100)", range=[0, 105]),
    height=350,
    margin=dict(t=20, r=110),
)
st.plotly_chart(fig_rd, use_container_width=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — Aerobic Efficiency (AES)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("💚 Aerobic Efficiency (AES)")

aes_data     = wk.dropna(subset=["aes_mean"]).copy()
aes_null_pct = wk["aes_mean"].isna().mean()

fig_aes = go.Figure()

if not aes_data.empty:
    fig_aes.add_trace(go.Scatter(
        x=aes_data["week_start"], y=aes_data["aes_mean"],
        mode="lines+markers", name="AES",
        line=dict(color="#48bb78", width=2),
        marker=dict(size=7, color="#48bb78"),
        hovertemplate="<b>%{x|%b %d}</b><br>AES: %{y:.3f} sec/bpm<extra></extra>",
    ))

    if len(aes_data) >= 3:
        xs = list(range(len(aes_data)))
        slope, intercept = np.polyfit(xs, aes_data["aes_mean"].values, 1)
        y_trend  = [slope * x + intercept for x in xs]
        pct_chg  = abs(slope) / float(aes_data["aes_mean"].iloc[0]) * 100 \
                   if aes_data["aes_mean"].iloc[0] != 0 else 0.0

        # AES falling = efficiency improving (faster pace at same HR)
        if slope < 0 and pct_chg >= 1.0:
            t_color  = "#38a169"
            ann_text = f"↑ Improving (+{pct_chg:.1f}%)"
        elif slope > 0 and pct_chg >= 1.0:
            t_color  = "#e53e3e"
            ann_text = f"↓ Declining (-{pct_chg:.1f}%)"
        else:
            t_color  = "#a0aec0"
            ann_text = "→ Steady"

        fig_aes.add_trace(go.Scatter(
            x=aes_data["week_start"], y=y_trend,
            mode="lines", name="Trend",
            line=dict(color=t_color, width=1.5, dash="dot"),
            showlegend=False,
        ))
        fig_aes.add_annotation(
            xref="paper", yref="paper", x=0.99, y=0.97,
            text=ann_text, showarrow=False, xanchor="right",
            font=dict(size=11, color=t_color),
        )

    fig_aes.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.03,
        text="(Lower = more efficient)", showarrow=False, xanchor="left",
        font=dict(size=10, color="#718096"),
    )

fig_aes.update_layout(
    template="plotly_white",
    xaxis=dict(title="", tickformat="%b %d"),
    yaxis_title="AES (sec/mi per bpm)",
    height=300,
    margin=dict(t=20, r=20),
)
st.plotly_chart(fig_aes, use_container_width=True)

if aes_null_pct > 0.5:
    st.warning("⚠️ Limited heart rate data — AES trend may not be reliable.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 6 — Readiness Breakdown
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("⚖️ What's Driving Your Score")

_components = [
    ("Volume (25%)",             "volume_score",       0.25),
    ("Long Run (25%)",           "long_run_score",     0.25),
    ("Consistency (20%)",        "consistency_score",  0.20),
    ("Aerobic Efficiency (15%)", "aes_trend_score",    0.15),
    ("Ramp Safety (15%)",        "ramp_score",         0.15),
]
comp_scores = {col: safe_float(cur.get(col), 0.0) for _, col, _ in _components}

left_col, right_col = st.columns([6, 4])

with left_col:
    bar_vals    = [comp_scores[col] for _, col, _ in _components]
    bar_labels  = [label for label, _, _ in _components]
    bar_colors  = [
        "#38a169" if s >= 70 else "#d69e2e" if s >= 40 else "#e53e3e"
        for s in bar_vals
    ]
    fig_bd = go.Figure(go.Bar(
        x=bar_vals, y=bar_labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:.0f}" for s in bar_vals],
        textposition="auto",
    ))
    fig_bd.update_layout(
        template="plotly_white",
        xaxis=dict(range=[0, 100], title="Score (0–100)"),
        yaxis=dict(autorange="reversed"),
        height=240,
        margin=dict(t=10, b=10, l=10),
    )
    st.plotly_chart(fig_bd, use_container_width=True)

with right_col:
    st.write("")  # vertical spacing
    total_score = 0.0
    for label, col, weight in _components:
        s       = comp_scores[col]
        contrib = s * weight
        total_score += contrib
        short   = label.split(" (")[0]
        st.markdown(f"`{short:<20}` **{s:.0f}** × {weight:.2f} = {contrib:.1f}")
    st.markdown("---")
    color = readiness_color(total_score)
    st.metric("Readiness Score", f"{total_score:.1f}")
    st.markdown(
        f'<span style="color:{color}; font-weight:600">{cur.get("readiness_label", "—")}</span>',
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 6b — Training Intensity Distribution
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 Training Intensity Distribution")

# Block-wide zone totals from all runs (not just most recent week)
_zone_names   = ["Easy", "Moderate", "Marathon", "Hard"]
_zone_colors  = {"Easy": "#4299e1", "Moderate": "#ed8936",
                 "Marathon": "#e53e3e", "Hard": "#805ad5"}
_zone_labels  = {"Easy": "Easy", "Moderate": "Moderate",
                 "Marathon": "Marathon Pace", "Hard": "Hard"}

_total_mi = runs_raw["distance_mi"].sum()
_zone_mi  = {
    z: runs_raw[runs_raw["effort_zone"] == z]["distance_mi"].sum()
    for z in _zone_names
}
_zone_pct = {
    z: (_zone_mi[z] / _total_mi) if _total_mi > 0 else 0.0
    for z in _zone_names
}

# 80/20 summary values
easy_true_pct  = _zone_pct["Easy"]
junk_miles_pct = _zone_pct["Moderate"]
quality_pct    = _zone_pct["Marathon"] + _zone_pct["Hard"]

goal_pace_str  = fmt_pace(config.GOAL_MP_SEC)

dist_left, dist_right = st.columns(2)

with dist_left:
    # Donut chart — block-wide miles by zone
    _pie_labels = [_zone_labels[z] for z in _zone_names if _zone_mi[z] > 0]
    _pie_values = [_zone_mi[z]     for z in _zone_names if _zone_mi[z] > 0]
    _pie_colors = [_zone_colors[z] for z in _zone_names if _zone_mi[z] > 0]
    _pie_pcts   = [_zone_pct[z]    for z in _zone_names if _zone_mi[z] > 0]

    fig_donut = go.Figure(go.Pie(
        labels=_pie_labels,
        values=_pie_values,
        hole=0.52,
        marker=dict(colors=_pie_colors),
        textinfo="percent",
        textfont=dict(size=13),
        hovertemplate="<b>%{label}</b><br>%{percent} · %{value:.0f} mi<extra></extra>",
        sort=False,
    ))
    fig_donut.update_layout(
        template="plotly_white",
        title=dict(text="Miles by Effort Zone (This Block)", x=0.5,
                   font=dict(size=13)),
        annotations=[dict(text="Intensity<br>Mix", x=0.5, y=0.5,
                          font_size=13, showarrow=False)],
        legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5),
        margin=dict(t=50, b=40, l=20, r=20),
        height=320,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with dist_right:
    st.write("**80/20 Analysis**")
    st.write("")

    # Row 1 — Easy
    if easy_true_pct >= 0.70:
        st.success(f"✅ **{easy_true_pct:.0%} easy miles** — ideal aerobic base building")
    elif easy_true_pct >= 0.55:
        st.info(f"→ **{easy_true_pct:.0%} easy miles** — slightly low, push easy runs easier")
    else:
        st.warning(f"⚠️ **{easy_true_pct:.0%} easy miles** — running too hard on easy days")

    # Row 2 — Moderate / junk
    if junk_miles_pct > 0.35:
        st.warning(
            f"⚠️ **{junk_miles_pct:.0%} moderate-effort miles** — the danger zone. "
            f"Too hard to fully recover from, too easy to drive race adaptation. "
            f"Shift these toward genuinely easy or genuinely hard."
        )
    elif junk_miles_pct > 0.20:
        st.info(
            f"→ **{junk_miles_pct:.0%} moderate miles** — acceptable but watch this. "
            f"Easy days should feel embarrassingly slow."
        )
    else:
        st.success(f"✅ **{junk_miles_pct:.0%} moderate miles** — well-controlled effort distribution")

    # Row 3 — Quality
    if quality_pct < 0.10:
        st.info(
            f"→ **{quality_pct:.0%} quality miles** — consider adding one MP or "
            f"tempo session per week as you move into Build phase."
        )
    elif quality_pct <= 0.25:
        st.success(f"✅ **{quality_pct:.0%} quality miles** — healthy balance of hard work")
    else:
        st.warning(
            f"⚠️ **{quality_pct:.0%} quality miles** — too much intensity. "
            f"Back off hard efforts and protect your easy days."
        )

    st.write("")

    # 80/20 summary callout
    if easy_true_pct >= 0.70 and quality_pct <= 0.25:
        st.success(
            "✅ Your effort distribution follows the 80/20 principle. "
            "This is the training pattern of elite endurance athletes."
        )
    elif junk_miles_pct > 0.35:
        st.warning(
            "⚠️ More than a third of your miles are in the moderate zone. "
            "The 80/20 principle says: easy days must be easy. "
            "Slowing your easy runs often improves race times."
        )
    else:
        st.info(
            "→ Effort distribution is reasonable. "
            "The goal is ≥70% easy, ≤25% quality, minimal moderate."
        )

    st.caption(
        f"Zones defined relative to your goal marathon pace "
        f"({goal_pace_str}/mi ± thresholds from config)"
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 6c — Next Week Prescription
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📅 Next Week's Training Prescription")

# ── Prescription inputs ──────────────────────────────────────────────────────

# 4-week avg excluding rest weeks (miles == 0 weeks are uninformative for volume targets)
_nonrest = wk[wk["miles"] > 0].tail(4)
recent_4wk_avg   = float(_nonrest["miles"].mean()) if not _nonrest.empty else 0.0
current_long_run = safe_float(cur.get("long_run_miles"), 0.0)
ramp_is_safe     = safe_float(cur.get("ramp_score"), 100.0) >= 50
consistency_ok   = safe_float(cur.get("consistency_score"), 0.0) >= 70

# 4-week AES slope — negative = improving (pace/HR going down)
_aes_recent = wk.dropna(subset=["aes_mean"]).tail(4)
if len(_aes_recent) >= 2:
    _aes_slope   = float(np.polyfit(range(len(_aes_recent)),
                                    _aes_recent["aes_mean"].values, 1)[0])
    aes_is_improving = _aes_slope < 0
else:
    aes_is_improving = False

# HR availability: majority of runs have recorded HR
hr_available_flag = runs_raw["avg_hr"].notna().mean() > 0.5

# Next Monday date
_today       = pd.Timestamp.today().normalize()
_days_ahead  = (7 - _today.weekday()) % 7
_days_ahead  = 7 if _days_ahead == 0 else _days_ahead
next_monday  = (_today + pd.Timedelta(days=_days_ahead)).strftime("%b %d")

# ── Target mileage ────────────────────────────────────────────────────────────
# Priority order: safety check (ramp) → phase → default build
if not ramp_is_safe:
    rx_target = round_half(recent_4wk_avg * 0.85)
    rx_label  = "Recovery week — ramp rate too high"
elif current_phase == "Taper":
    rx_target = round_half(recent_4wk_avg * cfg.TAPER_FACTOR)
    rx_label  = "Taper week — cut volume intentionally"
elif current_phase == "Peak":
    rx_target = round_half(recent_4wk_avg)
    rx_label  = "Hold volume — focus shifts to quality"
elif current_phase == "Rest":
    rx_target = round_half(recent_4wk_avg * 0.5)
    rx_label  = "Light week — refresh before resuming build"
else:  # Base or Build
    rx_target = round_half(min(recent_4wk_avg * 1.10, cfg.TARGET_WEEKLY_MILES))
    rx_label  = "Build week — add ~10% volume"

# ── Long run target ───────────────────────────────────────────────────────────
if current_phase == "Taper":
    lr_target = round_half(current_long_run * 0.80)
    lr_note   = "Cut long run — taper protocol"
elif current_long_run >= cfg.PEAK_LONG_RUN_MI:
    lr_target = round_half(current_long_run)
    lr_note   = "Maintain peak long run"
elif not ramp_is_safe:
    lr_target = round_half(current_long_run)
    lr_note   = "Hold long run — recovery week"
else:
    lr_target = round_half(min(current_long_run + 1.5, cfg.PEAK_LONG_RUN_MI))
    lr_note   = f"Add 1–2 mi — progressing toward {cfg.PEAK_LONG_RUN_MI:.0f} mi target"

# ── Quality session ───────────────────────────────────────────────────────────
tempo_pace_str = fmt_pace(config.GOAL_MP_SEC - 30)

if not ramp_is_safe:
    quality = "None — recovery week. No quality until ramp score recovers."
elif current_phase == "Base":
    quality = "None — base phase. All runs easy."
elif current_phase == "Build":
    quality = f"1 × MP effort: 4–6 mi at {goal_pace_str}/mi marathon pace"
elif current_phase == "Peak":
    quality = (
        f"1 × tempo: 4 mi at {tempo_pace_str}/mi "
        f"+ 1 × MP long run segment"
    )
elif current_phase == "Taper":
    quality = f"1 × short MP effort: 2–3 mi at {goal_pace_str}/mi to stay sharp"
else:
    quality = "None — rest week."

# ── Easy run guidance ─────────────────────────────────────────────────────────
hr_cap = config.ZONE2_HR_CAP
if hr_available_flag:
    easy_note = f"Keep HR under {hr_cap} bpm on all easy days"
else:
    easy_note = "Run at conversational pace — able to speak full sentences"

# ── Recovery signal ───────────────────────────────────────────────────────────
ramp_score_cur = safe_float(cur.get("ramp_score"), 100.0)
if ramp_score_cur < 50:
    recovery_fn   = st.warning
    recovery_text = (
        "⚠️ Ramp score is elevated. If legs feel heavy Monday, "
        "convert the quality session to an easy run."
    )
elif not aes_is_improving and not consistency_ok:
    recovery_fn   = st.info
    recovery_text = (
        "→ AES and consistency both need attention. "
        "Prioritize sleep and nutrition this week."
    )
else:
    recovery_fn   = st.success
    recovery_text = "✅ No recovery flags. Execute the plan."

# ── Render ─────────────────────────────────────────────────────────────────────
with st.container():
    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown(f"**Week of {next_monday}**")
    with hdr_r:
        _emoji = PHASE_EMOJIS.get(current_phase, "🏃")
        st.markdown(f"**{_emoji} {current_phase}**")

    rx_c1, rx_c2, rx_c3, rx_c4 = st.columns(4)

    with rx_c1:
        st.metric("🗓 Total Miles", f"{rx_target:.0f}–{rx_target + 2:.0f} mi")
        st.caption(rx_label)

    with rx_c2:
        st.metric("🏃 Long Run", f"{lr_target:.0f} mi")
        st.caption(lr_note)

    with rx_c3:
        st.markdown("**⚡ Quality Session**")
        st.markdown(quality)

    with rx_c4:
        st.markdown("**💚 Easy Day Target**")
        st.markdown(easy_note)
        st.caption("All other days")

    st.write("")
    recovery_fn(recovery_text)
    st.caption(
        "Prescription generated from your training data and phase logic. "
        "Adjust based on how your body feels — data informs, it doesn't override."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 7 — Coaching Insights
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🧠 This Week's Coaching Insights")

insights: list[tuple] = []   # (st.error / st.warning / st.info / st.success, text)

# ── Ramp spike ──
ramp_rate_val  = cur.get("ramp_rate")
ramp_score_val = safe_float(cur.get("ramp_score"), 100.0)

if pd.notna(ramp_rate_val):
    rr = float(ramp_rate_val)
    if ramp_score_val < 20:
        insights.append((st.error,
            f"🚨 Training load spiked {rr:.0%} this week — "
            f"well above your safe threshold of {cfg.RAMP_THRESHOLD:.0%}. "
            f"Back off tomorrow or make it genuinely easy. "
            f"Injury risk is elevated right now."
        ))
    elif ramp_score_val < 50:
        insights.append((st.warning,
            f"⚠️ Volume climbed {rr:.0%} this week. "
            f"Monitor how your legs feel on your next run. "
            f"Any tightness = treat it as a rest day signal."
        ))

# ── AES trend ──
aes_wk = wk.dropna(subset=["aes_mean"]).tail(4)
if len(aes_wk) >= 2:
    aes_slope = float(np.polyfit(range(len(aes_wk)), aes_wk["aes_mean"].values, 1)[0])
    first_aes = float(aes_wk["aes_mean"].iloc[0])
    aes_pct   = abs(aes_slope) / first_aes * 100 if first_aes != 0 else 0.0
    # AES slope < 0 = AES going down = efficiency improving
    if aes_slope < 0 and aes_pct >= 1.0:
        insights.append((st.success,
            f"✅ Aerobic efficiency improved {aes_pct:.1f}% over the last 4 weeks. "
            f"Your engine is running cleaner — the easy mileage is converting."
        ))
    elif aes_slope > 0 and aes_pct >= 1.0:
        insights.append((st.warning,
            f"⚠️ Aerobic efficiency dropped {aes_pct:.1f}% recently. "
            f"This usually signals accumulated fatigue. "
            f"Prioritize sleep and keep effort genuinely easy this week."
        ))
    else:
        insights.append((st.info,
            "→ Aerobic efficiency is holding steady. "
            "Consistent easy mileage will move this needle over 4–6 weeks."
        ))

# ── Long run status ──
long_run = safe_float(cur.get("long_run_miles"), 0.0)
if long_run >= cfg.PEAK_LONG_RUN_MI:
    insights.append((st.success,
        f"✅ Long run target hit ({long_run:.1f} mi). "
        f"This is your most race-specific stimulus. "
        f"Protect it with a real recovery day after."
    ))
elif long_run >= cfg.PEAK_LONG_RUN_MI * 0.8:
    gap = cfg.PEAK_LONG_RUN_MI - long_run
    insights.append((st.info,
        f"→ Long run at {long_run:.1f} mi — "
        f"{gap:.1f} mi from your peak target. "
        f"One more strong weekend effort could get you there."
    ))
else:
    insights.append((st.info,
        f"📈 Long run is at {long_run:.1f} mi and still building. "
        f"Add 1–2 mi per week until you reach {cfg.PEAK_LONG_RUN_MI} mi."
    ))

# ── Consistency ──
cons = safe_float(cur.get("consistency_score"), 0.0)
if cons >= 80:
    insights.append((st.success,
        f"✅ {cons:.0f}% consistency over 8 weeks. "
        f"Elite-level adherence. The compounding effect of this "
        f"will show up on race day."
    ))
elif cons >= 60:
    insights.append((st.info,
        f"→ {cons:.0f}% consistency is solid. "
        f"Missing one session in five is normal — "
        f"just avoid clustering missed days."
    ))
else:
    insights.append((st.warning,
        f"⚠️ Consistency has dipped to {cons:.0f}% over 8 weeks. "
        f"Even one additional run per week compounds significantly "
        f"into your readiness score."
    ))

# ── Projected time vs goal ──
if proj["predicted_minutes"] is not None:
    pred_min = proj["predicted_minutes"]
    if pred_min < goal_minutes - 5:
        insights.append((st.success,
            f"✅ Current fitness projects to {proj['predicted_str']} — "
            f"{goal_minutes - pred_min:.0f} min ahead of your {goal_time_str} goal. "
            f"Stay healthy and execute the taper."
        ))
    elif pred_min > goal_minutes + 5:
        insights.append((st.warning,
            f"⚠️ Current projection is {proj['predicted_str']} — "
            f"{pred_min - goal_minutes:.0f} min behind your {goal_time_str} goal. "
            f"Focus on consistent weekly volume and at least one MP effort per week."
        ))
    else:
        insights.append((st.info,
            f"→ Projecting {proj['predicted_str']} — right on target for {goal_time_str}. "
            f"Execution from here is what separates the goal from the result."
        ))

# ── Phase-specific (always one) ──
_phase_insights = {
    "Base":  (st.info,  "🏃 Base phase: resist the urge to run fast. The aerobic stimulus at easy effort is the entire point right now. Save the watch."),
    "Build": (st.info,  "🔨 Build phase: start threading one quality session per week — a tempo or MP effort. Long runs should still feel controlled, not heroic."),
    "Peak":  (st.info,  "🔥 Peak training. This is the hardest part and the most productive. Recovery between hard efforts matters as much as the efforts themselves."),
    "Taper": (st.info,  "🌀 Taper phase: the fitness is banked. Cutting volume now is not laziness — it is the final piece of race preparation. Protect your legs."),
    "Rest":  (st.info,  "😴 Rest week. Nothing to analyze — rest is training."),
}
if current_phase in _phase_insights:
    insights.append(_phase_insights[current_phase])

# Render top 5 only
for fn, text in insights[:5]:
    fn(text)
