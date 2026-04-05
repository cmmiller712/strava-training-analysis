"""
generate_card.py — Strava Training Intelligence Card

Exports a 1080×1080px PNG suitable for LinkedIn / Instagram.
Run from the repo root:
    python generate_card.py
    python generate_card.py --name "Christian Miller" --goal "Sub-3 Marathon" \
                            --goal-minutes 180 --output outputs/training_card.png
"""
import argparse
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd

# ── src/ on path ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
import config
from phases import classify_training_phase
from readiness import compute_readiness, compute_projected_time

# ── Constants ─────────────────────────────────────────────────────────────────
C = SimpleNamespace(
    bg        = "#0f1117",
    panel     = "#1a1f2e",
    border    = "#2d3748",
    orange    = "#fc4c02",   # Strava brand
    green     = "#38a169",
    yellow    = "#d69e2e",
    warn      = "#dd6b20",
    red       = "#e53e3e",
    gray      = "#a0aec0",
    white     = "#ffffff",
    # phase palette (matches dashboard)
    phase     = {
        "Base":  "#4299e1",
        "Build": "#ed8936",
        "Peak":  "#e53e3e",
        "Taper": "#805ad5",
        "Rest":  "#a0aec0",
    },
    # Text-safe phase labels (no emoji — DejaVu Sans lacks color emoji glyphs)
    phase_symbol = {
        "Base":  "BASE",
        "Build": "BUILD",
        "Peak":  "PEAK",
        "Taper": "TAPER",
        "Rest":  "REST",
    },
)

RECOMPUTE_COLS = [
    "phase", "readiness_score", "readiness_label",
    "volume_score", "long_run_score", "consistency_score",
    "aes_trend_score", "ramp_score",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def readiness_color(score: float) -> str:
    if score <= 40:   return C.red
    elif score <= 65: return C.warn
    elif score <= 80: return C.yellow
    else:             return C.green


def fmt_pace(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 60}:{s % 60:02d}"


def fmt_hms(minutes: float) -> str:
    total = int(round(minutes * 60))
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def strip_leading_emoji(text: str) -> str:
    """Remove a leading emoji + space from insight strings."""
    return re.sub(r"^[\U00010000-\U0010ffff\u2600-\u27BF\u2B50\u2B55\u231A-\u231B"
                  r"\u25AA-\u25FE\u2614-\u2615\u26AA-\u26AB\u2728\u274C\u274E"
                  r"\u2753-\u2755\u2757\u27B0\u27BF✅⚠️🚨📈🏃🔨🔥🌀😴→]+\s*", "", text)


def rounded_rect(ax, x, y, w, h, radius, color, zorder=1, alpha=1.0):
    """Draw a rounded rectangle as a FancyBboxPatch."""
    fancy = mpatches.FancyBboxPatch(
        (x + radius, y + radius),
        w - 2 * radius, h - 2 * radius,
        boxstyle=f"round,pad={radius}",
        facecolor=color, edgecolor="none",
        zorder=zorder, alpha=alpha,
        transform=ax.transData,
    )
    ax.add_patch(fancy)


def draw_bar(ax, x, y, w, h, score, label_left, label_right):
    """Draw a single component bar with background, fill, and text labels."""
    # background track
    ax.barh(y, w, left=x, height=h, color=C.border, zorder=2)
    # fill
    fill_w = w * max(0.0, min(score / 100.0, 1.0))
    fill_c = readiness_color(score)
    if fill_w > 0:
        ax.barh(y, fill_w, left=x, height=h, color=fill_c, zorder=3)
    # labels
    ax.text(x - 0.01, y, label_left, ha="right", va="center",
            color=C.white, fontsize=9, zorder=4)
    ax.text(x + w + 0.01, y, label_right, ha="left", va="center",
            color=C.white, fontsize=9, fontweight="bold", zorder=4)


# ── Data loading + pipeline ───────────────────────────────────────────────────

def load_data(data_dir: Path):
    runs_path   = data_dir / "runs_enriched.csv"
    weekly_path = data_dir / "weekly_model.csv"
    if not runs_path.exists() or not weekly_path.exists():
        raise FileNotFoundError(
            f"Missing CSVs in {data_dir}. Run `python src/build_datasets.py` first."
        )
    runs   = pd.read_csv(runs_path,   parse_dates=["activity_date"])
    weekly = pd.read_csv(weekly_path, parse_dates=["week_start"])
    weekly = weekly.drop(columns=[c for c in RECOMPUTE_COLS if c in weekly.columns])
    weekly, current_phase, phase_history = classify_training_phase(weekly.copy(), config)
    weekly = compute_readiness(weekly, config)
    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    return runs, weekly, current_phase, phase_history


# ── Insight generation (mirrors app.py Row 7 logic) ──────────────────────────

def top_insight(weekly: pd.DataFrame, current_phase: str,
                proj: dict, goal_minutes: float) -> tuple[str, str]:
    """
    Return (insight_text, severity) for the highest-priority insight.
    severity: "error" | "warning" | "success" | "info"
    Priority: ramp_spike > projected_time > aes_trend > long_run > consistency > phase
    """
    cur = weekly.iloc[-1]

    def safe(col, default=0.0):
        v = cur.get(col)
        return default if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    ramp_rate  = cur.get("ramp_rate")
    ramp_score = safe("ramp_score", 100.0)

    # 1 — ramp spike
    if pd.notna(ramp_rate):
        rr = float(ramp_rate)
        if ramp_score < 20:
            return (
                f"Training load spiked {rr:.0%} this week — well above your safe "
                f"threshold of {config.RAMP_THRESHOLD:.0%}. Back off and reduce "
                f"intensity. Injury risk is elevated right now.",
                "error"
            )
        if ramp_score < 50:
            return (
                f"Volume climbed {rr:.0%} this week. Monitor how your legs feel "
                f"on your next run. Any tightness = treat it as a rest day signal.",
                "warning"
            )

    # 2 — projected time
    if proj.get("predicted_minutes") is not None:
        pred = proj["predicted_minutes"]
        if pred < goal_minutes - 5:
            return (
                f"Current fitness projects to {proj['predicted_str']} — "
                f"{goal_minutes - pred:.0f} min ahead of your {fmt_hms(goal_minutes)} goal. "
                f"Stay healthy and execute the taper.",
                "success"
            )
        if pred > goal_minutes + 5:
            return (
                f"Current projection is {proj['predicted_str']} — "
                f"{pred - goal_minutes:.0f} min behind your {fmt_hms(goal_minutes)} goal. "
                f"Focus on consistent weekly volume and at least one MP effort per week.",
                "warning"
            )

    # 3 — AES trend
    aes_wk = weekly.dropna(subset=["aes_mean"]).tail(4)
    if len(aes_wk) >= 2:
        sl  = float(np.polyfit(range(len(aes_wk)), aes_wk["aes_mean"].values, 1)[0])
        pct = abs(sl) / float(aes_wk["aes_mean"].iloc[0]) * 100 if aes_wk["aes_mean"].iloc[0] != 0 else 0
        if sl < 0 and pct >= 1.0:
            return (
                f"Aerobic efficiency improved {pct:.1f}% over the last 4 weeks. "
                f"Your engine is running cleaner — the easy mileage is converting.",
                "success"
            )
        if sl > 0 and pct >= 1.0:
            return (
                f"Aerobic efficiency dropped {pct:.1f}% recently. This usually signals "
                f"accumulated fatigue. Prioritize sleep and keep effort genuinely easy.",
                "warning"
            )

    # 4 — long run
    lr = safe("long_run_miles", 0.0)
    if lr >= config.PEAK_LONG_RUN_MI:
        return (
            f"Long run target hit ({lr:.1f} mi). This is your most race-specific "
            f"stimulus. Protect it with a real recovery day after.",
            "success"
        )
    if lr >= config.PEAK_LONG_RUN_MI * 0.8:
        gap = config.PEAK_LONG_RUN_MI - lr
        return (
            f"Long run at {lr:.1f} mi — {gap:.1f} mi from your peak target. "
            f"One more strong weekend effort could get you there.",
            "info"
        )
    return (
        f"Long run is at {lr:.1f} mi and building. Add 1–2 mi per week "
        f"until you reach {config.PEAK_LONG_RUN_MI} mi.",
        "info"
    )


# ── Card drawing ──────────────────────────────────────────────────────────────

def draw_card(
    runs: pd.DataFrame,
    weekly: pd.DataFrame,
    current_phase: str,
    phase_history: dict,
    proj: dict,
    athlete_name: str,
    goal_race: str,
    goal_minutes: float,
    output_path: Path,
):
    cur  = weekly.sort_values("week_start").iloc[-1]
    wk   = weekly.sort_values("week_start")

    score          = float(cur.get("readiness_score", 0) or 0)
    readiness_label = str(cur.get("readiness_label", "—"))
    rc             = readiness_color(score)

    # Component scores
    components = [
        ("Volume (25%)",          float(cur.get("volume_score",      0) or 0)),
        ("Long Run (25%)",        float(cur.get("long_run_score",    0) or 0)),
        ("Consistency (20%)",     float(cur.get("consistency_score", 0) or 0)),
        ("AES Trend (15%)",       float(cur.get("aes_trend_score",   0) or 0)),
        ("Ramp Safety (15%)",     float(cur.get("ramp_score",        0) or 0)),
    ]

    date_min = runs["activity_date"].min().strftime("%b %-d")
    date_max = runs["activity_date"].max().strftime("%b %-d, %Y")
    date_range_str = f"{date_min} – {date_max}"

    avg_4wk      = float(wk.tail(4)["miles"].mean())
    max_miles    = float(wk["miles"].max())
    max_long_run = float(wk["long_run_miles"].dropna().max()) if wk["long_run_miles"].notna().any() else 0.0
    mean_cons    = float(wk["consistency_score"].dropna().mean()) if "consistency_score" in wk.columns else 0.0

    insight_text, insight_sev = top_insight(weekly, current_phase, proj, goal_minutes)
    insight_color = {"error": C.red, "warning": C.warn,
                     "success": C.green, "info": "#4299e1"}[insight_sev]

    today_str = pd.Timestamp.today().strftime("%B %-d, %Y")

    phase_symbol = C.phase_symbol.get(current_phase, current_phase.upper())
    phase_color  = C.phase.get(current_phase, C.gray)

    goal_time_str   = fmt_hms(goal_minutes)
    pred_str        = proj.get("predicted_str", "—")
    lower_str       = proj.get("lower_str", "—")
    upper_str       = proj.get("upper_str", "—")
    pred_min        = proj.get("predicted_minutes")
    pred_goal_color = C.green if (pred_min is not None and pred_min < goal_minutes) else C.red

    phase_hist_str = "  ·  ".join(
        f"{p} {n}wk" for p, n in sorted(phase_history.items())
    )

    # ── Figure setup ──────────────────────────────────────────────────────────
    # 1080×1080 at 150dpi = 7.2×7.2 inches
    fig = plt.figure(figsize=(7.2, 7.2), dpi=150, facecolor=C.bg)
    fig.patch.set_facecolor(C.bg)

    # Outer axes — full card, used for background panels + text overlays
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(C.bg)
    ax.axis("off")

    PAD = 0.03          # horizontal padding (fraction of card width)
    W   = 1 - 2 * PAD  # usable content width

    # ── Zone boundaries (y fractions, 0=bottom, 1=top) ────────────────────────
    Z = SimpleNamespace(
        header_top    = 1.00,
        header_bot    = 0.90,   # 10% — header bar
        hero_bot      = 0.70,   # 20% — hero metrics
        spark_bot     = 0.50,   # 20% — sparkline
        breakdown_bot = 0.32,   # 18% — component bars
        insight_bot   = 0.10,   # 22% — insight (enlarged slightly for legibility)
        footer_bot    = 0.00,   #  8% — footer (covered by insight_bot → 0)
    )

    # ── Zone 1: Header ────────────────────────────────────────────────────────
    # Panel background
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, Z.header_bot), 1.0, Z.header_top - Z.header_bot,
        boxstyle="square,pad=0", facecolor=C.panel, edgecolor="none", zorder=1,
    ))
    # Orange rule at bottom of header
    ax.plot([0, 1], [Z.header_bot, Z.header_bot],
            color=C.orange, linewidth=1.5, zorder=5)

    # Left side
    ax.text(PAD, 0.963, "TRAINING  INTELLIGENCE",
            color=C.orange, fontsize=9.5, fontweight="bold", va="center",
            zorder=5, fontfamily="DejaVu Sans")
    # Fix 1: drop to 0.910 to open a clear gap below the orange "TRAINING INTELLIGENCE"
    # line (was 0.926 — too close, causing antialiasing color bleed). zorder=10 ensures
    # the name renders above all surrounding patches.
    ax.text(PAD, 0.910, athlete_name,
            color="#ffffff", fontsize=17, fontweight="bold", va="center", zorder=10)

    # Right side
    ax.text(1 - PAD, 0.963, goal_race,
            color=C.gray, fontsize=9, ha="right", va="center", zorder=5)
    ax.text(1 - PAD, 0.934, date_range_str,
            color=C.gray, fontsize=8, ha="right", va="center", zorder=5)
    # Strava-orange dot accent (replaces emoji that doesn't render in DejaVu Sans)
    ax.add_patch(mpatches.Circle(
        (1 - PAD - 0.055, 0.947), 0.018,
        facecolor=C.orange, zorder=4,
    ))

    # ── Zone 2: Hero metrics (3 equal panels) ─────────────────────────────────
    HERO_H  = Z.hero_bot - Z.header_bot    # panel height fraction
    HERO_Y  = Z.header_bot                 # bottom y of zone
    GAP     = 0.015
    PW      = (W - 2 * GAP) / 3            # panel width fraction

    panel_xs = [PAD, PAD + PW + GAP, PAD + 2 * (PW + GAP)]

    for px in panel_xs:
        ax.add_patch(mpatches.FancyBboxPatch(
            (px, HERO_Y + 0.01), PW, HERO_H - 0.02,
            boxstyle="round,pad=0.005",
            facecolor=C.panel, edgecolor=C.border, linewidth=0.5, zorder=2,
        ))

    # Panel 1 — Readiness
    px1 = panel_xs[0]
    MID_Y1 = HERO_Y + HERO_H / 2
    ax.text(px1 + PW / 2, HERO_Y + HERO_H - 0.025, "READINESS",
            color=C.gray, fontsize=7.5, fontweight="bold", ha="center", va="top", zorder=5)
    ax.text(px1 + PW / 2, MID_Y1 + 0.012, f"{score:.0f}",
            color=rc, fontsize=36, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(px1 + PW / 2, MID_Y1 - 0.03, readiness_label,
            color=rc, fontsize=8.5, ha="center", va="center", zorder=5)
    # Mini score bar
    bar_y    = HERO_Y + 0.022
    bar_x    = px1 + 0.015
    bar_w    = PW - 0.03
    bar_h    = 0.008
    ax.add_patch(mpatches.Rectangle(
        (bar_x, bar_y), bar_w, bar_h, facecolor=C.border, zorder=4))
    fill_w = bar_w * score / 100.0
    if fill_w > 0:
        ax.add_patch(mpatches.FancyBboxPatch(
            (bar_x, bar_y), fill_w, bar_h,
            boxstyle="round,pad=0.001", facecolor=rc, edgecolor="none", zorder=5))

    # Panel 2 — Projected Finish
    px2   = panel_xs[1]
    MID_Y2 = HERO_Y + HERO_H / 2
    ax.text(px2 + PW / 2, HERO_Y + HERO_H - 0.025, "PROJECTED FINISH",
            color=C.gray, fontsize=7.5, fontweight="bold", ha="center", va="top", zorder=5)
    ax.text(px2 + PW / 2, MID_Y2 + 0.018, pred_str,
            color=C.white, fontsize=24, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(px2 + PW / 2, MID_Y2 - 0.018, f"{lower_str} – {upper_str}",
            color=C.gray, fontsize=7.5, ha="center", va="center", zorder=5)
    ax.text(px2 + PW / 2, MID_Y2 - 0.042, f"Goal: {goal_time_str}",
            color=pred_goal_color, fontsize=8.5, ha="center", va="center", zorder=5)

    # Panel 3 — Current Phase
    px3   = panel_xs[2]
    MID_Y3 = HERO_Y + HERO_H / 2
    ax.text(px3 + PW / 2, HERO_Y + HERO_H - 0.025, "TRAINING PHASE",
            color=C.gray, fontsize=7.5, fontweight="bold", ha="center", va="top", zorder=5)
    # Phase symbol — large colored tag in place of emoji
    ax.text(px3 + PW / 2, MID_Y3 + 0.024, phase_symbol,
            color=phase_color, fontsize=16, fontweight="bold",
            ha="center", va="center", zorder=5)
    ax.text(px3 + PW / 2, MID_Y3 - 0.012, current_phase,
            color=C.white, fontsize=14, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(px3 + PW / 2, MID_Y3 - 0.038, phase_hist_str,
            color=C.gray, fontsize=7, ha="center", va="center", zorder=5)

    # ── Zone 3: Weekly Mileage Sparkline ──────────────────────────────────────
    SPARK_Y  = Z.spark_bot
    SPARK_H  = Z.hero_bot - Z.spark_bot

    ax.text(PAD, Z.hero_bot - 0.016, "WEEKLY MILEAGE",
            color=C.gray, fontsize=7.5, fontweight="bold", va="top", zorder=5)
    ax.text(1 - PAD, Z.hero_bot - 0.016, f"{avg_4wk:.0f} mi avg (last 4 wks)",
            color=C.gray, fontsize=7.5, ha="right", va="top", zorder=5)

    # Sparkline axes: inset axes [left, bottom, width, height] in figure fraction
    spk_left   = PAD
    spk_bottom = SPARK_Y + 0.045
    spk_w      = W
    spk_h      = SPARK_H - 0.075

    ax_spk = fig.add_axes([spk_left, spk_bottom, spk_w, spk_h],
                           facecolor="none")
    ax_spk.patch.set_facecolor("none")

    for spine in ax_spk.spines.values():
        spine.set_visible(False)
    ax_spk.tick_params(left=False, labelleft=False,
                       bottom=True, labelbottom=True,
                       colors=C.gray, labelsize=7)
    ax_spk.yaxis.set_visible(False)

    # Bars colored by phase
    x_pos = np.arange(len(wk))
    for i, (_, row) in enumerate(wk.iterrows()):
        ph    = row.get("phase", "Base")
        bc    = C.phase.get(ph, C.gray)
        ax_spk.bar(i, row["miles"], color=bc, width=0.75, zorder=3)

    # Target line
    ax_spk.axhline(config.TARGET_WEEKLY_MILES, color=C.white,
                   linewidth=0.8, linestyle="--", zorder=4)

    # Long run diamonds
    for i, (_, row) in enumerate(wk.iterrows()):
        if pd.notna(row.get("long_run_miles")) and row["long_run_miles"] > 0:
            ph  = row.get("phase", "Base")
            bc  = C.phase.get(ph, C.gray)
            ax_spk.scatter(i, row["long_run_miles"],
                           marker="D", s=18, color=bc,
                           edgecolors=C.white, linewidths=0.6, zorder=5)

    # X tick labels: abbreviated month initials when month changes
    labels = []
    last_m = None
    for _, row in wk.iterrows():
        m = row["week_start"].strftime("%b")
        labels.append(m if m != last_m else "")
        last_m = m
    ax_spk.set_xticks(x_pos)
    ax_spk.set_xticklabels(labels, color=C.gray, fontsize=7)
    ax_spk.set_xlim(-0.5, len(wk) - 0.5)
    ax_spk.tick_params(axis="x", length=0)

    # Summary stats below sparkline
    ax.text(PAD, SPARK_Y + 0.016,
            f"Peak week: {max_miles:.0f} mi  ·  Longest run: {max_long_run:.0f} mi  ·  Consistency: {mean_cons:.0f}%",
            color=C.gray, fontsize=7.5, va="bottom", zorder=5)

    # ── Zone 4: Readiness Breakdown ───────────────────────────────────────────
    BD_Y  = Z.breakdown_bot
    BD_H  = Z.spark_bot - Z.breakdown_bot

    ax.text(PAD, Z.spark_bot - 0.012, "READINESS BREAKDOWN",
            color=C.gray, fontsize=7.5, fontweight="bold", va="top", zorder=5)

    N_COMP    = len(components)
    bar_area_h = BD_H - 0.045
    row_h      = bar_area_h / N_COMP
    label_w    = 0.18    # left label fraction
    score_w    = 0.05    # right score label fraction
    bar_x_start = PAD + label_w
    bar_total_w = W - label_w - score_w

    for i, (comp_label, comp_score) in enumerate(components):
        # bars stack from top downward
        y_center = Z.spark_bot - 0.040 - (i + 0.5) * row_h
        bar_h_px = 0.012

        # background
        ax.add_patch(mpatches.Rectangle(
            (bar_x_start, y_center - bar_h_px / 2),
            bar_total_w, bar_h_px,
            facecolor=C.border, zorder=3,
        ))
        # fill
        fill_frac = max(0.0, min(comp_score / 100.0, 1.0))
        fill_c    = readiness_color(comp_score)
        if fill_frac > 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (bar_x_start, y_center - bar_h_px / 2),
                bar_total_w * fill_frac, bar_h_px,
                boxstyle="round,pad=0.001",
                facecolor=fill_c, edgecolor="none", zorder=4,
            ))
        # left label
        ax.text(bar_x_start - 0.008, y_center, comp_label,
                color=C.white, fontsize=7.5, ha="right", va="center", zorder=5)
        # right score
        ax.text(bar_x_start + bar_total_w + 0.008, y_center, f"{comp_score:.0f}",
                color=C.white, fontsize=8, fontweight="bold", ha="left", va="center", zorder=5)

    # ── Zone 5: Top Insight ───────────────────────────────────────────────────
    INS_Y = Z.insight_bot
    INS_H = Z.breakdown_bot - Z.insight_bot

    ax.text(PAD, Z.breakdown_bot - 0.012, "THIS WEEK'S KEY INSIGHT",
            color=C.gray, fontsize=7.5, fontweight="bold", va="top", zorder=5)

    # Accent bar
    accent_bar_h = INS_H - 0.05
    ax.add_patch(mpatches.Rectangle(
        (PAD, INS_Y + 0.012),
        0.005, accent_bar_h,
        facecolor=insight_color, zorder=4,
    ))

    # Wrapped insight text
    clean = strip_leading_emoji(insight_text).strip()
    wrapped = textwrap.fill(clean, width=68)
    ax.text(PAD + 0.018, INS_Y + 0.012 + accent_bar_h * 0.85,
            wrapped,
            color=C.white, fontsize=8.5, va="top", zorder=5,
            linespacing=1.5,
            wrap=False)  # already wrapped manually

    # ── Zone 6: Footer ────────────────────────────────────────────────────────
    ax.plot([PAD, 1 - PAD], [0.06, 0.06],
            color=C.border, linewidth=0.5, zorder=3)

    ax.text(PAD, 0.032, f"Generated {today_str}",
            color=C.gray, fontsize=7, va="center", zorder=5)
    ax.text(0.5, 0.032, "• strava-training-intelligence •",
            color=C.gray, fontsize=7, ha="center", va="center", zorder=5)

    github_user = "cmmiller712"
    ax.text(1 - PAD, 0.032,
            f"github.com/{github_user}/strava-training-analysis",
            color=C.gray, fontsize=7, ha="right", va="center", zorder=5)

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Card saved → {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate Strava Training Intelligence card PNG")
    p.add_argument("--name",          default="Christian Miller", help="Athlete name")
    p.add_argument("--goal",          default="Sub-3 Marathon",   help="Goal race name")
    p.add_argument("--goal-minutes",  default=180, type=float,    help="Goal time in minutes (default 180)")
    p.add_argument("--output",        default="outputs/training_card.png",
                   help="Output path (default: outputs/training_card.png)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve data directory: prefer CWD/data/processed (normal usage),
    # then fall back to the script's parent tree (covers git worktree layouts).
    cwd_data = Path.cwd() / "data" / "processed"
    script_data = Path(__file__).resolve().parent / "data" / "processed"
    if cwd_data.exists():
        data_dir = cwd_data
    elif script_data.exists():
        data_dir = script_data
    else:
        data_dir = cwd_data  # let load_data raise a clear error
    runs, weekly, current_phase, phase_history = load_data(data_dir)

    proj = compute_projected_time(runs, weekly_df=weekly, cfg=config)

    draw_card(
        runs           = runs,
        weekly         = weekly,
        current_phase  = current_phase,
        phase_history  = phase_history,
        proj           = proj,
        athlete_name   = args.name,
        goal_race      = args.goal,
        goal_minutes   = args.goal_minutes,
        output_path    = Path(args.output),
    )


if __name__ == "__main__":
    main()
