"""Matplotlib chart generators for Chainlit UI — all dark-themed."""

import io
import re

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Dark theme palette (GitHub-dark inspired) ────────────────────────────────
_BG    = "#0d1117"
_PANEL = "#161b22"
_TEXT  = "#c9d1d9"
_MUTED = "#8b949e"
_GREEN = "#2ea043"
_YELLOW = "#d29922"
_RED   = "#da3633"
_BLUE  = "#388bfd"


def _close(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data


def _score_color(score: float) -> str:
    return _GREEN if score >= 66 else _YELLOW if score >= 33 else _RED


def _hide_spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Chart 1: Score Donuts ────────────────────────────────────────────────────

def make_score_chart(health_score: float, growth_score: float) -> bytes:
    """Two donut gauges: health score and growth score."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2), facecolor=_BG)
    fig.suptitle("Financial Scores", color=_TEXT, fontsize=12,
                 fontweight="bold", y=1.04)

    for ax, score, label in zip(
        axes,
        [health_score, growth_score],
        ["Health Score", "Growth Score"],
    ):
        ax.set_facecolor(_BG)
        color = _score_color(score)

        # Clamp to avoid zero-size wedge
        filled = max(score, 0.5)
        empty  = max(100 - score, 0.5)

        ax.pie(
            [filled, empty],
            colors=[color, _PANEL],
            startangle=90,
            wedgeprops=dict(width=0.42, edgecolor=_BG, linewidth=3),
        )
        ax.text(0, 0.12, f"{score:.0f}%", ha="center", va="center",
                fontsize=17, fontweight="bold", color=color)
        ax.set_title(label, color=_MUTED, fontsize=9, pad=10)

    plt.tight_layout(pad=1.5)
    return _close(fig)


# ── Chart 2: RSI Gauge + Technical Signals ───────────────────────────────────

def make_technical_chart(technicals: dict) -> bytes:
    """RSI zone bar with needle + signal badge grid."""
    fig = plt.figure(figsize=(8, 3.8), facecolor=_BG)
    fig.suptitle("Technical Analysis", color=_TEXT, fontsize=12,
                 fontweight="bold", y=1.02)

    # ── Left panel: RSI gauge ──────────────────────────────────────────────
    ax_rsi = fig.add_axes([0.04, 0.18, 0.40, 0.68], facecolor=_BG)
    rsi = float(technicals.get("rsi_14") or 50)

    # Colored zones as stacked horizontal bars
    ax_rsi.barh([0], [30],        color=_RED,    alpha=0.55, height=0.55, left=0)
    ax_rsi.barh([0], [40],        color=_YELLOW, alpha=0.55, height=0.55, left=30)
    ax_rsi.barh([0], [30],        color=_GREEN,  alpha=0.55, height=0.55, left=70)

    # Needle
    ax_rsi.axvline(rsi, color="white", linewidth=2.5, zorder=5, ymin=0.05, ymax=0.95)

    # Zone labels
    for x, lbl, col in [(15, "Oversold", _RED), (50, "Neutral", _YELLOW),
                         (85, "Overbought", _GREEN)]:
        ax_rsi.text(x, -0.52, lbl, ha="center", color=col, fontsize=7.5)

    rsi_color = _RED if rsi < 30 else _GREEN if rsi > 70 else _YELLOW
    ax_rsi.text(rsi, 0.62, f"{rsi:.1f}", ha="center", va="bottom",
                color=rsi_color, fontsize=10, fontweight="bold")

    signal = technicals.get("rsi_signal", "")
    ax_rsi.set_xlabel(f"RSI-14  ·  Signal: {signal}", color=_MUTED, fontsize=8)
    ax_rsi.set_xlim(0, 100)
    ax_rsi.set_ylim(-0.8, 0.8)
    ax_rsi.set_yticks([])
    ax_rsi.tick_params(axis="x", colors=_MUTED, labelsize=7.5)
    _hide_spines(ax_rsi)
    ax_rsi.set_title("RSI-14 Gauge", color=_TEXT, fontsize=9, pad=6)

    # ── Right panel: signal badges ─────────────────────────────────────────
    ax_sig = fig.add_axes([0.50, 0.10, 0.47, 0.78], facecolor=_BG)

    signals = [
        ("MACD",       technicals.get("macd_interpretation", "N/A")),
        ("SMA Cross",  technicals.get("cross_status", "N/A")),
        ("Volume",     technicals.get("volume_note", "N/A")),
        ("RSI Signal", technicals.get("rsi_signal", "N/A")),
    ]

    for i, (name, value) in enumerate(signals):
        y = 0.82 - i * 0.22
        val_up = str(value).upper()
        is_bull = any(k in val_up for k in ("BULL", "GOLDEN", "ABOVE_AVERAGE", "NEUTRAL"))
        is_bear = any(k in val_up for k in ("BEAR", "DEATH", "BELOW"))
        badge_color = _GREEN if is_bull else _RED if is_bear else _YELLOW

        # Rounded rect badge
        rect = mpatches.FancyBboxPatch(
            (0.02, y - 0.09), 0.96, 0.18,
            boxstyle="round,pad=0.03",
            facecolor=badge_color + "28",
            edgecolor=badge_color,
            linewidth=1.2,
            transform=ax_sig.transAxes,
        )
        ax_sig.add_patch(rect)
        ax_sig.text(0.07, y, name,   ha="left",  va="center", color=_MUTED,
                    fontsize=8.5, transform=ax_sig.transAxes)
        ax_sig.text(0.93, y, str(value), ha="right", va="center",
                    color=badge_color, fontsize=8.5, fontweight="bold",
                    transform=ax_sig.transAxes)

    ax_sig.set_xlim(0, 1)
    ax_sig.set_ylim(0, 1)
    ax_sig.axis("off")
    ax_sig.set_title("Technical Signals", color=_TEXT, fontsize=9, pad=6)

    return _close(fig)


# ── Chart 3: Risk-Tiered Position Sizing ─────────────────────────────────────

def _parse_pct(text: str) -> float:
    """Extract first percentage number from LLM-generated sizing text."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text or "")
    return float(m.group(1)) if m else 0.0


def make_risk_chart(conservative: str, neutral: str, aggressive: str) -> bytes:
    """Horizontal bar chart for the three risk-tier position sizes."""
    tiers = [
        ("Conservative 🛡️", _parse_pct(conservative), _BLUE),
        ("Neutral ⚖️",       _parse_pct(neutral),      _YELLOW),
        ("Aggressive 🚀",   _parse_pct(aggressive),   _RED),
    ]

    labels = [t[0] for t in tiers]
    values = [t[1] for t in tiers]
    colors = [t[2] for t in tiers]

    fig, ax = plt.subplots(figsize=(6, 2.6), facecolor=_BG)
    ax.set_facecolor(_BG)

    y_pos = [2, 1, 0]
    bars = ax.barh(y_pos, values, color=colors, alpha=0.88, height=0.52,
                   edgecolor=_BG, linewidth=0)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%  of portfolio", va="center", ha="left",
                    color=_TEXT, fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=_TEXT, fontsize=9)
    ax.set_xlabel("Portfolio Allocation (%)", color=_MUTED, fontsize=8)
    ax.set_xlim(0, max(max(values) * 1.6, 12))
    ax.tick_params(axis="x", colors=_MUTED, labelsize=8)
    ax.tick_params(axis="y", colors=_TEXT)
    _hide_spines(ax)
    ax.set_title("Risk-Tiered Position Sizing", color=_TEXT,
                 fontsize=10, fontweight="bold", pad=10)

    plt.tight_layout()
    return _close(fig)


# ── Chart 4: Playbook History ─────────────────────────────────────────────────

def make_history_chart(entries: list) -> bytes:
    """
    Decision color bar per ticker (top) + health score bar (bottom).
    Takes a list of PlaybookEntry objects.
    """
    if not entries:
        return b""

    decision_colors = {
        "BUY": _GREEN, "HOLD": _YELLOW, "SELL": _RED, "REJECT": _MUTED,
    }

    # Most recent decision per ticker
    ticker_map: dict = {}
    for e in sorted(entries, key=lambda x: x.date):
        ticker_map[e.ticker] = e

    items = list(ticker_map.values())[-16:]  # cap at 16 tickers
    tickers   = [e.ticker        for e in items]
    decisions = [e.decision      for e in items]
    scores    = [e.health_score  for e in items]
    bar_cols  = [decision_colors.get(d, _MUTED) for d in decisions]
    score_cols = [_score_color(s) for s in scores]

    n = len(tickers)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(max(6, n * 0.85), 5),
        facecolor=_BG,
        gridspec_kw={"height_ratios": [1.2, 2], "hspace": 0.55},
    )
    fig.suptitle("Playbook — Research History", color=_TEXT,
                 fontsize=12, fontweight="bold", y=1.02)

    x = np.arange(n)

    # ── Top: decision banners ──────────────────────────────────────────────
    ax_top.set_facecolor(_BG)
    ax_top.bar(x, [1] * n, color=bar_cols, alpha=0.92, width=0.7,
               edgecolor=_BG, linewidth=0)
    for i, (t, d) in enumerate(zip(tickers, decisions)):
        ax_top.text(i, 0.5, d, ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(tickers, color=_TEXT, fontsize=8.5,
                            rotation=35, ha="right")
    ax_top.set_yticks([])
    _hide_spines(ax_top)
    ax_top.set_title("Latest Decision per Ticker", color=_MUTED,
                     fontsize=8, pad=4)

    # ── Bottom: health scores ──────────────────────────────────────────────
    ax_bot.set_facecolor(_BG)
    ax_bot.bar(x, scores, color=score_cols, alpha=0.85, width=0.65,
               edgecolor=_BG, linewidth=0)
    ax_bot.axhline(50, color=_MUTED, linestyle="--", linewidth=0.9, alpha=0.6,
                   label="50% gate")
    ax_bot.axhline(66, color=_GREEN, linestyle=":", linewidth=0.8, alpha=0.4,
                   label="66% GOOD")

    for i, s in enumerate(scores):
        ax_bot.text(i, s + 1.5, f"{s:.0f}", ha="center", va="bottom",
                    color=_score_color(s), fontsize=7.5, fontweight="bold")

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(tickers, color=_TEXT, fontsize=8.5,
                            rotation=35, ha="right")
    ax_bot.set_ylim(0, 115)
    ax_bot.set_ylabel("Health Score %", color=_MUTED, fontsize=8)
    ax_bot.tick_params(axis="x", colors=_MUTED)
    ax_bot.tick_params(axis="y", colors=_MUTED, labelsize=7.5)
    _hide_spines(ax_bot)
    ax_bot.set_title("Health Score by Ticker", color=_MUTED, fontsize=8, pad=4)
    ax_bot.legend(fontsize=7, labelcolor=_MUTED, facecolor=_PANEL,
                  edgecolor=_PANEL, loc="upper right")

    plt.tight_layout()
    return _close(fig)
