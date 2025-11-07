"""
Visualization script for Cinarteta FPL manager strategy analysis
Creates charts showing performance trends, transfer patterns, and strategic evolution
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# Add legend
from matplotlib.patches import Patch

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10

# Cinarteta's GW data (from analysis)
gameweeks = list(range(1, 11))
points = [63, 58, 52, 64, 55, 59, 82, 79, 55, 56]
overall_rank = [
    2149481,
    1551969,
    1452550,
    1873453,
    1149948,
    1112668,
    632818,
    359956,
    274653,
    633095,
]
team_value = [100.0, 100.1, 100.7, 101.0, 101.4, 101.8, 102.2, 103.2, 103.5, 104.0]
bank = [0.0, 0.0, 2.5, 2.5, 2.1, 0.1, 0.1, 1.0, 1.0, 1.0]

# Transfer data
transfers = [0, 0, 2, 1, 1, 0, 0, 2, 1, 0]
is_wildcard = [False] * 10

# Captain choices
captains = [
    "Haaland",
    "Haaland",
    "Haaland",
    "Salah",
    "Salah",
    "Haaland",
    "Haaland",
    "Haaland",
    "Semenyo",
    "Haaland",
]

# Key strategic moves
strategic_moves = {
    3: "Palmer + Diouf →\nTimber + Johnson\n(£2.5m banked)",
    4: "Dorgu → Senesi\n(Bournemouth)",
    5: "Johnson → Semenyo\n(Bournemouth double)",
    6: "MAJOR RESTRUCTURE\nSalah exit, Arsenal in",
    7: "BREAKTHROUGH\n82 pts!\n(Bench Boost!)",
    8: "Fernandes + Richarlison →\nWoltemade + Eze\n(Arsenal triple)",
    9: "João Pedro → Mateta\n(Semenyo captain!\nOutscored Haaland)",
}

# Create subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

# ===== Plot 1: Points Per Gameweek with Trend =====
ax1 = fig.add_subplot(gs[0, :])
colors = ["#ff6b6b" if p < 50 else "#51cf66" if p >= 70 else "#ffd43b" for p in points]
bars = ax1.bar(
    gameweeks, points, color=colors, alpha=0.7, edgecolor="black", linewidth=1.2
)

# Add average lines
first_half_avg = np.mean(points[:5])
second_half_avg = np.mean(points[5:])
ax1.axhline(
    first_half_avg,
    xmin=0,
    xmax=0.45,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"GW1-5 avg: {first_half_avg:.1f}",
)
ax1.axhline(
    second_half_avg,
    xmin=0.55,
    xmax=1,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"GW6+ avg: {second_half_avg:.1f}",
)

# Add vertical line separating halves
ax1.axvline(5.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

# Annotate key gameweeks
for gw, move in strategic_moves.items():
    ax1.annotate(
        move,
        xy=(gw, points[gw - 1]),
        xytext=(gw, points[gw - 1] + 15),
        ha="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )

# Add point values on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 2,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax1.set_xlabel("Gameweek", fontsize=12, fontweight="bold")
ax1.set_ylabel("Points", fontsize=12, fontweight="bold")
ax1.set_title(
    "Cinarteta - Points Per Gameweek (Steady Improvement → Peak Performance)",
    fontsize=14,
    fontweight="bold",
)
ax1.legend(loc="upper left", fontsize=10)
ax1.set_xticks(gameweeks)
ax1.grid(axis="y", alpha=0.3)

# ===== Plot 2: Overall Rank Progression =====
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(
    gameweeks,
    [r / 1000000 for r in overall_rank],
    marker="o",
    linewidth=2.5,
    markersize=8,
    color="#4c9aff",
    markerfacecolor="white",
    markeredgewidth=2,
)

# Highlight key turning points
ax2.scatter(
    [6],
    [overall_rank[5] / 1000000],
    s=300,
    color="gold",
    marker="*",
    edgecolors="black",
    linewidths=2,
    zorder=5,
    label="Major Restructure",
)
ax2.scatter(
    [7],
    [overall_rank[6] / 1000000],
    s=200,
    color="lightgreen",
    marker="D",
    edgecolors="black",
    linewidths=2,
    zorder=5,
    label="Breakthrough GW",
)

ax2.set_xlabel("Gameweek", fontsize=12, fontweight="bold")
ax2.set_ylabel("Overall Rank (millions)", fontsize=12, fontweight="bold")
ax2.set_title("Rank Progression: 2.1M → 633k", fontsize=13, fontweight="bold")
ax2.invert_yaxis()  # Lower rank is better
ax2.legend(loc="upper right", fontsize=9)
ax2.set_xticks(gameweeks)
ax2.grid(alpha=0.3)

# Add improvement annotation
ax2.annotate(
    "",
    xy=(10, overall_rank[9] / 1000000),
    xytext=(1, overall_rank[0] / 1000000),
    arrowprops=dict(arrowstyle="->", color="green", lw=3, alpha=0.6),
)
ax2.text(
    5.5,
    0.5,
    "70.5% rank improvement",
    fontsize=11,
    color="green",
    fontweight="bold",
    ha="center",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
)

# ===== Plot 3: Team Value Growth =====
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(
    gameweeks,
    team_value,
    marker="s",
    linewidth=2.5,
    markersize=7,
    color="#20c997",
    markerfacecolor="white",
    markeredgewidth=2,
)
ax3.fill_between(gameweeks, 100, team_value, alpha=0.2, color="#20c997")

ax3.set_xlabel("Gameweek", fontsize=12, fontweight="bold")
ax3.set_ylabel("Team Value (£m)", fontsize=12, fontweight="bold")
ax3.set_title("Team Value: £100.0m → £104.0m (+£4.0m)", fontsize=13, fontweight="bold")
ax3.set_xticks(gameweeks)
ax3.set_ylim([99.5, 104.5])
ax3.grid(alpha=0.3)

# Add growth annotation
ax3.annotate(
    f"+£{team_value[-1] - team_value[0]:.1f}m",
    xy=(10, team_value[-1]),
    xytext=(8, 101.5),
    fontsize=11,
    fontweight="bold",
    color="green",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    arrowprops=dict(arrowstyle="->", color="green", lw=2),
)

# ===== Plot 4: Transfer Activity =====
ax4 = fig.add_subplot(gs[2, 0])
transfer_colors = ["gold" if wc else "#748ffc" for wc in is_wildcard]
bars = ax4.bar(
    gameweeks,
    transfers,
    color=transfer_colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=1.2,
)

ax4.set_xlabel("Gameweek", fontsize=12, fontweight="bold")
ax4.set_ylabel("Number of Transfers", fontsize=12, fontweight="bold")
ax4.set_title(
    "Transfer Activity (0 hits taken - Perfect!)", fontsize=13, fontweight="bold"
)
ax4.set_xticks(gameweeks)
ax4.set_ylim([0, 2.5])
ax4.grid(axis="y", alpha=0.3)

legend_elements = [
    Patch(facecolor="#748ffc", edgecolor="black", label="Free Transfer"),
]
ax4.legend(handles=legend_elements, loc="upper left", fontsize=9)

# ===== Plot 5: Captain Choices =====
ax5 = fig.add_subplot(gs[2, 1])
captain_counts = pd.Series(captains).value_counts()
colors_cap = {"Haaland": "#3498db", "Salah": "#e74c3c", "Semenyo": "#9b59b6"}
wedges, texts, autotexts = ax5.pie(
    captain_counts.values,
    labels=captain_counts.index,
    autopct="%1.0f%%",
    startangle=90,
    colors=[colors_cap.get(c, "gray") for c in captain_counts.index],
    textprops={"fontsize": 11, "fontweight": "bold"},
)

ax5.set_title("Captain Selection Distribution", fontsize=13, fontweight="bold")

# Add detail text
detail_text = "Haaland: GW1-3, 6-8, 10 (template)\nSalah: GW4-5 (fixture-based)\nSemenyo: GW9 (differential!\nOutscored Haaland)"
ax5.text(
    1.3,
    0,
    detail_text,
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# ===== Plot 6: Strategic Evolution Timeline =====
ax6 = fig.add_subplot(gs[3, :])
ax6.set_xlim(0, 11)
ax6.set_ylim(0, 5)

# Define phases
phases = [
    {
        "name": "Slow Start",
        "start": 0.5,
        "end": 2.5,
        "y": 4,
        "color": "#ff6b6b",
        "desc": "Unbalanced squad\n58.4 pts/GW",
    },
    {
        "name": "Early Adjustments",
        "start": 2.5,
        "end": 5.5,
        "y": 4,
        "color": "#ffd43b",
        "desc": "Reactive transfers\n60.5 pts/GW",
    },
    {
        "name": "Major Restructure",
        "start": 5.5,
        "end": 6.5,
        "y": 4,
        "color": "#748ffc",
        "desc": "Squad overhaul\n59 pts",
    },
    {
        "name": "Peak Performance",
        "start": 6.5,
        "end": 8.5,
        "y": 4,
        "color": "#51cf66",
        "desc": "82 + 79 pts\n(Bench Boost GW7)\nRank 360k",
    },
    {
        "name": "Consolidation",
        "start": 8.5,
        "end": 10.5,
        "y": 4,
        "color": "#ff8787",
        "desc": "Fine-tuning\n55.5 pts/GW",
    },
]

for phase in phases:
    width = phase["end"] - phase["start"]
    rect = Rectangle(
        (phase["start"], phase["y"] - 0.4),
        width,
        0.8,
        facecolor=phase["color"],
        edgecolor="black",
        linewidth=2,
        alpha=0.7,
    )
    ax6.add_patch(rect)
    ax6.text(
        phase["start"] + width / 2,
        phase["y"],
        phase["name"],
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax6.text(
        phase["start"] + width / 2,
        phase["y"] - 1,
        phase["desc"],
        ha="center",
        va="top",
        fontsize=8,
    )

# Add strategy characteristics
strategies = [
    {"gw": 3, "text": "Value\nbuilding\nstarts", "y": 1.5},
    {"gw": 6, "text": "Bold\nrestructure", "y": 1.5},
    {"gw": 7, "text": "Peak\nperformance\n(Bench Boost)", "y": 1.5},
    {"gw": 9, "text": "Differential\ncaptain\n(Semenyo > Haaland)", "y": 1.5},
]

for strat in strategies:
    ax6.scatter(
        [strat["gw"]],
        [strat["y"]],
        s=200,
        color="yellow",
        marker="*",
        edgecolors="black",
        linewidths=2,
        zorder=5,
    )
    ax6.text(
        strat["gw"],
        strat["y"] - 0.7,
        strat["text"],
        ha="center",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
    )

ax6.set_xlabel("Gameweek", fontsize=12, fontweight="bold")
ax6.set_title(
    "Strategic Evolution: From Slow Start to Peak Performance",
    fontsize=14,
    fontweight="bold",
)
ax6.set_xticks(gameweeks)
ax6.set_yticks([])
ax6.spines["left"].set_visible(False)
ax6.spines["right"].set_visible(False)
ax6.spines["top"].set_visible(False)

# Overall title
fig.suptitle(
    'Cinarteta (ID: 293492) - FPL Strategy Deep Dive\n"Steady Improvement with Perfect Transfer Discipline" ✅',
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

plt.tight_layout()
plt.savefig(
    "experiments/manager_analysis/cinarteta_gw10/cinarteta_strategy_visualization.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    "✅ Visualization saved to: experiments/manager_analysis/cinarteta_gw10/cinarteta_strategy_visualization.png"
)
plt.close()

# ===== Create second figure: Fixture Difficulty Analysis =====
fig2, ((ax7, ax8), (ax9, ax10)) = plt.subplots(2, 2, figsize=(14, 10))

# Fixture difficulty by gameweek (estimated from emoji indicators in analysis)
# 🟢 = 1-2 (easy), 🟡 = 3 (average), 🔴 = 4-5 (hard)
# Average difficulty of starting XI based on emoji counts
starting_11_avg_difficulty = [2.2, 2.0, 2.4, 2.0, 2.2, 1.8, 1.9, 2.0, 1.8, 2.0]

ax7.plot(
    gameweeks,
    starting_11_avg_difficulty,
    marker="o",
    linewidth=2.5,
    markersize=8,
    color="#e67e22",
)
ax7.axhline(2, color="gray", linestyle="--", alpha=0.5, label="Average difficulty")
ax7.set_xlabel("Gameweek", fontsize=11, fontweight="bold")
ax7.set_ylabel("Avg Fixture Difficulty", fontsize=11, fontweight="bold")
ax7.set_title("Starting XI Fixture Difficulty Trend", fontsize=12, fontweight="bold")
ax7.set_ylim([1, 3])
ax7.set_xticks(gameweeks)
ax7.legend()
ax7.grid(alpha=0.3)

# Transfer timing vs fixture difficulty
transfer_gws = [
    gw for gw, t in zip(gameweeks, transfers) if t > 0 and not is_wildcard[gw - 1]
]
ax8.scatter(
    transfer_gws,
    [starting_11_avg_difficulty[gw - 1] for gw in transfer_gws],
    s=200,
    color="#e74c3c",
    alpha=0.7,
    edgecolors="black",
    linewidths=2,
)
ax8.axhline(2, color="gray", linestyle="--", alpha=0.5)
ax8.set_xlabel("Gameweek", fontsize=11, fontweight="bold")
ax8.set_ylabel("Fixture Difficulty After Transfer", fontsize=11, fontweight="bold")
ax8.set_title("Transfer Impact on Fixture Difficulty", fontsize=12, fontweight="bold")
ax8.set_ylim([1, 3])
ax8.set_xticks(gameweeks)
ax8.grid(alpha=0.3)
ax8.text(
    8,
    2.5,
    "Lower = Better fixtures\nafter transfers",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
)

# Points vs Fixture Difficulty Correlation
ax9.scatter(
    starting_11_avg_difficulty,
    points,
    s=100,
    alpha=0.7,
    c=gameweeks,
    cmap="viridis",
    edgecolors="black",
    linewidths=1.5,
)
for i, gw in enumerate(gameweeks):
    ax9.annotate(
        f"GW{gw}",
        (starting_11_avg_difficulty[i], points[i]),
        fontsize=8,
        xytext=(5, 5),
        textcoords="offset points",
    )
ax9.set_xlabel(
    "Avg Fixture Difficulty (Lower = Easier)", fontsize=11, fontweight="bold"
)
ax9.set_ylabel("Points Scored", fontsize=11, fontweight="bold")
ax9.set_title(
    "Points vs Fixture Difficulty Correlation", fontsize=12, fontweight="bold"
)
ax9.grid(alpha=0.3)
colorbar = plt.colorbar(ax9.collections[0], ax=ax9, label="Gameweek")

# Bank balance strategy
ax10.bar(gameweeks, bank, color="#20c997", alpha=0.7, edgecolor="black", linewidth=1.2)
ax10.set_xlabel("Gameweek", fontsize=11, fontweight="bold")
ax10.set_ylabel("Bank Balance (£m)", fontsize=11, fontweight="bold")
ax10.set_title(
    "Bank Balance - Strategic Fund Management", fontsize=12, fontweight="bold"
)
ax10.set_xticks(gameweeks)
ax10.grid(axis="y", alpha=0.3)

# Annotate key banking moments
ax10.annotate(
    "Palmer sale\n+£2.5m",
    xy=(3, 2.5),
    xytext=(3, 3.5),
    fontsize=9,
    ha="center",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
)
ax10.annotate(
    "Major\nrestructure",
    xy=(6, 0.1),
    xytext=(6, 1.5),
    fontsize=9,
    ha="center",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
)

fig2.suptitle(
    "Cinarteta - Fixture & Financial Strategy Analysis",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    "experiments/manager_analysis/cinarteta_gw10/cinarteta_fixture_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
print(
    "✅ Fixture analysis saved to: experiments/manager_analysis/cinarteta_gw10/cinarteta_fixture_analysis.png"
)
plt.close()

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print(
    "1. experiments/manager_analysis/cinarteta_gw10/cinarteta_strategy_visualization.png"
)
print("2. experiments/manager_analysis/cinarteta_gw10/cinarteta_fixture_analysis.png")
print("\nKey findings visualized:")
print("✅ 13.4% improvement from first half to second half")
print("✅ Rank improvement from 2.1M to 633k")
print("✅ £4.0m team value growth (exceptional)")
print("✅ 0 hits taken (perfect transfer discipline)")
print("✅ Peak performance in GW7-8 (82 + 79 pts)")
