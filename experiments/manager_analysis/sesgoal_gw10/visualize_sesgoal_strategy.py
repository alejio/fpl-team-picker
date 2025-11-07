"""
Visualization script for Sesgoal (Alex Spanos) FPL manager strategy analysis
Creates charts showing performance trends, transfer patterns, and strategic evolution
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10

# Sesgoal's GW data (from analysis)
gameweeks = list(range(1, 11))
points = [47, 65, 55, 60, 49, 40, 69, 56, 48, 60]
overall_rank = [
    6638992,
    3126374,
    2168756,
    3001965,
    2926558,
    3856951,
    3497968,
    3790091,
    3740301,
    4275179,
]
team_value = [100.0, 100.1, 100.3, 100.7, 101.0, 101.1, 101.1, 101.3, 100.4, 101.0]
bank = [0.0, 0.0, 3.3, 1.3, 1.2, 1.0, 1.0, 4.2, 1.2, 1.0]

# Transfer data
transfers = [0, 1, 1, 1, 2, 1, 0, 1, 0, 2]
is_wildcard = [False] * 10
transfer_costs = [0, 0, 0, 0, 4, 0, 0, 0, 0, 0]

# Captain choices (extracted from analysis)
captains = [
    "Ekitiké",
    "Ekitiké",
    "João Pedro",
    "João Pedro",
    "J.Timber",
    "Semenyo",
    "Semenyo",
    "Semenyo",
    "Haaland",
    "Kudus",
]

# Key strategic moves
strategic_moves = {
    2: "Frimpong → Cucurella\n(Chelsea exposure)",
    3: "Palmer → Semenyo\n(£3.3m banked)",
    5: "Double transfer (-4)\nArsenal + Bournemouth",
    6: "Konsa → Guéhi\n(Crystal Palace)",
    8: "Wirtz → Longstaff\n(Leeds budget)",
    9: "Major squad overhaul",
    10: "Double transfer\nMunetsi + Sarr",
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
    if gw <= len(points):
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
    "Sesgoal (Alex Spanos) - Points Per Gameweek (Inconsistent Performance)",
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
    color="#ff6b6b",
    markerfacecolor="white",
    markeredgewidth=2,
)

# Highlight key turning points
ax2.scatter(
    [7],
    [overall_rank[6] / 1000000],
    s=300,
    color="lightgreen",
    marker="D",
    edgecolors="black",
    linewidths=2,
    zorder=5,
    label="Best GW (69 pts)",
)
ax2.scatter(
    [9],
    [overall_rank[8] / 1000000],
    s=200,
    color="orange",
    marker="s",
    edgecolors="black",
    linewidths=2,
    zorder=5,
    label="Squad Overhaul",
)

ax2.set_xlabel("Gameweek", fontsize=12, fontweight="bold")
ax2.set_ylabel("Overall Rank (millions)", fontsize=12, fontweight="bold")
ax2.set_title("Rank Progression: 6.6M → 4.3M", fontsize=13, fontweight="bold")
ax2.invert_yaxis()  # Lower rank is better
ax2.legend(loc="lower right", fontsize=9)
ax2.set_xticks(gameweeks)
ax2.grid(alpha=0.3)

# ===== Plot 3: Team Value Growth =====
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(
    gameweeks,
    team_value,
    marker="o",
    linewidth=2.5,
    markersize=8,
    color="#4c9aff",
    markerfacecolor="white",
    markeredgewidth=2,
)
ax3.fill_between(
    gameweeks, team_value, 100.0, alpha=0.2, color="green", label="Value Growth"
)
ax3.axhline(
    100.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Starting Value"
)

ax3.set_xlabel("Gameweek", fontsize=12, fontweight="bold")
ax3.set_ylabel("Team Value (£m)", fontsize=12, fontweight="bold")
ax3.set_title("Team Value: £100.0m → £101.0m (+£1.0m)", fontsize=13, fontweight="bold")
ax3.legend(loc="lower right", fontsize=9)
ax3.set_xticks(gameweeks)
ax3.grid(alpha=0.3)

# ===== Plot 4: Transfer Activity =====
ax4 = fig.add_subplot(gs[2, 0])
transfer_colors = ["#ff6b6b" if cost > 0 else "#51cf66" for cost in transfer_costs]
bars4 = ax4.bar(
    gameweeks,
    transfers,
    color=transfer_colors,
    alpha=0.7,
    edgecolor="black",
    linewidth=1.2,
)

# Add transfer cost annotations
for i, (gw, cost) in enumerate(zip(gameweeks, transfer_costs)):
    if cost > 0:
        ax4.text(
            gw,
            transfers[i] + 0.1,
            f"-{cost}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="red",
        )

ax4.set_xlabel("Gameweek", fontsize=12, fontweight="bold")
ax4.set_ylabel("Number of Transfers", fontsize=12, fontweight="bold")
ax4.set_title(
    "Transfer Activity (9 total, 4 points in hits)", fontsize=13, fontweight="bold"
)
ax4.set_xticks(gameweeks)
ax4.set_yticks(range(0, max(transfers) + 2))
ax4.grid(axis="y", alpha=0.3)

# ===== Plot 5: Bank Balance Management =====
ax5 = fig.add_subplot(gs[2, 1])
ax5.bar(gameweeks, bank, color="#ffd43b", alpha=0.7, edgecolor="black", linewidth=1.2)
ax5.axhline(0, color="black", linestyle="-", linewidth=1)

ax5.set_xlabel("Gameweek", fontsize=12, fontweight="bold")
ax5.set_ylabel("Bank Balance (£m)", fontsize=12, fontweight="bold")
ax5.set_title("Bank Balance Management", fontsize=13, fontweight="bold")
ax5.set_xticks(gameweeks)
ax5.grid(axis="y", alpha=0.3)

# Highlight highest bank
max_bank_gw = gameweeks[bank.index(max(bank))]
ax5.scatter(
    [max_bank_gw],
    [max(bank)],
    s=200,
    color="red",
    marker="*",
    edgecolors="black",
    linewidths=2,
    zorder=5,
)
ax5.annotate(
    f"Max: £{max(bank):.1f}m",
    xy=(max_bank_gw, max(bank)),
    xytext=(max_bank_gw, max(bank) + 0.5),
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
)

# ===== Plot 6: Captain Selection Distribution =====
ax6 = fig.add_subplot(gs[3, :])
captain_counts = {}
for cap in captains:
    captain_counts[cap] = captain_counts.get(cap, 0) + 1

captain_names = list(captain_counts.keys())
captain_freq = list(captain_counts.values())
colors_cap = plt.cm.Set3(np.linspace(0, 1, len(captain_names)))

bars6 = ax6.barh(
    captain_names,
    captain_freq,
    color=colors_cap,
    alpha=0.7,
    edgecolor="black",
    linewidth=1.2,
)

# Add frequency labels
for i, (name, freq) in enumerate(zip(captain_names, captain_freq)):
    ax6.text(freq + 0.1, i, f"{freq}x", va="center", fontsize=10, fontweight="bold")

ax6.set_xlabel("Times Captained", fontsize=12, fontweight="bold")
ax6.set_ylabel("Player", fontsize=12, fontweight="bold")
ax6.set_title(
    "Captain Selection Distribution (No Clear Template)", fontsize=13, fontweight="bold"
)
ax6.set_xlim(0, max(captain_freq) + 1)
ax6.grid(axis="x", alpha=0.3)

plt.suptitle(
    "Sesgoal (Alex Spanos) - FPL Strategy Analysis (GW1-10)",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(
    "experiments/manager_analysis/sesgoal_gw10/sesgoal_strategy_visualization.png",
    dpi=300,
    bbox_inches="tight",
)
print("✅ Saved: sesgoal_strategy_visualization.png")

# ===== Create Fixture Analysis Chart =====
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle("Sesgoal - Fixture Difficulty Analysis", fontsize=16, fontweight="bold")

# Count fixture difficulties from analysis (approximate based on emojis)
# 🟢 = Easy (1-2), 🟡 = Medium (3), 🔴 = Hard (4-5)
# This is approximate - would need full fixture data for exact counts

# Plot 1: Points vs Average Difficulty
avg_difficulty_estimate = [
    2.5,
    2.0,
    2.5,
    2.3,
    2.8,
    3.2,
    2.0,
    2.5,
    2.5,
    2.3,
]  # Estimated
ax_f1 = axes[0, 0]
scatter = ax_f1.scatter(
    avg_difficulty_estimate,
    points,
    s=100,
    c=gameweeks,
    cmap="viridis",
    edgecolors="black",
    linewidths=2,
    alpha=0.7,
)
for i, gw in enumerate(gameweeks):
    ax_f1.annotate(
        f"GW{gw}", (avg_difficulty_estimate[i], points[i]), fontsize=8, ha="center"
    )
ax_f1.set_xlabel(
    "Average Fixture Difficulty (Estimated)", fontsize=11, fontweight="bold"
)
ax_f1.set_ylabel("Points Scored", fontsize=11, fontweight="bold")
ax_f1.set_title("Points vs Fixture Difficulty", fontsize=12, fontweight="bold")
ax_f1.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax_f1, label="Gameweek")

# Plot 2: Transfer Timing vs Performance
ax_f2 = axes[0, 1]
transfer_gws = [gw for gw, t in zip(gameweeks, transfers) if t > 0]
transfer_points = [points[gw - 1] for gw in transfer_gws]
non_transfer_gws = [gw for gw, t in zip(gameweeks, transfers) if t == 0]
non_transfer_points = [points[gw - 1] for gw in non_transfer_gws]

ax_f2.scatter(
    transfer_gws,
    transfer_points,
    s=150,
    color="green",
    marker="o",
    label="Transfer GW",
    edgecolors="black",
    linewidths=2,
    alpha=0.7,
)
ax_f2.scatter(
    non_transfer_gws,
    non_transfer_points,
    s=150,
    color="blue",
    marker="s",
    label="No Transfer",
    edgecolors="black",
    linewidths=2,
    alpha=0.7,
)
ax_f2.set_xlabel("Gameweek", fontsize=11, fontweight="bold")
ax_f2.set_ylabel("Points Scored", fontsize=11, fontweight="bold")
ax_f2.set_title("Transfer Timing vs Performance", fontsize=12, fontweight="bold")
ax_f2.legend(fontsize=10)
ax_f2.set_xticks(gameweeks)
ax_f2.grid(alpha=0.3)

# Plot 3: Rank Change Per Gameweek
ax_f3 = axes[1, 0]
rank_changes = [0] + [
    overall_rank[i] - overall_rank[i - 1] for i in range(1, len(overall_rank))
]
colors_rank = ["green" if change < 0 else "red" for change in rank_changes]
bars_f3 = ax_f3.bar(
    gameweeks,
    rank_changes,
    color=colors_rank,
    alpha=0.7,
    edgecolor="black",
    linewidth=1.2,
)
ax_f3.axhline(0, color="black", linestyle="-", linewidth=1)
ax_f3.set_xlabel("Gameweek", fontsize=11, fontweight="bold")
ax_f3.set_ylabel("Rank Change", fontsize=11, fontweight="bold")
ax_f3.set_title(
    "Rank Change Per Gameweek (Negative = Improvement)", fontsize=12, fontweight="bold"
)
ax_f3.set_xticks(gameweeks)
ax_f3.grid(axis="y", alpha=0.3)

# Plot 4: Points Distribution
ax_f4 = axes[1, 1]
ax_f4.hist(points, bins=8, color="#4c9aff", alpha=0.7, edgecolor="black", linewidth=1.2)
ax_f4.axvline(
    np.mean(points),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {np.mean(points):.1f}",
)
ax_f4.axvline(
    np.median(points),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median: {np.median(points):.1f}",
)
ax_f4.set_xlabel("Points", fontsize=11, fontweight="bold")
ax_f4.set_ylabel("Frequency", fontsize=11, fontweight="bold")
ax_f4.set_title("Points Distribution", fontsize=12, fontweight="bold")
ax_f4.legend(fontsize=10)
ax_f4.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(
    "experiments/manager_analysis/sesgoal_gw10/sesgoal_fixture_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
print("✅ Saved: sesgoal_fixture_analysis.png")

plt.close("all")
print("\n✅ All visualizations generated successfully!")
