"""
BA3.2 AI Solution
Time-Series Sample: fraud by hour of day

File:
    AI_FD_TIMESERIES_SAMPLE.py

Purpose:
    A worked sample of time-series analysis on the project data.

    Fraud_Detection_Dataset.csv has NO calendar dates, only the hour of day
    (0-23) of each transaction, so a forecast over dates (ARIMA, see
    forecast_time_series() in AI_FD_ADVANCED.py) is not possible on this
    data. What the data does support is a daily-cycle analysis: how many
    transactions happen in each hour, and whether the fraud rate changes
    through the day.

    Two questions are answered:
        1. Does the fraud rate differ between hours of the day?
           (chi-square test across the 24 hours)
        2. Does any single hour stand out more than random variation would
           explain? (95% band around the overall fraud rate)

How to run (from the project folder):
    python AI_FD_TIMESERIES_SAMPLE.py            prints the table and tests
    python AI_FD_TIMESERIES_SAMPLE.py --plot     also saves hourly_fraud_rate.png
                                                 (needs: pip install matplotlib)

Needs AI_FD_DATA.py and AI_FD_ADVANCED.py in the same folder.
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

from AI_FD_DATA import (
    TARGET_COLUMN,
    TIME_COLUMN,
    load_dataset,
    preprocess_data
)
from AI_FD_ADVANCED import analyze_time_series


PLOT_FILE = "hourly_fraud_rate.png"


def hourly_fraud_table(data, time_column=TIME_COLUMN, target_column=TARGET_COLUMN):
    """
    Transactions, fraud cases and fraud rate (%) for each hour of the day.
    Rows with a missing hour are left out.
    """
    known = data.dropna(subset=[time_column])
    hours = known[time_column].astype(int)

    table = known.groupby(hours)[target_column].agg(transactions="size", fraud="sum")
    table["fraud_rate"] = table["fraud"] / table["transactions"] * 100
    table.index.name = "hour"

    return table


def test_hour_effect(data, time_column=TIME_COLUMN, target_column=TARGET_COLUMN):
    """
    Chi-square test of independence between hour of day and fraud.
    A small p-value (below 0.05) would mean the fraud rate really does
    depend on the hour. Returns (chi2, degrees of freedom, p-value).
    """
    known = data.dropna(subset=[time_column])
    counts = pd.crosstab(known[time_column].astype(int), known[target_column])
    chi2, p_value, degrees, _ = stats.chi2_contingency(counts)
    return chi2, degrees, p_value


def random_variation_band(table, confidence=1.96):
    """
    The range the hourly fraud rate would be expected to stay inside purely
    by chance, if every hour had the overall fraud rate. Returns
    (overall rate %, lower %, upper %), using the average hourly volume.
    """
    overall = table["fraud"].sum() / table["transactions"].sum()
    typical_volume = table["transactions"].mean()
    margin = confidence * np.sqrt(overall * (1 - overall) / typical_volume)

    return overall * 100, (overall - margin) * 100, (overall + margin) * 100


def plot_hourly_fraud_rate(table, band, file_path=PLOT_FILE):
    """Save a line chart of the hourly fraud rate with the random-variation band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall, lower, upper = band

    surface, ink, ink_soft, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    grid, band_fill, line = "#e1e0d9", "#e1e0d9", "#2a78d6"

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.axhspan(lower, upper, color=band_fill, alpha=0.8, linewidth=0)
    ax.axhline(overall, color=muted, linewidth=1, linestyle="--")
    ax.plot(table.index, table["fraud_rate"], color=line, linewidth=2,
            marker="o", markersize=5, markeredgecolor=surface, markeredgewidth=1.5)

    ax.set_ylim(0, 8)
    ax.set_xlim(-0.6, 23.6)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (0 = midnight)", color=ink_soft, fontsize=9)
    ax.set_ylabel("Fraud rate (% of transactions)", color=ink_soft, fontsize=9)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    ax.yaxis.grid(True, color=grid, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    # Labels sit outside the band so the data line never runs through them
    ax.text(-0.4, upper + 0.15, "Fraud rate by hour",
            ha="left", va="bottom", color=line, fontsize=9, fontweight="bold")
    ax.text(23.5, upper + 0.15, "Shaded: range expected from random variation alone",
            ha="right", va="bottom", color=ink_soft, fontsize=8)
    ax.text(-0.4, lower - 0.15, f"Dashed line: overall rate {overall:.2f}%",
            ha="left", va="top", color=ink_soft, fontsize=8)

    fig.tight_layout()
    fig.savefig(file_path, facecolor=surface)
    plt.close(fig)

    print(f"\nChart saved to: {file_path}")


def run_sample(make_plot=False):
    data = preprocess_data(load_dataset())

    table = hourly_fraud_table(data)
    chi2, degrees, p_value = test_hour_effect(data)
    overall, lower, upper = random_variation_band(table)

    print("\n========== FRAUD BY HOUR OF DAY ==========")
    print(f"Transactions with a known hour: {int(table['transactions'].sum()):,} "
          f"(rows with a missing hour left out)")
    print(table.round({"fraud_rate": 2}).to_string())

    # Reuse the project's time-series statistics on the 24-hour profile
    analyze_time_series(table["fraud_rate"])

    print("\n========== DOES THE HOUR MATTER? ==========")
    print(f"Overall fraud rate: {overall:.2f}%")
    print(f"Lowest hour: {table['fraud_rate'].idxmin()}:00 at {table['fraud_rate'].min():.2f}%")
    print(f"Highest hour: {table['fraud_rate'].idxmax()}:00 at {table['fraud_rate'].max():.2f}%")
    print(f"Chi-square test across 24 hours: chi2 = {chi2:.1f}, df = {degrees}, p = {p_value:.3f}")
    print(f"Range expected from random variation alone: {lower:.2f}% to {upper:.2f}%")

    outside = table[(table["fraud_rate"] < lower) | (table["fraud_rate"] > upper)]
    print(f"Hours outside that range: {list(outside.index) if len(outside) else 'none'}")

    if p_value >= 0.05:
        print("\nConclusion: the fraud rate does not depend on the hour of day in this "
              "dataset. The hourly differences are what chance alone would produce.")
    else:
        print("\nConclusion: the fraud rate differs between hours more than chance "
              "would explain. Look at the hours outside the range above.")

    if make_plot:
        plot_hourly_fraud_rate(table, (overall, lower, upper))

    return table


if __name__ == "__main__":
    run_sample(make_plot="--plot" in sys.argv)
