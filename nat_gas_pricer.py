"""
Natural Gas Price Estimator
===========================
Quantitative Research — Commodity Trading Desk

This script:
  1. Loads monthly end-of-month natural gas price data (Oct 2020 – Sep 2024).
  2. Fits a trend + Fourier seasonality model to capture annual price cycles.
  3. Uses a cubic spline for smooth interpolation within the historical range.
  4. Extrapolates up to one additional year beyond the last data point using
     the fitted model.
  5. Exposes a single public function:  estimate_price(date) -> float
  6. Produces a four-panel visualisation of the data and model.

Usage
-----
    from nat_gas_pricer import estimate_price
    price = estimate_price("2025-06-15")   # returns a float (USD)

Or run directly:
    python nat_gas_pricer.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load & pre-process data
# ──────────────────────────────────────────────────────────────────────────────
#  WHY: The raw CSV stores dates as strings ("10/31/20") and prices as floats.
#  We parse dates properly and sort chronologically so every downstream step
#  (spline fitting, indexing) is well-defined.

DATA_PATH = "C:\Users\faaru\OneDrive\Desktop\Forage\JP Task\Nat_Gas.xlsx"

df = pd.read_csv(DATA_PATH)
df["Dates"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
df = df.sort_values("Dates").reset_index(drop=True)

# Numeric time axis: days elapsed since the first observation.
# WHY: Scipy's curve_fit and CubicSpline require numeric inputs, not datetime
# objects.  Using "days since t0" keeps numbers small and the interpretation
# intuitive (slope ≈ $/day).
T0 = df["Dates"].iloc[0]                          # anchor: 2020-10-31
df["t"] = (df["Dates"] - T0).dt.days.astype(float)

t_data = df["t"].values
p_data = df["Prices"].values

T_LAST = t_data[-1]          # last observed time point (in days)
T_HORIZON = T_LAST + 365.25  # one-year extrapolation horizon


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Seasonal model (trend + Fourier harmonics)
# ──────────────────────────────────────────────────────────────────────────────
#  WHY: Natural gas prices exhibit a clear annual cycle (higher in winter,
#  lower in summer) driven by heating demand.  We capture this with Fourier
#  terms at the annual frequency (period ≈ 365.25 days) and its second
#  harmonic.  A linear trend (a + b·t) accounts for the steady upward drift
#  visible across the four-year window.
#
#  Model:
#    price(t) = a + b·t
#             + c1·sin(2π t / 365.25)  +  d1·cos(2π t / 365.25)   ← 1st harmonic
#             + c2·sin(4π t / 365.25)  +  d2·cos(4π t / 365.25)   ← 2nd harmonic
#
#  The 1st harmonic captures the dominant winter/summer swing.
#  The 2nd harmonic captures the asymmetry (e.g., the price can dip in spring
#  and again in mid-summer within a single year).

PERIOD = 365.25  # mean tropical year in days


def seasonal_model(t, a, b, c1, d1, c2, d2):
    """Linear trend + two Fourier harmonics."""
    return (
        a + b * t
        + c1 * np.sin(2 * np.pi * t / PERIOD)
        + d1 * np.cos(2 * np.pi * t / PERIOD)
        + c2 * np.sin(4 * np.pi * t / PERIOD)
        + d2 * np.cos(4 * np.pi * t / PERIOD)
    )


# Fit with least-squares (Levenberg-Marquardt under the hood in curve_fit).
# WHY: curve_fit minimises the sum of squared residuals between the model and
# the observed monthly prices, giving us optimal parameter values.
p0 = [10.0, 0.001, 0.5, -0.5, 0.1, -0.1]   # reasonable starting guesses
popt, _ = curve_fit(seasonal_model, t_data, p_data, p0=p0, maxfev=20_000)

# Quick diagnostics
fitted_values = seasonal_model(t_data, *popt)
residuals     = p_data - fitted_values
rmse          = np.sqrt(np.mean(residuals ** 2))
r_squared     = 1 - np.sum(residuals**2) / np.sum((p_data - p_data.mean())**2)

print("─" * 55)
print("  Seasonal model fit diagnostics")
print("─" * 55)
print(f"  Intercept (a):         {popt[0]:+.4f}  (base price, USD)")
print(f"  Trend slope (b):       {popt[1]:+.6f}  (USD per day ≈ "
      f"{popt[1]*365:.3f} USD/yr)")
print(f"  1st harmonic sin (c1): {popt[2]:+.4f}")
print(f"  1st harmonic cos (d1): {popt[3]:+.4f}")
print(f"  2nd harmonic sin (c2): {popt[4]:+.4f}")
print(f"  2nd harmonic cos (d2): {popt[5]:+.4f}")
print(f"\n  RMSE  : {rmse:.4f} USD")
print(f"  R²    : {r_squared:.4f}")
print("─" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Cubic spline (interpolation within historical range)
# ──────────────────────────────────────────────────────────────────────────────
#  WHY: The Fourier model is excellent for extrapolation but smooths over the
#  precise month-end values recorded in the data.  Within the historical window
#  we want to honour those exact observed prices.  A cubic spline passes
#  through every data point while keeping the curve smooth (continuous first
#  and second derivatives), making it ideal for intra-month interpolation.

cs = CubicSpline(t_data, p_data)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Combined price estimator
# ──────────────────────────────────────────────────────────────────────────────
#  WHY: We combine both models into a single function with clear rules:
#    • date ≤ last observed date   →  cubic spline  (interpolation)
#    • last observed < date ≤ +1yr →  seasonal model (extrapolation)
#    • beyond +1yr                 →  raise an error (too uncertain)

def estimate_price(date) -> float:
    """
    Estimate the natural gas price on a given date.

    Parameters
    ----------
    date : str | datetime-like
        Any date string or datetime pandas/python object.
        Examples: "2022-06-15", "2025-03-01", pd.Timestamp("2024-12-31").

    Returns
    -------
    float
        Estimated price in USD.

    Raises
    ------
    ValueError
        If the requested date is more than one year beyond the last
        observed data point (extrapolation too uncertain).
    """
    ts = pd.Timestamp(date)
    t  = (ts - T0).days  # convert to our numeric time axis

    if t > T_HORIZON:
        last_date = (T0 + pd.Timedelta(days=int(T_LAST))).date()
        raise ValueError(
            f"Date {ts.date()} is more than one year beyond the last "
            f"observation ({last_date}).  Extrapolation this far is unreliable."
        )

    if t <= T_LAST:
        # Historical range: use the cubic spline for maximum accuracy
        price = float(cs(t))
    else:
        # Future range: use the calibrated seasonal model
        price = float(seasonal_model(t, *popt))

    return round(price, 4)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Demo: sample price lookups
# ──────────────────────────────────────────────────────────────────────────────

demo_dates = [
    ("2021-06-15", "mid-history, summer"),
    ("2022-01-15", "mid-history, winter"),
    ("2023-09-15", "late-history, autumn"),
    ("2024-09-30", "last observed month"),
    ("2024-12-31", "1 month into future"),
    ("2025-03-15", "~5 months into future, spring"),
    ("2025-09-30", "~1 year into future"),
]

print("\n  Sample price estimates")
print("─" * 55)
print(f"  {'Date':<15}  {'Method':<12}  {'Price (USD)':>11}  {'Context'}")
print("─" * 55)
for d, ctx in demo_dates:
    ts = pd.Timestamp(d)
    t  = (ts - T0).days
    method = "spline" if t <= T_LAST else "seasonal"
    price  = estimate_price(d)
    print(f"  {d:<15}  {method:<12}  ${price:>10.4f}  {ctx}")
print("─" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Visualisation — four panels
# ──────────────────────────────────────────────────────────────────────────────
#  Panel A: Raw data + spline + seasonal model + 1-yr extrapolation
#  Panel B: Residuals of the seasonal model (model fit quality)
#  Panel C: Average price by calendar month (seasonal heatmap)
#  Panel D: Model decomposition — trend vs seasonal component

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Natural Gas Price Analysis & Forecasting", fontsize=16, fontweight="bold")

# Dense time grid for smooth curves
t_hist_grid   = np.linspace(t_data[0], T_LAST, 600)
t_future_grid = np.linspace(T_LAST, T_HORIZON, 200)
t_full_grid   = np.concatenate([t_hist_grid, t_future_grid])

dates_hist   = [T0 + pd.Timedelta(days=int(d)) for d in t_hist_grid]
dates_future = [T0 + pd.Timedelta(days=int(d)) for d in t_future_grid]
dates_full   = [T0 + pd.Timedelta(days=int(d)) for d in t_full_grid]

spline_curve  = cs(t_hist_grid)
model_hist    = seasonal_model(t_hist_grid,   *popt)
model_future  = seasonal_model(t_future_grid, *popt)

# ── Panel A: Full picture ──────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(dates_hist,   spline_curve, color="#1f77b4", lw=1.8,
        label="Cubic spline (interpolation)", zorder=2)
ax.plot(dates_future, model_future, color="#ff7f0e", lw=1.8,
        linestyle="--", label="Seasonal model (extrapolation)", zorder=2)
ax.scatter(df["Dates"], df["Prices"], color="black", s=28, zorder=3,
           label="Observed monthly prices")
ax.axvline(df["Dates"].iloc[-1], color="grey", lw=1, linestyle=":",
           label="Last observation (Sep 2024)")
ax.set_title("A — Historical Prices & 1-Year Forecast")
ax.set_ylabel("Price (USD)")
ax.legend(fontsize=7.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.grid(alpha=0.3)

# ── Panel B: Residuals ────────────────────────────────────────────────────
ax = axes[0, 1]
ax.bar(df["Dates"], residuals, color=np.where(residuals >= 0, "#2ca02c", "#d62728"),
       width=20)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(rmse,  color="blue", lw=1, linestyle="--", label=f"+RMSE ({rmse:.3f})")
ax.axhline(-rmse, color="blue", lw=1, linestyle="--", label=f"−RMSE ({rmse:.3f})")
ax.set_title("B — Seasonal Model Residuals (Actual − Fitted)")
ax.set_ylabel("Residual (USD)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.grid(alpha=0.3)

# ── Panel C: Monthly seasonal profile ────────────────────────────────────
ax = axes[1, 0]
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_avg = df.groupby(df["Dates"].dt.month)["Prices"].mean()
colors = ["#d62728" if v >= monthly_avg.mean() else "#1f77b4"
          for v in monthly_avg.values]
bars = ax.bar(range(1, 13), monthly_avg.values, color=colors, edgecolor="white")
ax.axhline(monthly_avg.mean(), color="black", lw=1, linestyle="--",
           label=f"Grand mean = ${monthly_avg.mean():.2f}")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names, fontsize=8)
ax.set_title("C — Average Price by Calendar Month (Seasonal Pattern)")
ax.set_ylabel("Avg Price (USD)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, monthly_avg.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7)

# ── Panel D: Trend vs seasonal decomposition ──────────────────────────────
ax = axes[1, 1]
trend_component    = popt[0] + popt[1] * t_full_grid
seasonal_component = (
    popt[2] * np.sin(2 * np.pi * t_full_grid / PERIOD)
    + popt[3] * np.cos(2 * np.pi * t_full_grid / PERIOD)
    + popt[4] * np.sin(4 * np.pi * t_full_grid / PERIOD)
    + popt[5] * np.cos(4 * np.pi * t_full_grid / PERIOD)
)
ax.plot(dates_full, trend_component, color="#9467bd", lw=1.8,
        label="Linear trend")
ax.plot(dates_full, seasonal_component, color="#8c564b", lw=1.8,
        linestyle="-.", label="Seasonal component")
ax.axvline(df["Dates"].iloc[-1], color="grey", lw=1, linestyle=":")
ax.set_title("D — Model Decomposition: Trend & Seasonal Component")
ax.set_ylabel("Component value (USD)")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/nat_gas_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nChart saved to /mnt/user-data/outputs/nat_gas_analysis.png")
