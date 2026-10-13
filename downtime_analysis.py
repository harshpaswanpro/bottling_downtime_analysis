"""
Bottling-line downtime analysis: availability, downtime Pareto, operator vs equipment split.

Input : 4 pipe-separated files in data/raw/ (downloaded from Kaggle, see README)
Output: data/processed/*.csv (for Power BI) and images/*.png

Run:  python downtime_analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import pandas as pd

RAW, OUT, IMG = Path("data/raw"), Path("data/processed"), Path("images")
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. load
# Files are pipe-separated, not comma-separated.
prod  = pd.read_csv(RAW / "line-productivity.csv", sep="|")
fact  = pd.read_csv(RAW / "downtime-factors.csv", sep="|")
prods = pd.read_csv(RAW / "products.csv", sep="|")

# line-downtime.csv has a second header row (factor IDs): drop it and rename columns.
FACTORS = list(range(1, 13))
down = pd.read_csv(RAW / "line-downtime.csv", sep="|").iloc[1:].copy()
down.columns = ["Batch"] + FACTORS
down["Batch"] = pd.to_numeric(down["Batch"], errors="coerce")
down = down.dropna(subset=["Batch"])
down["Batch"] = down["Batch"].astype(int)
down[FACTORS] = down[FACTORS].apply(pd.to_numeric, errors="coerce")

# ---------------------------------------------------------------- 2. batch duration
# Some End Time cells carry a stray "1900-01-01" prefix: keep only HH:MM:SS.
t_start = prod["Start Time"].astype(str).str.extract(r"(\d{1,2}:\d{2}:\d{2})")[0]
t_end   = prod["End Time"].astype(str).str.extract(r"(\d{1,2}:\d{2}:\d{2})")[0]
assert t_start.notna().all() and t_end.notna().all(), "unparsed time values"

start = pd.to_datetime(prod["Date"] + " " + t_start)
end   = pd.to_datetime(prod["Date"] + " " + t_end)
end   = end.where(end >= start, end + pd.Timedelta(days=1))  # batch crossed midnight
prod["BatchMinutes"] = (end - start).dt.total_seconds() / 60

# ---------------------------------------------------------------- 3. join
down["DowntimeMinutes"] = down[FACTORS].sum(axis=1)   # blank = no downtime logged
orphans = sorted(set(down["Batch"]) - set(prod["Batch"]))
print(f"Downtime rows with no production record (excluded): {orphans}")

df = (prod.merge(prods, on="Product", how="left")
          .merge(down[["Batch", "DowntimeMinutes"]], on="Batch", how="left"))
df["RunMinutes"]   = df["BatchMinutes"] - df["DowntimeMinutes"]
df["RunVsMin"]     = df["RunMinutes"] - df["Min batch time"]
df["Availability"] = df["Min batch time"] / df["BatchMinutes"]

# Finding: run time equals the minimum batch time in every batch, so Performance is
# always 100%. Quality (scrap/units) is not recorded. OEE therefore reduces to Availability.
print("Run time == min batch time in all batches:", bool((df["RunVsMin"].abs() < 1e-9).all()))

# downtime in long form (only batches that exist in production)
long = down[["Batch"] + FACTORS].melt(id_vars="Batch", var_name="Factor",
                                      value_name="DowntimeMinutes")
long = long.dropna(subset=["DowntimeMinutes"])
long = long[(long["DowntimeMinutes"] > 0) & long["Batch"].isin(df["Batch"])]
long = long.merge(fact, on="Factor", how="left")

# ---------------------------------------------------------------- 4. sanity checks
assert abs(long["DowntimeMinutes"].sum() - df["DowntimeMinutes"].sum()) < 1e-9
assert long["Description"].notna().all()
print(f"\nBatches: {len(df)} | Total downtime: {df['DowntimeMinutes'].sum():.0f} min")
print(f"Plant availability: {df['Min batch time'].sum() / df['BatchMinutes'].sum():.1%}")

# ---------------------------------------------------------------- 5. analysis
pareto = long.groupby("Description")["DowntimeMinutes"].sum().sort_values(ascending=False)
cum = (pareto.cumsum() / pareto.sum() * 100).round(1)
print("\nPareto:\n", pd.DataFrame({"minutes": pareto, "cum_%": cum}))
print(f"Causes needed to reach 80%: {(cum < 80).sum() + 1} of {len(pareto)}")

split = long.groupby("Operator Error")["DowntimeMinutes"].sum()
print("\nOperator-error split (%):\n", (split / split.sum() * 100).round(1))

daily = df.groupby("Date").agg(BatchMin=("BatchMinutes", "sum"),
                               MinTime=("Min batch time", "sum"),
                               Batches=("Batch", "count"))
daily["Availability"] = daily["MinTime"] / daily["BatchMin"]
print("\nAvailability by day:\n", daily.round(3))

by_prod = df.groupby("Product").agg(Batches=("Batch", "count"),
                                    BatchMin=("BatchMinutes", "sum"),
                                    MinTime=("Min batch time", "sum"))
by_prod["Availability"] = (by_prod["MinTime"] / by_prod["BatchMin"]).round(3)
print("\nBy product (small samples, esp. OR-600 = 1 batch):\n", by_prod)

# ---------------------------------------------------------------- 6. charts
x = range(len(pareto))
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(x, pareto.values, color="#0B7285")
ax1.set_xticks(list(x)); ax1.set_xticklabels(pareto.index, rotation=45, ha="right")
ax1.set_ylabel("Downtime (min)")
ax2 = ax1.twinx()
ax2.plot(list(x), cum.values, color="#F28C28", marker="o")
ax2.axhline(80, ls="--", color="grey"); ax2.set_ylim(0, 105); ax2.set_ylabel("Cumulative %")
plt.title(f"Downtime Pareto ({len(df)} batches, {pareto.sum():.0f} min)")
plt.tight_layout(); plt.savefig(IMG / "pareto.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5, 4))
vals = [split["No"], split["Yes"]]
ax.bar(["Equipment", "Operator"], vals, color=["#6B7A90", "#F28C28"])
for i, v in enumerate(vals):
    ax.text(i, v + 5, f"{v:.0f} min ({v / split.sum():.1%})", ha="center")
ax.set_ylabel("Downtime (min)"); plt.title("Downtime by attribution")
plt.tight_layout(); plt.savefig(IMG / "operator_split.png", dpi=150); plt.close()

daily["Availability"].plot(kind="bar", figsize=(8, 4), ylim=(0, 1), color="#0B7285",
                           title="Availability by day (last day = 1 batch)")
plt.tight_layout(); plt.savefig(IMG / "daily_availability.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 7. export for Power BI
df.to_csv(OUT / "batches_clean.csv", index=False)
long.to_csv(OUT / "downtime_long.csv", index=False)
print("\nSaved data/processed/*.csv and images/*.png")
