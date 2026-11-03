# Checks your real dataset against what app.py expects, and shows what the monthly temperatures add.
# Run it inside the image so it uses the same package versions as the API:
#   docker run --rm nexvoy python verify_dataset.py
# Nothing here changes any file.
import json
import sys

import numpy as np
import pandas as pd

import app  # loads the model + dataset exactly as the API does (stops with a clear error if something is wrong)

raw = pd.read_csv(app.DATA_PATH)
print(f"CSV rows: {len(raw)}   usable destinations after cleaning: {len(app.DEST)}\n")

print("1. Columns the API needs")
needed = ["city", "country", "region", "budget_level", "avg_hotel_usd", "avg_temp_monthly",
          "ideal_durations", "short_description", "cost_of_living_index", "tourist_infra_score"] + app.PREF_COLS
for c in needed:
    print(f"   {'ok     ' if c in raw.columns else 'MISSING'} {c}")

print("\n2. Monthly temperature data")
bad = raw["avg_temp_monthly"].apply(lambda r: len(app.monthly_averages(r)) == 0).sum()
lengths = raw["avg_temp_monthly"].apply(lambda r: len(app.monthly_averages(r))).value_counts().to_dict()
print(f"   rows with no readable monthly data : {bad}")
print(f"   months per row (months: rows)       : {lengths}")
sample = raw["avg_temp_monthly"].dropna().iloc[0]
print(f"   example of the raw value            : {str(sample)[:160]}...")

print("\n3. Destinations per region and budget (a region needs at least "
      f"{app.MIN_REGION_RESULTS} in the chosen budget to be applied)")
print(pd.crosstab(app.DEST["region"], app.DEST["budget_level"]).to_string())

print("\n4. Does the model agree with the dataset's budget labels?")
pred = app.PIPELINE.predict(app.DEST[app.FEATURE_COLS])
print(f"   agreement on all cleaned cities: {(pred == app.DEST['budget_level']).mean():.1%}")
print("   (the model was trained on these same cities, so this is NOT a test score; "
      "the held-out result is in nexvoy_model_info.json)")

print("\n5. Why the monthly temperatures matter")
swing = app.MONTHLY.max(axis=1) - app.MONTHLY.min(axis=1)
d = app.DEST.assign(coldest=np.nanmin(app.MONTHLY, axis=1), warmest=np.nanmax(app.MONTHLY, axis=1), swing=swing)
print(f"   average gap between coldest and warmest month: {swing.mean():.1f} C   (largest: {swing.max():.1f} C)")
print("   Cities with the biggest seasons (the yearly average hides this):")
print(d.sort_values("swing", ascending=False)[["city", "country", "avg_temp", "coldest", "warmest"]].head(8)
      .round(1).to_string(index=False))
wish = 22.0
yearly = np.exp(-np.abs(app.DEST["avg_temp"].to_numpy(float) - wish) / app.TEMP_DECAY_C)
monthly = np.nanmean(np.exp(-np.abs(app.MONTHLY - wish) / app.TEMP_DECAY_C), axis=1)
d = d.assign(yearly_score=yearly * 100, monthly_score=monthly * 100)
d["change"] = d["monthly_score"] - d["yearly_score"]
print(f"\n   For a wish of {wish:.0f} C, the cities whose temperature score drops most when monthly data is used:")
print(d.sort_values("change")[["city", "country", "avg_temp", "coldest", "warmest", "yearly_score", "monthly_score"]]
      .head(6).round(1).to_string(index=False))
print("\nDone.")
sys.exit(0)
