# build_climate.py
# Builds city_climate_era5.csv: the average temperature of every month for every city in
# Worldwide_Travel_Cities_FIXED.csv, taken from the Open-Meteo Historical Weather API (ERA5 data).
#
# Run this on your own computer, NOT on Render. It needs:  pip install requests pandas numpy
#
# Put this file in the same folder as Worldwide_Travel_Cities_FIXED.csv, then:
#   python build_climate.py --check        one tiny request to confirm the API works (costs 1 call)
#   python build_climate.py --limit 10     fetch 10 cities, then print how they compare with your dataset
#   python build_climate.py                fetch as many cities as today's budget allows
#
# It is safe to stop it at any time and run it again. Finished cities are saved after every request,
# and cities already in city_climate_era5.csv are skipped.
#
# Why it is slow: the free Open-Meteo plan allows 10,000 calls per day and 5,000 per hour, and a request
# that covers more than 2 weeks of data counts as several calls (about days divided by 14). Five years of
# daily data is about 130 calls per city, so the script waits between requests and stops when the daily
# budget is used. Run it once a day until it says everything is finished.
# The free plan is for non-commercial use only.

import argparse
import json
import os
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "Worldwide_Travel_Cities_FIXED.csv")
OUTPUT = os.path.join(HERE, "city_climate_era5.csv")
USAGE_FILE = os.path.join(HERE, "climate_usage.json")

# ---- Settings you may change -------------------------------------------------------------------
START_YEAR = 2020            # first full year to average
END_YEAR = 2024              # last full year to average (must be at least 5 days in the past)
MODEL = "era5"               # ERA5 gives consistent data over the years (see the Open-Meteo docs)
DAILY_BUDGET = 8000          # calls to use per day. The free limit is 10,000, so this keeps a safety margin
CALLS_PER_HOUR = 4500        # pacing. The free limit is 5,000 per hour
# -------------------------------------------------------------------------------------------------

URL = "https://archive-api.open-meteo.com/v1/archive"
MONTH_COLS = [f"m{i}" for i in range(1, 13)]
YEARS_LABEL = f"{START_YEAR} to {END_YEAR}"


class StopNow(Exception):
    """Raised when the script must stop (daily limit reached, bad request, ...)."""


def make_key(lat, lon):
    # Same key the API (app.py) builds, so both sides always match.
    return f"{float(lat):.3f},{float(lon):.3f}"


def request_cost(start: dt.date, end: dt.date) -> float:
    # My reading of Open-Meteo's pricing page: one variable, 14 days or less = 1 call; longer = days / 14.
    days = (end - start).days + 1
    return max(1.0, days / 14.0)


# ---- Daily budget memory -----------------------------------------------------------------------
def load_usage() -> dict:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    try:
        with open(USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == today:
            return data
    except Exception:
        pass
    return {"date": today, "calls": 0.0}


def save_usage(usage: dict) -> None:
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(usage, f)


# ---- Talking to Open-Meteo ---------------------------------------------------------------------
def call_api(lat, lon, start: dt.date, end: dt.date) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": "auto",      # the API needs a timezone when daily values are requested
        "models": MODEL,
    }
    waits = 0
    failures = 0
    while True:
        try:
            resp = requests.get(URL, params=params, timeout=120)
        except requests.RequestException as exc:
            failures += 1
            if failures > 3:
                raise StopNow(f"Network problem: {exc}")
            print(f"  network problem, trying again in 20 seconds ({exc})")
            time.sleep(20)
            continue

        if resp.status_code == 200:
            return resp.json()

        try:
            reason = str(resp.json().get("reason", resp.text))[:300]
        except Exception:
            reason = resp.text[:300]

        if resp.status_code == 429:
            if "daily" in reason.lower():
                raise StopNow("Daily limit reached. Run the script again tomorrow.")
            waits += 1
            if waits > 5:
                raise StopNow(f"Still rate limited after waiting: {reason}")
            print(f"  rate limited ({reason}). Waiting 70 seconds.")
            time.sleep(70)
            continue

        if resp.status_code >= 500:
            failures += 1
            if failures > 3:
                raise StopNow(f"Open-Meteo server error {resp.status_code}: {reason}")
            time.sleep(20)
            continue

        # 400 and others: something is wrong with our request, so retrying will not help.
        raise StopNow(f"Open-Meteo rejected the request (HTTP {resp.status_code}): {reason}")


def monthly_means(payload: dict) -> list:
    """12 numbers: the mean of the daily mean temperature for each calendar month."""
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    values = daily.get("temperature_2m_mean", [])
    if not times or len(times) != len(values):
        raise StopNow("The API answer had no daily temperatures. Run --check to see the raw answer.")
    frame = pd.DataFrame({"month": [int(t[5:7]) for t in times],
                          "t": pd.to_numeric(pd.Series(values), errors="coerce")})
    result = []
    for month in range(1, 13):
        part = frame.loc[frame["month"] == month, "t"].dropna()
        if len(part) < 20:   # a month should have at least 28 days per year
            raise StopNow(f"Month {month} has only {len(part)} valid days. The answer looks incomplete.")
        result.append(round(float(part.mean()), 2))
    return result


# ---- Reading the dataset -----------------------------------------------------------------------
def dataset_months(raw) -> list:
    """The dataset's own 12 monthly averages (from the avg_temp_monthly JSON)."""
    try:
        data = json.loads(raw)
        return [float(data[str(m)]["avg"]) for m in range(1, 13)]
    except Exception:
        return [np.nan] * 12


def load_cities() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    df["key"] = [make_key(a, b) for a, b in zip(df["latitude"], df["longitude"])]
    df = df.drop_duplicates(subset="key").reset_index(drop=True)   # same coordinates = same weather
    return df[["key", "city", "country", "latitude", "longitude", "avg_temp_monthly"]]


def load_done() -> pd.DataFrame:
    if os.path.exists(OUTPUT):
        return pd.read_csv(OUTPUT, dtype={"key": str})
    return pd.DataFrame(columns=["key", "city", "country", "years"] + MONTH_COLS)


def append_row(row: dict) -> None:
    new_file = not os.path.exists(OUTPUT)
    frame = pd.DataFrame([row])[["key", "city", "country", "years"] + MONTH_COLS]
    frame.to_csv(OUTPUT, mode="a", header=new_file, index=False)


# ---- Report: how far is the dataset from ERA5? ------------------------------------------------
def report() -> None:
    done = load_done()
    if done.empty:
        return
    cities = load_cities().set_index("key")
    diffs = []
    for _, row in done.iterrows():
        if row["key"] not in cities.index:
            continue
        era = np.array([row[c] for c in MONTH_COLS], dtype=float)
        old = np.array(dataset_months(cities.loc[row["key"], "avg_temp_monthly"]), dtype=float)
        gap = era - old
        diffs.append((f'{row["city"]}, {row["country"]}', float(np.nanmean(np.abs(gap))),
                      float(np.nanmean(gap)), float(np.nanmax(np.abs(gap)))))
    if not diffs:
        return
    table = pd.DataFrame(diffs, columns=["city", "mean_abs_gap", "bias", "worst_month_gap"])
    print("\n=== Dataset compared with Open-Meteo ERA5 (", len(table), "cities ) ===")
    print(f"Average gap per month : {table['mean_abs_gap'].mean():.2f} C")
    print(f"Average bias (ERA5 minus dataset): {table['bias'].mean():+.2f} C")
    print(f"Largest single month gap          : {table['worst_month_gap'].max():.1f} C")
    print("\nCities with the biggest average gap:")
    print(table.sort_values("mean_abs_gap", ascending=False).head(8).round(2).to_string(index=False))


# ---- Modes -------------------------------------------------------------------------------------
def check_mode() -> None:
    cities = load_cities()
    first = cities.iloc[0]
    end = dt.date(END_YEAR, 12, 31)
    start = end - dt.timedelta(days=13)          # 14 days = 1 call
    print(f"Testing with {first['city']}, {first['country']} ({start} to {end})")
    payload = call_api(first["latitude"], first["longitude"], start, end)
    daily = payload.get("daily", {})
    print("Grid cell used:", payload.get("latitude"), payload.get("longitude"),
          "elevation", payload.get("elevation"))
    print("Dates :", daily.get("time", [])[:3], "...")
    print("Values:", daily.get("temperature_2m_mean", [])[:3], "...")
    print("Looks good. Now run:  python build_climate.py --limit 10")


def fetch_mode(limit: int) -> None:
    cities = load_cities()
    done_keys = set(load_done()["key"].astype(str))
    todo = cities[~cities["key"].isin(done_keys)]
    print(f"{len(cities)} cities in total, {len(done_keys)} done, {len(todo)} left.")
    if limit > 0:
        todo = todo.head(limit)

    start, end = dt.date(START_YEAR, 1, 1), dt.date(END_YEAR, 12, 31)
    cost = request_cost(start, end)
    pause = cost / (CALLS_PER_HOUR / 3600.0)          # seconds to wait so the hourly limit is respected
    usage = load_usage()
    fetched = 0

    try:
        for _, city in todo.iterrows():
            if usage["calls"] + cost > DAILY_BUDGET:
                print(f"Today's budget ({DAILY_BUDGET} calls) is used. Run again tomorrow.")
                break
            print(f"[{fetched + 1}] {city['city']}, {city['country']} ...", flush=True)
            payload = call_api(city["latitude"], city["longitude"], start, end)
            months = monthly_means(payload)
            row = {"key": city["key"], "city": city["city"], "country": city["country"], "years": YEARS_LABEL}
            row.update({col: val for col, val in zip(MONTH_COLS, months)})
            append_row(row)
            usage["calls"] += cost
            save_usage(usage)
            fetched += 1
            print(f"     saved. Calls used today: about {usage['calls']:.0f} of {DAILY_BUDGET}")
            time.sleep(pause)
    except StopNow as stop:
        print(f"\nStopped: {stop}")
        if "Daily limit" in str(stop):
            usage["calls"] = DAILY_BUDGET
            save_usage(usage)
    except KeyboardInterrupt:
        print("\nStopped by you. Progress is saved.")

    left = len(cities) - len(load_done())
    print(f"\nFetched {fetched} cities in this run. Cities still missing: {left}.")
    if left == 0:
        print(f"Everything is finished. Copy {os.path.basename(OUTPUT)} next to app.py and redeploy.")
    report()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build monthly climate data from Open-Meteo (ERA5).")
    parser.add_argument("--check", action="store_true", help="one small request to test the connection")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of new cities in this run")
    args = parser.parse_args()
    if not os.path.exists(DATASET):
        sys.exit(f"Cannot find {DATASET}. Put this script next to the dataset file.")
    try:
        check_mode() if args.check else fetch_mode(args.limit)
    except StopNow as stop:
        sys.exit(f"Stopped: {stop}")


if __name__ == "__main__":
    main()
