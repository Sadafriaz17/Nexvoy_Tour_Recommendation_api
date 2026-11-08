# NexVoy API
# - Loads the trained model and the destination dataset once, at startup.
# - POST /recommend ranks destinations for a user, fully from the dataset (nothing is hardcoded).
# - POST /predict returns the model's budget class for destinations you send in.
#
# How the pieces fit together:
#   * RANKING   = similarity of the user's 9 ratings + similarity of the wanted temperature.
#                 The trained model is NOT used for ranking.
#   * MODEL     = the saved scikit-learn pipeline (an SVM with RBF kernel). It estimates which budget
#                 class a destination belongs to. It feeds "predicted_budget" and "budget_confidence".
#   * CLIMATE   = (v2.2) the monthly temperatures used for the temperature match. If city_climate_era5.csv
#                 exists (made by build_climate.py from the Open-Meteo Historical Weather API), those values
#                 are used. Otherwise the dataset's own monthly values are used, exactly as before.
#                 The model's own inputs (avg_temp, temp_std) are NOT changed, so the model stays intact.
#   * MONTH     = (v2.2) an optional "travel_month" (1 to 12). When it is sent, the temperature match uses
#                 that month only. When it is missing, all 12 months are used, exactly as before.
import json
import logging
import os
import secrets
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger("nexvoy")
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------------------
# Files and settings
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "nexvoy_model.pkl")
INFO_PATH = os.path.join(BASE_DIR, "nexvoy_model_info.json")
DATA_PATH = os.path.join(BASE_DIR, "Worldwide_Travel_Cities_FIXED.csv")
CLIMATE_PATH = os.path.join(BASE_DIR, "city_climate_era5.csv")   # made by build_climate.py (optional)

# How the final compatibility score is built (both parts are similarities from 0 to 1).
WEIGHT_PREFERENCES = 0.65   # how close the city is to the user's 9 ratings
WEIGHT_TEMPERATURE = 0.35   # how close the city's temperature is to the wanted temperature
TEMP_DECAY_C = 12.0         # a month this many degrees away keeps about 37 percent of the temperature score
FORECAST_DAYS = 16
FORECAST_SHORTLIST_SIZE = 50  # shortlist before querying Open-Meteo, rather than querying every dataset city
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
PROFILE_SIZE = 10           # best matching cities used for the model's budget estimate
MIN_REGION_RESULTS = 3      # fewer matches than this in the chosen region means all regions are shown

# "monthly": temperature similarity is the average, over the 12 months, of how close each month's
#            average temperature is to the wanted temperature. A city that is always mild beats a
#            city that only averages out to mild (for example -10 C in winter and 30 C in summer).
# "yearly" : compare the wanted temperature with the single yearly average only.
TEMPERATURE_METHOD = "monthly"

PREF_COLS = ["culture", "adventure", "nature", "beaches", "nightlife",
             "cuisine", "wellness", "urban", "seclusion"]

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# Optional protection. If NEXVOY_API_KEY is set, calls must send it in the X-API-Key header.
API_KEY = os.environ.get("NEXVOY_API_KEY", "")
# Which websites may call the API from a browser. Use a comma separated list to restrict it.
# Calls from a server (for example an ASP.NET controller) are not affected by CORS.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("NEXVOY_ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# ----------------------------------------------------------------------------
# Load the model files
# ----------------------------------------------------------------------------
with open(INFO_PATH, "r", encoding="utf-8") as f:
    INFO = json.load(f)

if INFO["sklearn_version"] != sklearn.__version__:
    raise RuntimeError(
        f"scikit-learn version mismatch. The model was saved with {INFO['sklearn_version']} "
        f"but {sklearn.__version__} is installed. "
        f"Run: pip install scikit-learn=={INFO['sklearn_version']}"
    )

BUNDLE = joblib.load(MODEL_PATH)
PIPELINE = BUNDLE["pipeline"]
FEATURE_COLS: List[str] = INFO["feature_cols"]          # exact order used in training
KNOWN_REGIONS: List[str] = INFO["known_regions"]
LABELS: List[str] = INFO["label_names"]
MODEL_NAME: str = INFO["model_name"]

if list(PIPELINE.classes_) != LABELS:
    raise RuntimeError("The model classes do not match nexvoy_model_info.json.")
if list(getattr(PIPELINE, "feature_names_in_", FEATURE_COLS)) != FEATURE_COLS:
    raise RuntimeError("The model's feature names or order do not match nexvoy_model_info.json.")
if BUNDLE.get("feature_cols") != FEATURE_COLS:
    raise RuntimeError("The feature list inside the .pkl does not match nexvoy_model_info.json.")
missing_prefs = [c for c in PREF_COLS if c not in FEATURE_COLS]
if missing_prefs:
    raise RuntimeError(f"Preference columns missing from the model features: {missing_prefs}")
if TEMPERATURE_METHOD not in ("monthly", "yearly"):
    raise RuntimeError("TEMPERATURE_METHOD must be 'monthly' or 'yearly'.")


# ----------------------------------------------------------------------------
# Load the dataset (same cleaning as training) and run the model on every city
# ----------------------------------------------------------------------------
def parse_duration(raw):
    try:
        items = json.loads(raw)
        order = ["Day trip", "Weekend", "Short trip", "One week", "Long trip"]
        days = {"Day trip": "1 day", "Weekend": "2-3 days", "Short trip": "3-5 days",
                "One week": "5-7 days", "Long trip": "1-2+ weeks"}
        items = sorted(items, key=lambda x: order.index(x) if x in order else 99)
        return " / ".join(days.get(i, "") for i in items), " / ".join(items)
    except Exception:
        return "Flexible", "Flexible"


def monthly_averages(raw) -> List[float]:
    """The monthly average temperatures from the avg_temp_monthly JSON ({"January": {"avg": 3.1, ...}, ...})."""
    if not isinstance(raw, str):
        return []
    try:
        data = json.loads(raw)
        entries = data.values() if isinstance(data, dict) else data
        values = []
        for entry in entries:
            value = pd.to_numeric(entry.get("avg") if isinstance(entry, dict) else entry, errors="coerce")
            if pd.notna(value):
                values.append(float(value))
        return values
    except Exception:
        return []


def monthly_by_month(raw) -> np.ndarray:
    """12 numbers, one per calendar month (January first). A missing month is NaN.
    The dataset stores months as "1" to "12"; month names such as "January" also work."""
    out = np.full(12, np.nan)
    if not isinstance(raw, str):
        return out
    try:
        data = json.loads(raw)
    except Exception:
        return out
    if not isinstance(data, dict):
        return out
    for i, name in enumerate(MONTH_NAMES):
        entry = data.get(str(i + 1), data.get(name))
        value = pd.to_numeric(entry.get("avg") if isinstance(entry, dict) else entry, errors="coerce")
        if pd.notna(value):
            out[i] = float(value)
    return out


def climate_key(lat, lon) -> str:
    # Must match make_key() in build_climate.py.
    return f"{float(lat):.3f},{float(lon):.3f}"


def merge_climate(df: pd.DataFrame, dataset_months: np.ndarray):
    """Use Open-Meteo values where city_climate_era5.csv has them, the dataset's own values elsewhere.
    Returns (n cities x 12 array, one source label per city, label for the health page)."""
    sources = np.array(["Dataset"] * len(df), dtype=object)
    if not os.path.exists(CLIMATE_PATH):
        return dataset_months, sources, "Dataset (no Open-Meteo file found)"
    clim = pd.read_csv(CLIMATE_PATH, dtype={"key": str})
    month_cols = [f"m{i}" for i in range(1, 13)]
    clim = clim.dropna(subset=month_cols)
    table = {str(k): row.astype(float) for k, row in zip(clim["key"], clim[month_cols].to_numpy())}
    years = str(clim["years"].iloc[0]) if len(clim) else ""
    label = f"Open-Meteo ERA5, {years}" if years else "Open-Meteo ERA5"

    merged = dataset_months.copy()
    for i, (lat, lon) in enumerate(zip(df["latitude"], df["longitude"])):
        values = table.get(climate_key(lat, lon))
        if values is not None:
            merged[i] = values
            sources[i] = label
    covered = int((sources != "Dataset").sum())
    if covered < len(df):
        log.warning("Open-Meteo climate covers only %d of %d destinations. Run build_climate.py until it "
                    "reports that everything is finished, so every city is scored the same way.",
                    covered, len(df))
    return merged, sources, f"{label} ({covered} of {len(df)} destinations)"


def plain_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def load_destinations() -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_PATH)
    df = df[df["budget_level"].isin(LABELS) & df["region"].isin(KNOWN_REGIONS)]

    # Same cleaning as training: drop cities whose copies disagree on budget,
    # then keep one copy of each city. The CSV file itself is not changed.
    label_count = df.groupby(["city", "country"])["budget_level"].transform("nunique")
    df = df[label_count == 1].drop_duplicates(subset=["city", "country"]).reset_index(drop=True)

    # Some cities are spelled twice with and without accents (for example Medellin and Medellín).
    # Treat them as one city so a destination never appears twice in the results.
    df["_key"] = df["city"].map(plain_text) + "|" + df["country"].map(plain_text)
    label_count = df.groupby("_key")["budget_level"].transform("nunique")
    df = df[label_count == 1].drop_duplicates(subset="_key").drop(columns="_key").reset_index(drop=True)

    # Temperature: yearly average and seasonal swing (same as training) plus the monthly values.
    df["_monthly"] = df["avg_temp_monthly"].apply(monthly_averages)
    df["avg_temp"] = df["_monthly"].apply(lambda v: float(np.mean(v)) if len(v) else np.nan)
    df["temp_std"] = df["_monthly"].apply(lambda v: float(np.std(v)) if len(v) else np.nan)

    for col in PREF_COLS + ["cost_of_living_index", "tourist_infra_score", "avg_hotel_usd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # A city without ratings or temperature cannot be compared, so it is left out.
    df = df.dropna(subset=PREF_COLS + ["avg_temp"]).reset_index(drop=True)

    # One row per city, one column per calendar month (column 0 = January). The model above keeps using
    # the dataset's own avg_temp and temp_std. Only the temperature MATCH uses the Open-Meteo values.
    dataset_months = np.vstack([monthly_by_month(raw) for raw in df["avg_temp_monthly"]])
    monthly, sources, climate_label = merge_climate(df, dataset_months)
    df["clim_avg"] = np.nanmean(monthly, axis=1)     # yearly average used for display and "yearly" mode
    df["temp_source"] = sources
    df = df.drop(columns="_monthly")

    durations = df["ideal_durations"].apply(parse_duration)
    df["trip_days"] = durations.apply(lambda d: d[0])
    df["trip_labels"] = durations.apply(lambda d: d[1])

    # The trained model scores every city: how likely is each budget class.
    proba = PIPELINE.predict_proba(df[FEATURE_COLS])
    for i, cls in enumerate(PIPELINE.classes_):
        df[f"proba_{cls}"] = proba[:, i]
    return df, monthly, climate_label


DEST, MONTHLY, CLIMATE_LABEL = load_destinations()
if DEST.empty:
    raise RuntimeError("No usable destinations were found in the dataset.")
log.info("Loaded %d destinations. Model: %s. Temperature method: %s. Climate: %s",
         len(DEST), MODEL_NAME, TEMPERATURE_METHOD, CLIMATE_LABEL)

# ----------------------------------------------------------------------------
# App, CORS and errors
# ----------------------------------------------------------------------------
app = FastAPI(title="NexVoy API", version="2.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    log.exception("Unexpected error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server."})


def check_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and not secrets.compare_digest((x_api_key or "").encode("utf-8"), API_KEY.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ----------------------------------------------------------------------------
# Request and response shapes
# ----------------------------------------------------------------------------
class Preferences(BaseModel):
    culture: int = Field(ge=1, le=5)
    adventure: int = Field(ge=1, le=5)
    nature: int = Field(ge=1, le=5)
    beaches: int = Field(ge=1, le=5)
    nightlife: int = Field(ge=1, le=5)
    cuisine: int = Field(ge=1, le=5)
    wellness: int = Field(ge=1, le=5)
    urban: int = Field(ge=1, le=5)
    seclusion: int = Field(ge=1, le=5)


class RecommendRequest(BaseModel):
    temperature: float = Field(ge=-50, le=50, description="Preferred average temperature in °C")
    budget: str = Field(max_length=30, description="Budget, Mid-range or Luxury")
    region: Optional[str] = Field(default=None, max_length=30, description="Optional region, for example europe")
    preferences: Preferences
    top_n: int = Field(default=5, ge=1, le=10)
    travel_month: Optional[int] = Field(default=None, ge=1, le=12,
                                        description="Optional month of travel, 1 = January ... 12 = December")
    travel_start_date: Optional[date] = Field(default=None, description="First day of the trip (YYYY-MM-DD)")
    travel_end_date: Optional[date] = Field(default=None, description="Last day of the trip (YYYY-MM-DD)")

    @model_validator(mode="after")
    def validate_trip_dates(self):
        if (self.travel_start_date is None) != (self.travel_end_date is None):
            raise ValueError("Provide both travel_start_date and travel_end_date, or neither.")
        if self.travel_start_date and self.travel_end_date:
            if self.travel_end_date < self.travel_start_date:
                raise ValueError("travel_end_date must be on or after travel_start_date.")
            if (self.travel_end_date - self.travel_start_date).days > 30:
                raise ValueError("Trip date range cannot exceed 31 calendar days.")
        return self


class Recommendation(BaseModel):
    rank: int
    city: str
    country: str
    region: str
    budget_level: str
    avg_hotel_usd: Optional[int]
    avg_temp: float             # yearly average of the monthly temperatures
    ideal_trip_days: str
    ideal_trip_labels: str
    match_score: float          # overall compatibility, 0 to 100
    preference_match: float     # similarity of the 9 ratings, 0 to 100
    temperature_match: float    # similarity of the temperature, 0 to 100
    budget_confidence: float    # model probability that this city is in the selected budget class
    description: str
    month_temp: Optional[float] = None   # average temperature in the travel month (only when a month is sent)
    temperature_source: str = ""         # where this city's temperatures come from
    forecast_avg_temp: Optional[float] = None
    forecast_precipitation_probability_max: Optional[float] = None
    forecast_weather_code: Optional[int] = None


class RecommendResponse(BaseModel):
    model: str
    selected_budget: str
    predicted_budget: str
    predicted_budget_probabilities: Dict[str, float]
    predicted_budget_basis: str
    preferred_temperature: float
    region_applied: Optional[str]
    travel_month: Optional[int] = None
    travel_month_name: Optional[str] = None
    travel_start_date: Optional[date] = None
    travel_end_date: Optional[date] = None
    weather_source: str = "monthly_climate"
    forecast_candidates_evaluated: int = 0
    climate_source: str = ""
    notes: List[str]
    count: int
    results: List[Recommendation]


class City(BaseModel):
    # Ranges are wide on purpose: they only reject impossible values, not unusual real cities.
    culture: float = Field(ge=1, le=5)
    adventure: float = Field(ge=1, le=5)
    nature: float = Field(ge=1, le=5)
    beaches: float = Field(ge=1, le=5)
    nightlife: float = Field(ge=1, le=5)
    cuisine: float = Field(ge=1, le=5)
    wellness: float = Field(ge=1, le=5)
    urban: float = Field(ge=1, le=5)
    seclusion: float = Field(ge=1, le=5)
    avg_temp: float = Field(ge=-60, le=60)
    temp_std: float = Field(ge=0, le=40)
    cost_of_living_index: float = Field(ge=0, le=1000)
    tourist_infra_score: float = Field(ge=0, le=1000)
    region: str


class PredictRequest(BaseModel):
    cities: List[City] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    budget_level: str
    probabilities: Dict[str, float]


class PredictResponse(BaseModel):
    model: str
    predictions: List[Prediction]


_missing = set(FEATURE_COLS) - set(City.model_fields)
if _missing:
    raise RuntimeError(f"Request schema is missing model features: {sorted(_missing)}")

# ----------------------------------------------------------------------------
# Input cleaning
# ----------------------------------------------------------------------------
_BUDGETS = {"budget": "Budget", "mid-range": "Mid-range", "midrange": "Mid-range", "luxury": "Luxury"}


def clean_budget(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "-").replace("_", "-")
    if key not in _BUDGETS or _BUDGETS[key] not in LABELS:
        raise HTTPException(status_code=422, detail=f"Unknown budget '{value}'. Use one of: {LABELS}")
    return _BUDGETS[key]


def clean_region(value: Optional[str]) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    key = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in KNOWN_REGIONS:
        raise HTTPException(status_code=422, detail=f"Unknown region '{value}'. Allowed regions: {KNOWN_REGIONS}")
    return key


def pretty(region: str) -> str:
    return region.replace("_", " ").title()


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def month_temperatures(frame: pd.DataFrame, month: int) -> np.ndarray:
    """The average temperature of one calendar month (1 to 12) for every city in `frame`.
    If a city has no value for that month, its yearly average is used instead."""
    values = MONTHLY[frame.index.to_numpy(), month - 1]
    return np.where(np.isnan(values), frame["clim_avg"].to_numpy(float), values)


def temperature_similarity(frame: pd.DataFrame, desired_temp: float, month: Optional[int] = None) -> np.ndarray:
    """0 to 1 for every city in `frame` (its index labels are positions in DEST / MONTHLY)."""
    positions = frame.index.to_numpy()
    if month:
        # The traveller knows when they will go, so compare with that month only.
        return np.exp(-np.abs(month_temperatures(frame, month) - desired_temp) / TEMP_DECAY_C)
    if TEMPERATURE_METHOD == "monthly":
        gaps = np.abs(MONTHLY[positions] - desired_temp)               # cities x months
        return np.nanmean(np.exp(-gaps / TEMP_DECAY_C), axis=1)       # missing months are ignored
    return np.exp(-np.abs(frame["clim_avg"].to_numpy(float) - desired_temp) / TEMP_DECAY_C)


def compatibility(frame: pd.DataFrame, user_ratings: np.ndarray, desired_temp: float,
                  month: Optional[int] = None):
    """Return (preference similarity, temperature similarity, overall score), each from 0 to 1."""
    # Ratings are on a 1 to 5 scale, so the largest possible gap on one rating is 4.
    pref = 1.0 - np.abs(frame[PREF_COLS].to_numpy(float) - user_ratings).mean(axis=1) / 4.0
    temp = temperature_similarity(frame, desired_temp, month)
    overall = WEIGHT_PREFERENCES * pref + WEIGHT_TEMPERATURE * temp
    return pref, temp, overall


def rank_order(overall: np.ndarray, pref: np.ndarray) -> np.ndarray:
    # Highest score first. Equal scores are ordered by preference similarity.
    return np.lexsort((-pref, -overall))


# ----------------------------------------------------------------------------
# Open-Meteo short-range forecast (does not alter/retrain the saved SVM)
# ----------------------------------------------------------------------------
def forecast_window_covers(start: date, end: date) -> bool:
    """True only when the complete trip fits in the next 16 calendar forecast days."""
    today = datetime.now(timezone.utc).date()
    last_day = today + timedelta(days=FORECAST_DAYS - 1)
    return start >= today and end <= last_day


def fetch_trip_forecasts(frame: pd.DataFrame, start: date, end: date) -> Dict[int, dict]:
    """Fetch one batched Open-Meteo request for a shortlist of city rows.

    Returns a mapping keyed by the original DEST row index. Any network/API/data error
    is handled by the caller as a monthly-climate fallback.
    """
    if frame.empty:
        return {}
    coords = frame[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
    valid = coords.dropna()
    if valid.empty:
        return {}
    # Open-Meteo accepts comma-separated coordinates for multi-location requests.
    latitudes = ",".join(f"{float(v):.4f}" for v in valid["latitude"])
    longitudes = ",".join(f"{float(v):.4f}" for v in valid["longitude"])
    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
        "forecast_days": FORECAST_DAYS,
        "timezone": ",".join(["auto"] * len(valid)),
        "temperature_unit": "celsius",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    locations = payload if isinstance(payload, list) else [payload]
    row_indices = list(valid.index)
    output: Dict[int, dict] = {}
    for idx, item in zip(row_indices, locations):
        daily = item.get("daily") or {}
        dates = daily.get("time") or []
        means = daily.get("temperature_2m_mean") or []
        precip = daily.get("precipitation_probability_max") or []
        codes = daily.get("weather_code") or []
        selected = [i for i, day in enumerate(dates) if start.isoformat() <= day <= end.isoformat()]
        temps = [means[i] for i in selected if i < len(means) and means[i] is not None]
        rain = [precip[i] for i in selected if i < len(precip) and precip[i] is not None]
        weather_codes = [codes[i] for i in selected if i < len(codes) and codes[i] is not None]
        expected_days = (end - start).days + 1
        if len(temps) != expected_days:
            continue
        # The maximum daily precipitation probability is a useful trip-level caution signal.
        output[int(idx)] = {
            "avg_temp": float(np.mean(temps)),
            "precipitation_probability_max": float(max(rain)) if rain else None,
            "weather_code": int(pd.Series(weather_codes).mode().iloc[0]) if weather_codes else None,
        }
    if len(output) != len(frame):
        log.warning("Open-Meteo returned complete trip data for %d of %d shortlisted destinations; using monthly fallback for consistency.",
                    len(output), len(frame))
        return {}
    return output


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "NexVoy API",
        "model": MODEL_NAME,
        "endpoints": ["GET /health", "GET /model-info", "POST /recommend", "POST /predict", "GET /docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "labels": LABELS,
        "destinations": int(len(DEST)),
        "temperature_method": TEMPERATURE_METHOD,
        "climate_source": CLIMATE_LABEL,
        "sklearn_version": sklearn.__version__,
    }


@app.get("/model-info")
def model_info():
    return INFO


@app.post("/recommend", response_model=RecommendResponse, dependencies=[Depends(check_key)])
def recommend(req: RecommendRequest):
    budget = clean_budget(req.budget)
    region = clean_region(req.region)
    month = req.travel_month
    user_ratings = np.array([getattr(req.preferences, c) for c in PREF_COLS], dtype=float)
    notes: List[str] = []
    trip_forecasts: Dict[int, dict] = {}
    weather_source = "monthly_climate"
    forecast_candidates_evaluated = 0
    forecast_active = False
    if req.travel_start_date and req.travel_end_date:
        if forecast_window_covers(req.travel_start_date, req.travel_end_date):
            forecast_active = True
        else:
            notes.append("The complete trip is outside the 16-day Open-Meteo forecast window. Monthly climate averages were used instead.")

    # Step 1: region. Use it only if enough destinations in the chosen budget exist there.
    pool = DEST
    region_applied = None
    if region:
        in_region = DEST[DEST["region"] == region]
        in_region_budget = in_region[in_region["budget_level"] == budget]
        if len(in_region_budget) >= MIN_REGION_RESULTS:
            pool, region_applied = in_region, region
        else:
            notes.append(f"Only {len(in_region_budget)} {budget} destinations in {pretty(region)}. "
                         f"Showing all regions.")

    # Step 2: the model's budget estimate for this user's taste. Rank every city in the pool
    # (all budgets) by compatibility, then average the model's probabilities for the best ones.
    pref_all, temp_all, overall_all = compatibility(pool, user_ratings, req.temperature, month)
    best = rank_order(overall_all, pref_all)[:PROFILE_SIZE]
    mean_proba = {cls: float(pool.iloc[best][f"proba_{cls}"].mean()) for cls in LABELS}
    predicted_budget = max(mean_proba, key=mean_proba.get)

    # Step 3: keep only the selected budget and rank.
    candidates = pool[pool["budget_level"] == budget]
    results: List[Recommendation] = []
    if candidates.empty:
        notes.append(f"No {budget} destinations were found for these filters.")
    else:
        pref, temp, overall = compatibility(candidates, user_ratings, req.temperature, month)
        candidate_frame = candidates
        forecast_scores = None
        if forecast_active:
            # First narrow to a manageable group using the existing preference + monthly-climate score.
            # Then replace temperature similarity with actual trip forecast for those shortlisted cities.
            shortlist_order = rank_order(overall, pref)[:min(FORECAST_SHORTLIST_SIZE, len(candidates))]
            shortlist = candidates.iloc[shortlist_order]
            try:
                trip_forecasts = fetch_trip_forecasts(shortlist, req.travel_start_date, req.travel_end_date)
                if trip_forecasts:
                    forecast_positions = [i for i, row_idx in enumerate(shortlist.index) if row_idx in trip_forecasts]
                    candidate_frame = shortlist.iloc[forecast_positions]
                    pref = 1.0 - np.abs(candidate_frame[PREF_COLS].to_numpy(float) - user_ratings).mean(axis=1) / 4.0
                    forecast_temps = np.array([trip_forecasts[i]["avg_temp"] for i in candidate_frame.index], dtype=float)
                    forecast_scores = np.exp(-np.abs(forecast_temps - req.temperature) / TEMP_DECAY_C)
                    overall = WEIGHT_PREFERENCES * pref + WEIGHT_TEMPERATURE * forecast_scores
                    temp = forecast_scores
                    forecast_candidates_evaluated = len(candidate_frame)
                    weather_source = "open_meteo_forecast"
                else:
                    notes.append("Open-Meteo did not return usable trip weather. Monthly climate averages were used instead.")
            except Exception as exc:
                log.warning("Open-Meteo forecast failed; using monthly climate fallback: %s", exc)
                notes.append("Open-Meteo was unavailable. Monthly climate averages were used instead.")
        order = rank_order(overall, pref)[: req.top_n]
        month_values = month_temperatures(candidate_frame, month) if month and weather_source == "monthly_climate" else None
        for position, idx in enumerate(order, start=1):
            row = candidate_frame.iloc[idx]
            hotel = row["avg_hotel_usd"]
            forecast = trip_forecasts.get(candidate_frame.index[idx]) if weather_source == "open_meteo_forecast" else None
            results.append(Recommendation(
                rank=position,
                city=str(row["city"]),
                country=str(row["country"]),
                region=pretty(str(row["region"])),
                budget_level=str(row["budget_level"]),
                avg_hotel_usd=int(hotel) if pd.notna(hotel) else None,
                avg_temp=round(float(row["clim_avg"]), 1),
                ideal_trip_days=str(row["trip_days"]),
                ideal_trip_labels=str(row["trip_labels"]),
                match_score=round(float(overall[idx]) * 100, 1),
                preference_match=round(float(pref[idx]) * 100, 1),
                temperature_match=round(float(temp[idx]) * 100, 1),
                budget_confidence=round(float(row[f"proba_{budget}"]) * 100, 1),
                description=str(row["short_description"]) if pd.notna(row.get("short_description")) else "",
                month_temp=round(float(month_values[idx]), 1) if month_values is not None else None,
                temperature_source=("Open-Meteo 16-day forecast" if forecast else str(row["temp_source"])),
                forecast_avg_temp=round(forecast["avg_temp"], 1) if forecast else None,
                forecast_precipitation_probability_max=forecast["precipitation_probability_max"] if forecast else None,
                forecast_weather_code=forecast["weather_code"] if forecast else None,
            ))

    return RecommendResponse(
        model=MODEL_NAME,
        selected_budget=budget,
        predicted_budget=predicted_budget,
        predicted_budget_probabilities={k: round(v, 4) for k, v in mean_proba.items()},
        predicted_budget_basis=(f"Average of the model's probabilities for the {len(best)} destinations "
                                f"that best match your ratings and temperature."),
        preferred_temperature=req.temperature,
        region_applied=pretty(region_applied) if region_applied else None,
        travel_month=month,
        travel_month_name=MONTH_NAMES[month - 1] if month else None,
        travel_start_date=req.travel_start_date,
        travel_end_date=req.travel_end_date,
        weather_source=weather_source,
        forecast_candidates_evaluated=forecast_candidates_evaluated,
        climate_source=CLIMATE_LABEL,
        notes=notes,
        count=len(results),
        results=results,
    )


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(check_key)])
def predict(req: PredictRequest):
    bad = sorted({c.region for c in req.cities if c.region not in KNOWN_REGIONS})
    if bad:
        raise HTTPException(status_code=422, detail=f"Unknown region {bad}. Allowed regions: {KNOWN_REGIONS}")
    frame = pd.DataFrame([c.model_dump() for c in req.cities])[FEATURE_COLS]
    # Note: for an SVM, `predict` and the highest probability can occasionally disagree on borderline
    # cases (the probabilities come from a separate calibration step). `budget_level` is the model's
    # class decision, the same one used for the test scores in the notebook.
    labels = PIPELINE.predict(frame)
    probas = PIPELINE.predict_proba(frame)
    classes = list(PIPELINE.classes_)
    return PredictResponse(
        model=MODEL_NAME,
        predictions=[
            Prediction(budget_level=str(lbl),
                       probabilities={c: round(float(p), 4) for c, p in zip(classes, row)})
            for lbl, row in zip(labels, probas)
        ],
    )
