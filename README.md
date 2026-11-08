   ---
   title: NexVoy API
   sdk: docker
   app_port: 7860
   ---

# NexVoy API

Travel recommendations for the NexVoy tour recommendation system. Every result is calculated from the
destination dataset when the request arrives. Nothing is stored or hardcoded.

## The model

`nexvoy_model.pkl` is a scikit-learn pipeline (median imputer, scaler, region one-hot encoder) followed by an
**SVM with RBF kernel** (`C=10`, `gamma=0.01`, `class_weight='balanced'`). It was trained to estimate a
destination's budget class (Budget, Mid-range, Luxury) from 14 features. It was chosen by nested cross-validation
over Logistic Regression, SVM, KNN and Random Forest. Held-out test result: accuracy 0.8947, macro F1 0.8859.
It was saved with **scikit-learn 1.6.1**, so that exact version is pinned in `requirements.txt`.

## What happens on each request

1. **Budget filter.** Only destinations in the selected budget are ranked (the dataset's own `budget_level`).
2. **Region filter (optional).** Used when the region has at least 3 destinations in that budget. If not, all regions
   are searched and a note says so.
3. **Preference similarity.** The user's nine ratings are compared with the nine ratings stored for each destination.
   Similarity = 1 minus the average rating gap divided by 4. A perfect match is 100 percent.
4. **Temperature similarity.** The wanted temperature is compared with each of the destination's 12 monthly average
   temperatures from the dataset. Each month scores `exp(-gap / 12)`, and the destination's temperature score is the
   average over the months. A destination that stays mild all year scores higher than one that only averages out to mild
   (for example -10 C in winter and 30 C in summer). The reported `avg_temp` is still the yearly average.
   To compare with the yearly-average method, set `TEMPERATURE_METHOD = "yearly"` at the top of `app.py`.
5. **Match score.** 65 percent preference similarity plus 35 percent temperature similarity. It is shown as it is and
   never rescaled, so the top result is not forced to a high number.
6. **Model.** The trained model scores every destination. It is used for `predicted_budget` (the average of its
   probabilities across the 10 destinations that best match the user's ratings and temperature) and for each result's
   `budget_confidence`. **The model does not rank destinations.** The ranking comes from steps 3 to 5.
   The model was trained on these same destinations, so `budget_confidence` is an in-sample estimate.

The weights are named constants at the top of `app.py`.

## Files

| File | Purpose |
|---|---|
| `app.py` | The API |
| `Dockerfile` | Builds the image |
| `.dockerignore` | Keeps tests, notebook and README out of the image (the file name starts with a dot) |
| `requirements.txt` | Pinned packages |
| `nexvoy_model.pkl` | Your trained model |
| `nexvoy_model_info.json` | Your model settings |
| `Worldwide_Travel_Cities_FIXED.csv` | **Your dataset. Copy it into this folder before building.** |
| `verify_dataset.py` | Checks your dataset against what the API expects (runs inside the image) |
| `nexvoy_cli.py` | Terminal client, runs on your computer |
| `test_recommendations.py` | Checks that changing the inputs changes the results |

## Run locally with Docker Desktop (Windows PowerShell)

Start Docker Desktop and wait until it says it is running. Then, in this folder:

```powershell
docker build -t nexvoy .
docker run -d --name nexvoy -p 7860:7860 nexvoy
docker logs nexvoy
```

The logs should end with `Loaded ... destinations. Model: SVM (RBF).` and `Uvicorn running on http://0.0.0.0:7860`.

```powershell
docker ps                       # STATUS should become "healthy" after about 40 seconds
curl.exe http://127.0.0.1:7860/health
```

Check your dataset (recommended once):

```powershell
docker run --rm nexvoy python verify_dataset.py
```

Stop, restart, remove, rebuild after changing a file:

```powershell
docker stop nexvoy
docker start nexvoy
docker rm -f nexvoy
docker build -t nexvoy . ; docker rm -f nexvoy ; docker run -d --name nexvoy -p 7860:7860 nexvoy
```

If port 7860 is busy, use `-p 8000:7860` and replace 7860 with 8000 in the URLs below.

## Test it

The CLI and tests use only Python's standard library, so any Python 3 on Windows works (no installs):

```powershell
python nexvoy_cli.py
python test_recommendations.py
```

Interactive page: open http://127.0.0.1:7860/docs in a browser.

`test_recommendations.py` prints `PASS` / `FAIL` for things that must be true for any dataset, and `NOTE` for things
that are normally true but depend on your data (for example "warmer wish gives warmer results"). A `NOTE` does not
fail the run. Read it instead of forcing it to pass.

## Optional settings (environment variables)

| Name | Use |
|---|---|
| `NEXVOY_API_KEY` | If set, every `/recommend` and `/predict` call must send it in the `X-API-Key` header |
| `NEXVOY_ALLOWED_ORIGINS` | Websites allowed to call the API from a browser, separated by commas. Default is all (`*`) |

```powershell
docker run -d --name nexvoy -p 7860:7860 -e NEXVOY_API_KEY=choose-a-long-secret nexvoy
python test_recommendations.py http://127.0.0.1:7860 choose-a-long-secret
```

Calls from your ASP.NET controller are made by your server, so CORS does not affect them. Keep the key in
`Web.config`, never in a view or JavaScript. Before the API is reachable from the internet, set a key and restrict
the origins.

## API

`GET /health` status, destination count, temperature method.

`GET /model-info` returns `nexvoy_model_info.json`.

`POST /recommend`

```json
{
  "temperature": 15,
  "budget": "Mid-range",
  "region": "europe",
  "preferences": {
    "culture": 4, "adventure": 3, "nature": 5, "beaches": 2, "nightlife": 5,
    "cuisine": 5, "wellness": 4, "urban": 4, "seclusion": 5
  },
  "top_n": 5
}
```

`region` is optional. `budget` is Budget, Mid-range or Luxury. Ratings are whole numbers from 1 to 5. `top_n` is 1 to 10. `travel_month` (1–12) remains optional. For forecast-based ranking, send both `travel_start_date` and `travel_end_date` as `YYYY-MM-DD`.
The reply has `predicted_budget`, `selected_budget`, `notes`, `region_applied`, and a `results` list. Each result has
`rank`, `city`, `country`, `region`, `budget_level`, `avg_hotel_usd`, `avg_temp`, `ideal_trip_days`,
`ideal_trip_labels`, `match_score`, `preference_match`, `temperature_match`, `budget_confidence` and `description`.

`POST /predict` takes a list of destinations with their 14 model values and returns the model's budget class and
probabilities for each.

Errors: 422 for invalid input with a clear message, 401 for a wrong API key, 500 for an unexpected server problem.

From ASP.NET on the same Windows laptop, call `http://localhost:7860/recommend` with a JSON body as above.
The property names in the reply are lower case with underscores (for example `match_score`).

## Free hosting options

- **Hugging Face Spaces (Docker).** Hugging Face's documentation says creating a Docker Space requires a paid plan
  (PRO for personal accounts). Free accounts can still create Static Spaces, but those cannot run this API. Not an
  option if you will not pay.
- **Render (free web service).** Runs Docker images, no card needed to start. Limits: 512 MB RAM, sleeps after 15
  minutes without requests (the next call waits about a minute), not meant for production.
- **Google Cloud Run.** Has an always-free monthly allowance, but needs a billing card on file. Scales to zero, so the
  first call after idle is slower.
- **Run it on your own machine.** Docker Desktop as above. Free, and enough for a final-year project demo.

Check each provider's current terms before relying on them; free tiers change often.


## Weather data from Open-Meteo (version 2.3)

The temperature match can now use real monthly temperatures from the Open-Meteo Historical Weather API (ERA5 data), and an optional travel month.

**Travel month.** `POST /recommend` accepts an optional `"travel_month"` (1 to 12). When it is sent, the temperature match uses that month only. When it is missing, all 12 months are used, exactly as before. Each result then also has `month_temp` and `temperature_source`.

**Climate file.** `build_climate.py` makes `city_climate_era5.csv`. If this file sits next to `app.py`, its values replace the dataset's monthly temperatures for the temperature match. If it is missing, the API works exactly as before. The model's own inputs (`avg_temp`, `temp_std`) are never changed.

```
pip install requests pandas numpy
python build_climate.py --check       # one small request, tests the connection
python build_climate.py --limit 10    # fetch 10 cities and compare them with the dataset
python build_climate.py               # fetch as many cities as today's budget allows
```

The free Open-Meteo plan allows 10,000 calls per day, so the full dataset takes about 10 days. The script can be stopped and run again at any time. Deploy only when it reports that everything is finished. The free plan is for non-commercial use and needs credit to Open-Meteo.com.


### 16-day trip forecast

`POST /recommend` also accepts optional `travel_start_date` and `travel_end_date` values in `YYYY-MM-DD` format.
Provide both dates together. If the **entire trip** fits inside Open-Meteo's next 16 forecast days, the API requests daily
mean/minimum/maximum temperature, maximum precipitation probability, and weather code for a shortlist of up to 50
candidate destinations. The temperature part of the ranking uses the average daily mean temperature across the trip;
the existing 65% preference / 35% temperature weights remain unchanged. The saved SVM and its input features are not
modified or retrained.

Example request additions:

```json
{
  "travel_start_date": "2026-11-12",
  "travel_end_date": "2026-11-16"
}
```

When the dates are outside the 16-day window, cross its boundary, or Open-Meteo is unavailable or returns incomplete
data, the API falls back to the existing monthly climate method for the **whole recommendation request**. It does not
mix forecast and monthly-climate scores in one ranking. The response includes `weather_source` (`open_meteo_forecast`
or `monthly_climate`), the requested dates, and `forecast_candidates_evaluated`. Forecast-backed results also include
`forecast_avg_temp`, `forecast_precipitation_probability_max`, `forecast_weather_code`, and a
`temperature_source` label. If no dates are supplied, existing behavior is unchanged.

The free Open-Meteo Forecast endpoint does not require an API key for non-commercial use. Do not add an API key to
`.env`, Docker, or ASP.NET for this endpoint. Check Open-Meteo's current terms before using it commercially.
