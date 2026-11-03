# NexVoy terminal client.
# Asks the questions, sends them to the NexVoy API, and prints the recommendations.
# Usage:
#   python nexvoy_cli.py                                   (uses http://127.0.0.1:7860)
#   python nexvoy_cli.py http://127.0.0.1:7860
# If the API has a key, set it first:  set NEXVOY_API_KEY=your-key   (Windows)
import json
import os
import sys
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

API_URL = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NEXVOY_API_URL", "http://127.0.0.1:7860")).rstrip("/")
API_KEY = os.environ.get("NEXVOY_API_KEY", "")


def call(method, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(API_URL + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if API_KEY:
        req.add_header("X-API-Key", API_KEY)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            detail = ""
        sys.exit(f"\n❌ The API returned an error ({e.code}): {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"\n❌ Could not reach the API at {API_URL}\n   {e.reason}")


def ask(prompt):
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        sys.exit("\n\nStopped.")


info = call("GET", "/model-info")
known_regions = info["known_regions"]

print("=" * 60)
print("   🌍  AI TRAVEL RECOMMENDATION SYSTEM  🌍")
print("   Ranking : similarity to your ratings and temperature")
print(f"   Model   : {info['model_name']} (estimates the budget class)")
print("=" * 60)

# STEP 1: Temperature
while True:
    try:
        desired_temp = float(ask("\n🌡️  Preferred average temperature (°C, e.g. 22): "))
        if -50 <= desired_temp <= 50:
            break
        print("   ↳ Enter a temperature between -50 and 50")
    except ValueError:
        print("   ↳ Please type a number")

# STEP 2: Budget
print("\n💰 What is your travel budget per night?")
print("   1 → Budget      (under $75/night)")
print("   2 → Mid-range   ($75–$220/night)")
print("   3 → Luxury      ($220+/night)")
while True:
    try:
        b = int(ask("   Enter 1, 2, or 3: "))
        if b in [1, 2, 3]:
            user_budget = ["Budget", "Mid-range", "Luxury"][b - 1]
            break
        print("   ↳ Enter 1, 2, or 3")
    except ValueError:
        print("   ↳ Please type a number")

# STEP 3: Region (optional)
print(f"\n🗺️  Regions: {', '.join(known_regions)}")
region_input = ask("Preferred region (press Enter to skip): ").strip().lower().replace(" ", "_")
if region_input and region_input not in known_regions:
    print("   ↳ Region not recognised. Searching all regions.")
chosen_region = region_input if region_input in known_regions else None

# STEP 4: Preferences
print("\n⭐ Rate each preference 1–5  (1=low interest | 5=high interest)\n")
pref_desc = {
    "culture":   "Culture     (history, museums, art)",
    "adventure": "Adventure   (hiking, sports, thrills)",
    "nature":    "Nature      (parks, wildlife, scenery)",
    "beaches":   "Beaches     (sea, sand, water activities)",
    "nightlife": "Nightlife   (bars, clubs, entertainment)",
    "cuisine":   "Cuisine     (local food, restaurants)",
    "wellness":  "Wellness    (spas, yoga, relaxation)",
    "urban":     "Urban       (city life, shopping, buzz)",
    "seclusion": "Seclusion   (peaceful, off-beaten-path)",
}
user_prefs = {}
for key, text in pref_desc.items():
    while True:
        try:
            v = int(ask(f"   {text}: "))
            if 1 <= v <= 5:
                user_prefs[key] = v
                break
            print("   ↳ Enter 1–5")
        except ValueError:
            print("   ↳ Please type a number")

# STEP 5: Ask the API
result = call("POST", "/recommend", {
    "temperature": desired_temp,
    "budget": user_budget,
    "region": chosen_region,
    "preferences": user_prefs,
    "top_n": 5,
})

print(f"\n🤖 Model's budget estimate for your taste : {result['predicted_budget']}")
print(f"📌 Your chosen budget                     : {result['selected_budget']}")
if result["predicted_budget"] != result["selected_budget"]:
    print(f"   ℹ️  Your ratings and temperature lean toward {result['predicted_budget']} destinations.")
for note in result["notes"]:
    print(f"   ℹ️  {note}")

# STEP 6: Show the results
if not result["results"]:
    print("\n😕 No destinations matched your filters. Try another budget or region.")
    sys.exit(0)

medals = ["🥇", "🥈", "🥉", "4️⃣ ", "5️⃣ ", "6.", "7.", "8.", "9.", "10."]
print(f"\n{'═' * 60}")
print(f"  🌍 TOP {len(result['results'])} RECOMMENDED DESTINATIONS")
print(f"  Budget: {user_budget}  |  Preferred temp: {desired_temp}°C")
if result["region_applied"]:
    print(f"  Region: {result['region_applied']}")
print(f"{'═' * 60}\n")

for item in result["results"]:
    hotel = f"~${item['avg_hotel_usd']}/night" if item["avg_hotel_usd"] is not None else ""
    print(f"  {medals[item['rank'] - 1]}  {item['city']}, {item['country']}")
    print(f"       📍 Region      : {item['region']}")
    print(f"       💰 Budget      : {item['budget_level']}  {hotel}")
    print(f"       🌡️  Avg Temp    : {item['avg_temp']:.1f}°C")
    print(f"       📅 Ideal Trip  : {item['ideal_trip_days']}  ({item['ideal_trip_labels']})")
    print(f"       🎯 Match Score : {item['match_score']:.1f}%  "
          f"(ratings {item['preference_match']:.1f}% · temperature {item['temperature_match']:.1f}%)")
    print()
