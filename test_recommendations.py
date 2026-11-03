# Checks that the NexVoy API really reacts to the user's inputs.
# Usage:
#   python test_recommendations.py                         (tests http://127.0.0.1:7860)
#   python test_recommendations.py http://127.0.0.1:7860 [api key]
#
# Two kinds of result:
#   PASS / FAIL  : something that must be true for ANY dataset (the logic of the API).
#   NOTE         : something that is true for most data but is not guaranteed by the dataset.
#                  A NOTE never fails the run. It tells you what to look at.
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:7860"
KEY = sys.argv[2] if len(sys.argv) > 2 else ""

EXAMPLE = dict(culture=4, adventure=3, nature=5, beaches=2, nightlife=5, cuisine=5, wellness=4, urban=4, seclusion=5)
BEACH = dict(culture=2, adventure=3, nature=2, beaches=5, nightlife=4, cuisine=4, wellness=3, urban=3, seclusion=1)
MOUNTAIN = dict(culture=3, adventure=5, nature=5, beaches=1, nightlife=1, cuisine=3, wellness=3, urban=1, seclusion=5)
CITY = dict(culture=5, adventure=1, nature=2, beaches=1, nightlife=3, cuisine=5, wellness=2, urban=5, seclusion=1)
PREFS = list(EXAMPLE)
BUDGETS = ("Budget", "Mid-range", "Luxury")

failures = []


def post(body, path="/recommend"):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if KEY:
        req.add_header("X-API-Key", KEY)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return json.loads(r.read())


def rec(temp, budget, prefs, region=None, top_n=5):
    status, out = post({"temperature": temp, "budget": budget, "region": region,
                        "preferences": prefs, "top_n": top_n})
    assert status == 200, (status, out)
    return out


def names(out):
    return [r["city"] for r in out["results"]]


def mean_of(out, key):
    return sum(r[key] for r in out["results"]) / len(out["results"])


def check(label, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}  {detail}")
    if not condition:
        failures.append(label)


def note(label, condition, detail=""):
    print(f"  {'ok  ' if condition else 'NOTE'}  {label}  {detail}")


health = get("/health")
print(f"API: {BASE}  | destinations: {health['destinations']}  | model: {health['model']}  "
      f"| temperature method: {health['temperature_method']}")
regions = get("/model-info")["known_regions"]

print("\n1. Temperature changes the ranking (same ratings, same budget, all regions)")
means = {}
for t in (-5, 15, 28):
    out = rec(t, "Mid-range", EXAMPLE)
    means[t] = mean_of(out, "avg_temp")
    print(f"     {t:>3} C -> {names(out)}  average temp of results: {means[t]:.1f} C")
check("Top lists for -5 C and 28 C are different", names(rec(-5, "Mid-range", EXAMPLE)) != names(rec(28, "Mid-range", EXAMPLE)))
note("Warmer wish gives warmer results (-5 < 15 < 28)", means[-5] < means[15] < means[28],
     "(depends on which cities exist in the dataset, so it is a NOTE, not a failure)")

print("\n2. Each wanted temperature scores a city differently")
a = rec(17, "Mid-range", EXAMPLE, top_n=10)
b = rec(17.5, "Mid-range", EXAMPLE, top_n=10)
first = {r["city"]: r["temperature_match"] for r in a["results"]}
common = [(r["city"], first[r["city"]], r["temperature_match"]) for r in b["results"] if r["city"] in first]
print(f"     (city, temperature_match at 17 C, at 17.5 C): {common}")
check("At least one city appears in both lists", len(common) > 0)
check("Its temperature score changes when the wish changes", all(x != y for _, x, y in common))

print("\n3. Ratings change the ranking (same temperature, same budget)")
tops = {}
for label, prefs in (("beach lover", BEACH), ("mountain lover", MOUNTAIN), ("city lover", CITY)):
    tops[label] = names(rec(22, "Mid-range", prefs))
    print(f"     {label:15s} -> {tops[label]}")
check("The three rating profiles give three different top lists", len({tuple(v) for v in tops.values()}) == 3)

print("\n4. Every one of the nine ratings affects the scores")
for p in PREFS:
    low = [r["match_score"] for r in rec(22, "Mid-range", dict(EXAMPLE, **{p: 1}), top_n=10)["results"]]
    high = [r["match_score"] for r in rec(22, "Mid-range", dict(EXAMPLE, **{p: 5}), top_n=10)["results"]]
    check(f"Changing '{p}' from 1 to 5 changes the scores", low != high)

print("\n5. Budget filter")
for budget in BUDGETS:
    out = rec(22, budget, EXAMPLE)
    levels = {r["budget_level"] for r in out["results"]}
    print(f"     {budget:10s} -> {names(out)}")
    check(f"All {budget} results are {budget} destinations", levels == {budget} and out["selected_budget"] == budget)
    note(f"Five {budget} destinations were found", out["count"] == 5, f"(got {out['count']})")

print("\n6. Region filter (every region x every budget)")
for region in regions:
    for budget in BUDGETS:
        out = rec(22, budget, EXAMPLE, region=region, top_n=10)
        shown = region.replace("_", " ").title()
        got = {r["region"] for r in out["results"]}
        if out["region_applied"]:
            ok = got == {shown} and not out["notes"]
            check(f"{region:14s} {budget:9s} region applied, only {shown} results", ok, f"({out['count']} results)")
        else:
            ok = bool(out["notes"]) and out["count"] > 0
            check(f"{region:14s} {budget:9s} too few matches, note + all regions shown", ok, f"({out['notes']})")

print("\n7. Scores are real similarities")
out = rec(22, "Mid-range", EXAMPLE)
scores = [r["match_score"] for r in out["results"]]
print(f"     scores: {scores}")
check("Scores are sorted from high to low", scores == sorted(scores, reverse=True))
check("Scores are between 0 and 100", all(0 <= s <= 100 for s in scores))
for r in out["results"]:
    expected = 0.65 * r["preference_match"] + 0.35 * r["temperature_match"]
    check(f"{r['city']}: score = 65% ratings + 35% temperature", abs(expected - r["match_score"]) < 0.2)
note("Top score is not forced to a fixed high value", scores[0] < 99, f"(top score {scores[0]})")

print("\n8. Bad input is rejected")
good = {"temperature": 20, "budget": "Mid-range", "region": None, "preferences": EXAMPLE}
for label, patch in (("unknown budget", {"budget": "Cheap"}), ("unknown region", {"region": "atlantis"}),
                     ("rating of 9", {"preferences": dict(EXAMPLE, culture=9)}),
                     ("rating of 0", {"preferences": dict(EXAMPLE, culture=0)}),
                     ("missing rating", {"preferences": {k: v for k, v in EXAMPLE.items() if k != "urban"}}),
                     ("temperature 99", {"temperature": 99}),
                     ("top_n 0", {"top_n": 0}), ("top_n 11", {"top_n": 11})):
    status, _ = post({**good, **patch})
    check(f"{label} returns 422", status == 422, f"(got {status})")

print("\n9. The trained model's budget estimate")
out = rec(22, "Mid-range", EXAMPLE)
print(f"     predicted: {out['predicted_budget']}  probabilities: {out['predicted_budget_probabilities']}")
check("Predicted budget is a valid class", out["predicted_budget"] in BUDGETS)
check("Probabilities add up to 1", abs(sum(out["predicted_budget_probabilities"].values()) - 1) < 0.01)
lux = rec(22, "Mid-range", dict(culture=5, adventure=2, nature=3, beaches=5, nightlife=4, cuisine=5, wellness=5, urban=4, seclusion=4))
check("Model estimate depends on the inputs", lux["predicted_budget_probabilities"] != out["predicted_budget_probabilities"])
cb = [r["budget_confidence"] for r in out["results"]]
check("budget_confidence is a percentage", all(0 <= c <= 100 for c in cb))

print()
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}")
sys.exit(1 if failures else 0)
