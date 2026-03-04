import json
import os
from datetime import datetime
import requests
import re

API_URL = "http://127.0.0.1:8000/api/query/"
MODE="answer"
FORMAT_TYPE="html"

TEST_QUERIES = {
    # "short_mel": "MEL",
    # "short_amp": "AMP",
    # "short_camo": "Part-CAMO",

    # "medium_deferred_defects": "requirements for deferred defects under MEL",
    # "medium_arc": "who can issue an ARC",
    "medium_camo_resp": "continuing airworthiness management responsibilities",

    # "long_scenario_mel": (
    #     "Under what conditions may an operator defer a defect under the MEL "
    #     "when the aircraft is away from base and no certifying staff are available?"
    # ),
    # "long_scenario_transition": (
    #     "Explain the responsibilities of the CAMO when an aircraft transitions "
    #     "between operators and the maintenance programme changes."
    # ),

    # "edge_empty": "",
    # "edge_nonsense": "asdfghjk",
    # "edge_french": "Quelles sont les responsabilités du CAMO?"
}

def slugify(text):
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")

def save_result(name, query, response_json):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = slugify(name)
    path = f"tests/results/{timestamp}_{slug}.json"

    os.makedirs("tests/results", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "query": query,
            "response": response_json
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved: {path}")

def run_tests():
    for name, query in TEST_QUERIES.items():
        print(f"\n=== Running test: {name} ===")
        payload = {"query": query,
                   "mode": MODE,
                   "format": FORMAT_TYPE}

        r = requests.post(API_URL, json=payload)
        print(f"Status: {r.status_code}")

        try:
            data = r.json()
        except Exception:
            print("Invalid JSON response:")
            print(r.text)
            continue

        save_result(name, query, data)

if __name__ == "__main__":
    run_tests()