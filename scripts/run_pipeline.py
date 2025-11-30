import json
from scripts.download_data import download_and_preprocess

EVENTS = [
    "GW150914", "GW151226", "GW170104",
    "GW170608", "GW170814", "GW170729",
    "GW170823", "GW190412", "GW190521",
    "GW190814"
]

DETECTORS = ["H1", "L1"]

if __name__ == "__main__":
    print("=== PIPELINE GWOSC + GWpy ===\n")

    with open("events.json") as f:
        meta = json.load(f)

    for ev in EVENTS:
        gps = float(meta[ev]["gps"])
        print(f"\n>>> EVENTO: {ev}")
        for det in DETECTORS:
            download_and_preprocess(ev, det, gps)

    print("\n=== PIPELINE COMPLETO ===")
