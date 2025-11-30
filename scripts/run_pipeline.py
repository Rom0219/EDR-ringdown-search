import os
from scripts.download_data import process_event

# ============================================
# EVENTOS (GPS oficiales GWOSC)
# ============================================

EVENTS = {
    "GW150914": 1126259462.4,
    "GW151226": 1135136350.6,
    "GW170104": 1167559936.6,
    "GW170608": 1180922494.5,
    "GW170814": 1186741861.5,
    "GW170729": 1185389807.3,
    "GW170823": 1187529256.5,
    "GW190412": 1239082262.1,
    "GW190521": 1242442967.4,
    "GW190814": 1249852257.0,
}

DETECTORS = ["H1", "L1"]


# ============================================
# MAIN PIPELINE
# ============================================

def run():
    print("=== PIPELINE GWOSC + GWpy (fetch_open_data) ===\n")

    for event, gps in EVENTS.items():
        print(f">>> EVENTO: {event}\n")

        for det in DETECTORS:
            print(f"===== {event} — {det} =====")
            process_event(event, det, gps)

    print("\n=== PIPELINE COMPLETO ===")


# ============================================

if __name__ == "__main__":
    run()
