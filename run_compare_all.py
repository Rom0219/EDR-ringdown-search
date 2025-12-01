from scripts.compare_gr_edr import compare_GR_EDR
import json

events_params = {
    "GW150914":  {"Mrem": 68,  "chi": 0.67},
    "GW151226":  {"Mrem": 20.5,"chi": 0.74},
    "GW170104":  {"Mrem": 49,  "chi": 0.66},
    "GW170608":  {"Mrem": 19,  "chi": 0.74},
    "GW170814":  {"Mrem": 54.5,"chi": 0.74},
    "GW170729":  {"Mrem": 80,  "chi": 0.81},
    "GW170823":  {"Mrem": 60,  "chi": 0.72},
    "GW190412":  {"Mrem": 34,  "chi": 0.67},
    "GW190521":  {"Mrem": 142, "chi": 0.72},
    "GW190814":  {"Mrem": 25,  "chi": 0.91}
}

results_all = {}

for ev, pars in events_params.items():
    for det in ["H1", "L1"]:
        print(f"\n>>> Ejecutando {ev} — {det}")
        try:
            out = compare_GR_EDR(det, ev, pars["Mrem"], pars["chi"])
            results_all[f"{ev}_{det}"] = out
        except Exception as e:
            print(f"ERROR en {ev} {det}: {e}")

with open("comparison_results/all_results.json", "w") as f:
    json.dump(results_all, f, indent=2)

print("\n✔ Archivo guardado: comparison_results/all_results.json")
