import os
import json
from scripts.fit_edr_full import fit_edr_full
from scripts.compare_gr_edr import compare_GR_EDR

EVENTS = {
    "GW150914":  (68, 0.67),
    "GW151226":  (20.5, 0.74),
    "GW170104":  (49, 0.66),
    "GW170608":  (19, 0.74),
    "GW170814":  (54.5, 0.74),
    "GW170729":  (80, 0.81),
    "GW170823":  (60, 0.72),
    "GW190412":  (34, 0.67),
    "GW190521":  (142, 0.72),
    "GW190814":  (25, 0.91),
}

DETECTORS = ["H1", "L1"]

OUTDIR = "results/edr_full_pipeline"
os.makedirs(OUTDIR, exist_ok=True)

def run_full(det, event, Mrem, chi):
    print(f"\n===== {event} — {det} =====")

    # 1. Fit FULL EDR
    edr_params, h_best, resid = fit_edr_full(det, event, Mrem, chi)

    # 2. Comparación GR vs EDR (likelihoods)
    cmp = compare_GR_EDR(det, event, Mrem, chi)

    # 3. Guardar todo
    out = {
        "event": event,
        "detector": det,
        "edr_full_params": {
            "A22": edr_params[0],
            "A33": edr_params[1],
            "A21": edr_params[2],
            "d_om22": edr_params[3],
            "d_tau22": edr_params[4],
            "d_om33": edr_params[5],
            "d_tau33": edr_params[6],
            "d_om21": edr_params[7],
            "d_tau21": edr_params[8],
            "phi22": edr_params[9],
            "phi33": edr_params[10],
            "phi21": edr_params[11],
            "t0": edr_params[12],
        },
        "gr_vs_edr": cmp
    }

    fout = f"{OUTDIR}/{event}_{det}.json"
    with open(fout, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✔ Guardado: {fout}")


if __name__ == "__main__":
    for ev, (Mrem, chi) in EVENTS.items():
        for det in DETECTORS:
            try:
                run_full(det, ev, Mrem, chi)
            except Exception as e:
                print(f"❌ Error en {ev} {det}: {e}")
