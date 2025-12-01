# analyze_edr_vs_gr.py (esquema lógico)
import json
import numpy as np

from gr_qnm_fits import f220_GR, tau220_GR   # lo creas luego

with open("results/qnm_moduleC_summary.json") as f:
    res = json.load(f)

with open("events.json") as f:
    meta = json.load(f)   # aquí tendrías M_f, a_f por evento

rows = []
for r in res:
    if not r["ok"]:
        continue

    ev = r["event"]
    f_obs = r["f_qnm"]
    tau_obs = r["tau"]

    M_f = meta[ev]["M_final"]
    a_f = meta[ev]["a_final"]

    f_gr = f220_GR(M_f, a_f)
    tau_gr = tau220_GR(M_f, a_f)

    alpha_flow = (f_obs - f_gr) / f_gr
    beta_flow  = (tau_obs - tau_gr) / tau_gr

    rows.append((ev, r["det"], f_obs, tau_obs, f_gr, tau_gr, alpha_flow, beta_flow))

# luego imprimes tabla o guardas otro JSON
