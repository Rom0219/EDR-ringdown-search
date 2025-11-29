"""
scripts/compare_gr_edr.py

Comparación estadística GR vs EDR usando:
 - Likelihood GR
 - Likelihood EDR
 - AIC (Akaike)
 - BIC (Bayesian Information Criterion)
 - Likelihood Ratio Test
 - Bayes Factor aproximado (via BIC)

Este módulo resume en un solo archivo cuál teoría explica mejor
los datos del ringdown para un evento y detector específicos.

Requiere que existan:
  - fit_gr.py
  - fit_edr.py
  - model_gr.py
  - model_edr.py
  - datos whitened: data/processed/*

Devuelve un diccionario y guarda un archivo .txt con los resultados.
"""

import os
import numpy as np
from gwpy.timeseries import TimeSeries
from scripts.fit_gr import fit_gr_mode22
from scripts.fit_edr import fit_edr_mode22
from scripts.model_gr import freq_tau

RESULTS_DIR = "comparison_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Helper: cargar dato procesado
# ============================================================
def load_processed(det, event):
    fname = os.path.join("data/processed", f"{event}_{det}_processed.hdf5")
    ts = TimeSeries.read(fname)
    return ts.value, ts.times.value


# ============================================================
# Calcular log-likelihood para una señal reconstruida
# ============================================================
def logL(data, template):
    resid = data - template
    return -0.5 * np.sum(resid * resid)


# ============================================================
# Calcular AIC y BIC
# ============================================================
def aic(logL_val, k):
    return 2*k - 2*logL_val

def bic(logL_val, k, n):
    return k*np.log(n) - 2*logL_val


# ============================================================
# Proceso completo de comparación
# ============================================================
def compare_GR_EDR(det, event, Mrem, chi):

    print("\n==============================")
    print(f"COMPARANDO GR vs EDR — {event} — {det}")
    print("==============================")

    # ========= 1) cargar datos =========
    data, t = load_processed(det, event)
    n = len(data)

    # ========= 2) Ajuste GR =========
    gr_params, gr_template, gr_resid = fit_gr_mode22(det, event, Mrem, chi)

    ll_gr = logL(data, gr_template)
    k_gr = 5  # A, f0, tau, phi, t0

    aic_gr = aic(ll_gr, k_gr)
    bic_gr = bic(ll_gr, k_gr, n)

    print("\n--- RESULTADOS GR ---")
    print(f"logL_GR = {ll_gr:.3f}")
    print(f"AIC_GR  = {aic_gr:.3f}")
    print(f"BIC_GR  = {bic_gr:.3f}")


    # ========= 3) Ajuste EDR =========
    edr_params, edr_template, edr_resid = fit_edr_mode22(det, event, Mrem, chi)

    ll_edr = logL(data, edr_template)
    k_edr = 5  # A, δω/ω, δτ/τ, phi, t0

    aic_edr = aic(ll_edr, k_edr)
    bic_edr = bic(ll_edr, k_edr, n)

    print("\n--- RESULTADOS EDR ---")
    print(f"logL_EDR = {ll_edr:.3f}")
    print(f"AIC_EDR  = {aic_edr:.3f}")
    print(f"BIC_EDR  = {bic_edr:.3f}")


    # ========= 4) Likelihood Ratio Test =========
    LRT = 2 * (ll_edr - ll_gr)
    print(f"\nLikelihood Ratio Test (EDR - GR) = {LRT:.3f}")


    # ========= 5) Bayes Factor (aprox via BIC) =========
    #    BF ≈ exp((BIC_GR - BIC_EDR)/2)
    dBIC = bic_gr - bic_edr
    BayesFactor = np.exp(dBIC / 2)

    print(f"ΔBIC = BIC_GR - BIC_EDR = {dBIC:.3f}")
    print(f"Bayes Factor (aprox) = {BayesFactor:.3f}")

    if BayesFactor > 1:
        favored = "EDR preferido"
    else:
        favored = "GR preferido"

    print(f"\n>>> MODELO FAVORECIDO: {favored} <<<")


    # ========= 6) Guardar en archivo =========
    outname = os.path.join(RESULTS_DIR, f"{event}_{det}_comparison.txt")
    with open(outname, "w") as f:
        f.write("COMPARACIÓN GR vs EDR\n")
        f.write(f"Evento: {event}\nDetector: {det}\n\n")

        f.write("--- GR ---\n")
        f.write(f"logL_GR = {ll_gr:.6f}\n")
        f.write(f"AIC_GR  = {aic_gr:.6f}\n")
        f.write(f"BIC_GR  = {bic_gr:.6f}\n\n")

        f.write("--- EDR ---\n")
        f.write(f"logL_EDR = {ll_edr:.6f}\n")
        f.write(f"AIC_EDR  = {aic_edr:.6f}\n")
        f.write(f"BIC_EDR  = {bic_edr:.6f}\n\n")

        f.write(f"Likelihood Ratio = {LRT:.6f}\n")
        f.write(f"ΔBIC = {dBIC:.6f}\n")
        f.write(f"Bayes Factor = {BayesFactor:.6f}\n\n")
        f.write(f"Modelo favorecido: {favored}\n")

    print(f"\n✔ Resultados guardados en: {outname}")

    return {
        "logL_GR": ll_gr,
        "logL_EDR": ll_edr,
        "AIC_GR": aic_gr,
        "AIC_EDR": aic_edr,
        "BIC_GR": bic_gr,
        "BIC_EDR": bic_edr,
        "ΔBIC": dBIC,
        "BayesFactor": BayesFactor,
        "favored": favored
    }
