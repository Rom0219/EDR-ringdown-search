"""
scripts/fit_gr.py

Módulo de ajuste GR:
 - Carga de datos procesados
 - Construcción de plantillas GR (modo 22 o multimodo)
 - Likelihood Gaussiana para datos whitened
 - Alineamiento temporal
 - Optimización (scipy) para encontrar los parámetros
 - Gráficos comparativos y residuales

Dependencias:
 - numpy
 - scipy
 - matplotlib
 - gwpy (solo para lectura del dato)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from gwpy.timeseries import TimeSeries

from scripts.model_gr import gr_multimode_template, freq_tau

DATA_DIR = "data/processed"
PLOT_DIR = "plots/fit_gr"
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# Cargar señal whitened
# ============================================================
def load_processed(det, event):
    fname = os.path.join(DATA_DIR, f"{event}_{det}_processed.hdf5")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"No existe archivo procesado: {fname}")
    ts = TimeSeries.read(fname)
    return ts.value, ts.times.value


# ============================================================
# Likelihood Gaussiana para dato whitened
# ============================================================
def neg_log_likelihood(params, data, t, fs):
    """
    Parámetros:
        params = [A22, f0, tau, phi, t0]
    """
    A22, f0, tau, phi, t0 = params

    # reconstruir señal GR
    template = A22 * np.exp(-(t - t0) / tau) * np.sin(2*np.pi*f0*(t - t0) + phi)
    template[t < t0] = 0.0  # ringdown solo después de t0

    # whitened => covarianza = identidad => L ~ sum((d-h)^2)
    resid = data - template
    return 0.5 * np.sum(resid * resid)


# ============================================================
# Ajuste GR completo (modo 22 por defecto)
# ============================================================
def fit_gr_mode22(det, event, Mrem, chi, fs_override=None):

    print(f"\n===== Ajustando modo GR 22 para {event} — {det} =====")

    # cargar dato procesado
    data, t = load_processed(det, event)
    dt = t[1] - t[0]
    fs = 1.0 / dt if fs_override is None else fs_override

    # iniciales desde teoría (GR)
    f0_th, tau_th = freq_tau(Mrem, chi, mode="22")

    params0 = [
        1.0,       # amplitud inicial
        f0_th,     # frecuencia inicial
        tau_th,    # tau inicial
        0.0,       # fase
        0.01       # t0 inicial (10 ms)
    ]

    bounds = [
        (0, None),         # A22 >= 0
        (f0_th*0.5, f0_th*1.5),
        (tau_th*0.5, tau_th*1.5),
        (-2*np.pi, 2*np.pi),
        (0.0, 0.05)
    ]

    res = minimize(
        neg_log_likelihood,
        params0,
        args=(data, t, fs),
        bounds=bounds,
        method="L-BFGS-B"
    )

    best = res.x
    A22, fbest, taubest, phibest, t0best = best

    print("\n✔ Parámetros recuperados (GR 22):")
    print(f"A22 = {A22:.4f}")
    print(f"f0  = {fbest:.3f} Hz")
    print(f"tau = {taubest:.5f} s")
    print(f"phi = {phibest:.3f}")
    print(f"t0  = {t0best*1000:.2f} ms")

    # construir plantilla final
    template = np.zeros_like(data)
    mask = t >= t0best
    template[mask] = A22 * np.exp(-(t[mask] - t0best)/taubest) * np.sin(
        2*np.pi*fbest*(t[mask]-t0best) + phibest
    )

    # residuo
    resid = data - template

    # ========================================================
    # Gráficos
    # ========================================================
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(t, data, label="Datos whitened")
    ax[0].plot(t, template, label="Plantilla GR ajustada")
    ax[0].set_title(f"{event} — {det}: Ajuste GR modo (2,2)")
    ax[0].legend()

    ax[1].plot(t, resid)
    ax[1].set_title("Residuo (dato - GR)")

    # FFT del residuo
    freqs = np.fft.rfftfreq(len(resid), dt)
    spec = np.abs(np.fft.rfft(resid))

    ax[2].plot(freqs, spec)
    ax[2].set_xlim(0, 800)
    ax[2].set_title("Espectro del residuo")

    out = os.path.join(PLOT_DIR, f"{event}_{det}_fit22.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    print(f"✔ Figura guardada en: {out}")

    return best, template, resid
