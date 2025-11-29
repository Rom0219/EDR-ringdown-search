"""
scripts/fit_edr.py

Ajuste EDR:
 - Carga de datos whitened
 - Plantilla EDR (modo 22 por defecto)
 - Likelihood Gaussiana
 - Recuperación de parámetros:
        A22, δω/ω, δτ/τ, phi, t0
 - Gráficos de:
        - ajuste vs dato
        - residuo
        - espectro del residuo

Este compara cómo de bien la teoría EDR puede reproducir el ringdown real.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from gwpy.timeseries import TimeSeries

from scripts.model_gr import freq_tau
from scripts.model_edr import edr_damped_sine

DATA_DIR = "data/processed"
PLOT_DIR = "plots/fit_edr"
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# Cargar dato whitened
# ============================================================
def load_processed(det, event):
    fname = os.path.join(DATA_DIR, f"{event}_{det}_processed.hdf5")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"No existe archivo procesado: {fname}")
    ts = TimeSeries.read(fname)
    return ts.value, ts.times.value


# ============================================================
# Likelihood EDR
# ============================================================
def neg_log_like_edr(params, data, t, fs, f0_gr, tau_gr):
    """
    params = [A22, delta_omega_ratio, delta_tau_ratio, phi, t0]
    delta_omega_ratio = δω/ω
    delta_tau_ratio   = δτ/τ
    """
    A22, d_om, d_tau, phi, t0 = params

    template = edr_damped_sine(
        t, A22, f0_gr, tau_gr, phi, t0,
        delta_omega_ratio=d_om,
        delta_tau_ratio=d_tau
    )

    resid = data - template
    return 0.5 * np.sum(resid * resid)


# ============================================================
# Ajuste principal EDR modo 22
# ============================================================
def fit_edr_mode22(det, event, Mrem, chi, fs_override=None):

    print(f"\n===== Ajustando EDR (22) para {event} — {det} =====")

    # cargar dato procesado
    data, t = load_processed(det, event)
    dt = t[1] - t[0]
    fs = 1.0 / dt if fs_override is None else fs_override

    # obtener GR como base
    f0_gr, tau_gr = freq_tau(Mrem, chi, "22")

    # iniciales
    params0 = [
        1.0,        # A22
        0.0,        # delta_omega_ratio
        0.0,        # delta_tau_ratio
        0.0,        # phi
        0.01        # t0
    ]

    bounds = [
        (0, None),       # amplitud
        (-0.5, 0.5),     # delta_omega (±50%)
        (-0.5, 0.5),     # delta_tau   (±50%)
        (-2*np.pi, 2*np.pi),
        (0.0, 0.05)      # t0 entre 0 y 50 ms
    ]

    res = minimize(
        neg_log_like_edr,
        params0,
        args=(data, t, fs, f0_gr, tau_gr),
        bounds=bounds,
        method="L-BFGS-B"
    )

    A22, d_om, d_tau, phi, t0 = res.x

    print("\n✔ Parámetros recuperados (EDR 22):")
    print(f"A22           = {A22:.4f}")
    print(f"δω/ω          = {d_om:.5f}")
    print(f"δτ/τ          = {d_tau:.5f}")
    print(f"phi           = {phi:.3f}")
    print(f"t0            = {t0*1000:.2f} ms")

    # plantilla final
    template = edr_damped_sine(
        t, A22, f0_gr, tau_gr, phi, t0,
        delta_omega_ratio=d_om,
        delta_tau_ratio=d_tau
    )

    resid = data - template

    # ========================================================
    # Gráficos
    # ========================================================
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(t, data, label="Datos whitened")
    ax[0].plot(t, template, label="Plantilla EDR ajustada")
    ax[0].set_title(f"{event} — {det}: Ajuste EDR modo (2,2)")
    ax[0].legend()

    ax[1].plot(t, resid)
    ax[1].set_title("Residuo EDR (dato - EDR)")

    # espectro del residuo
    freqs = np.fft.rfftfreq(len(resid), dt)
    spec = np.abs(np.fft.rfft(resid))

    ax[2].plot(freqs, spec)
    ax[2].set_xlim(0, 800)
    ax[2].set_title("Espectro del residuo EDR")

    out = os.path.join(PLOT_DIR, f"{event}_{det}_fit_edr.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    print(f"✔ Figura guardada en: {out}")

    return res.x, template, resid
