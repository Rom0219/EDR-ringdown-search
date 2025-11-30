# scripts/run_module_c.py

import os
import json
import numpy as np

from gwpy.timeseries import TimeSeries
from scipy.signal import tukey
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit

EVENTS = [
    "GW150914",
    "GW151226",
    "GW170104",
    "GW170608",
    "GW170814",
    "GW170729",
    "GW170823",
    "GW190412",
    "GW190521",
    "GW190814",
]

DETECTORS = ["H1", "L1"]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_events_metadata(json_path="events.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def damped_sinusoid(t, A, f, tau, phi):
    """
    h(t) = A exp(-t/tau) cos(2π f t + phi)
    t asumido en segundos, relativo al inicio del ringdown.
    """
    return A * np.exp(-t / tau) * np.cos(2 * np.pi * f * t + phi)


def load_white_strain(event, det):
    """
    Carga el strain blanqueado generado en el Módulo A.
    Intenta leer como HDF5 de gwpy.
    """
    fname = os.path.join("data", "white", f"{event}_{det}_white.hdf5")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"White file not found: {fname}")

    # Gwpy suele auto-detectar el formato
    ts = TimeSeries.read(fname)
    return ts


def estimate_initial_params(t, h):
    """
    Estima f0 y tau iniciales a partir de un segmento de ringdown.
    t: array (s), relativo al inicio del segmento (t=0 en el primer punto).
    h: array strain (whitened).
    """

    # Ventana suave para evitar artefactos
    window = tukey(len(h), alpha=0.2)
    h_win = h * window

    # === Estimar frecuencia con FFT ===
    dt = float(t[1] - t[0])
    freqs = rfftfreq(len(h), dt)
    H = np.abs(rfft(h_win))

    # Solo buscamos entre 100 Hz y 3000 Hz
    fmin, fmax = 100.0, 3000.0
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        f0 = 1500.0
    else:
        idx_peak = np.argmax(H[band])
        f0 = float(freqs[band][idx_peak])

    # === Estimar tau con ajuste exponencial al envolvente ===
    env = np.abs(h)
    # Evitar ceros
    env = np.where(env <= 1e-24, 1e-24, env)

    # Usamos solo la parte donde el envolvente es relativamente grande
    env_max = float(env.max())
    mask_env = env > (0.1 * env_max)
    t_env = t[mask_env]
    env_sel = env[mask_env]

    if len(t_env) < 10:
        tau0 = 0.01
    else:
        log_env = np.log(env_sel)
        # Ajuste lineal: log(env) ~ a + b t -> tau ~ -1/b si b<0
        a, b = np.polyfit(t_env, log_env, 1)
        if b < 0:
            tau0 = float(-1.0 / b)
        else:
            tau0 = 0.01

    # Amplitud inicial
    A0 = float(env_max)
    phi0 = 0.0

    return A0, f0, tau0, phi0


def fit_qnm(ts_white, gps, event, det):
    """
    Ajusta un modo QNM sobre el ringdown usando:
      - búsqueda automática de pico alrededor del merger
      - estimación inicial de f, tau
      - curve_fit para refinar
    """

    # Tiempo absoluto en GPS y strain
    t_abs = ts_white.times.value  # GPS
    h = ts_white.value

    # Tiempo relativo al merger
    t = t_abs - gps

    # 1) Buscamos el pico alrededor de t ~ 0 (merger)
    #    Primero en una ventana pequeña, si no hay suficiente, la ampliamos.
    def find_peak_window(t, h, tmin, tmax):
        mask = (t >= tmin) & (t <= tmax)
        if np.sum(mask) < 10:
            return None, None, None
        h_seg = h[mask]
        t_seg = t[mask]
        idx_peak = np.argmax(np.abs(h_seg))
        return t_seg[idx_peak], t_seg, h_seg

    t_peak, t_seg, h_seg = find_peak_window(t, h, -0.02, 0.02)
    if t_peak is None:
        # Ampliar a [-0.05, 0.05]
        t_peak, t_seg, h_seg = find_peak_window(t, h, -0.05, 0.05)
    if t_peak is None:
        raise RuntimeError("No se pudo localizar un pico claro de ringdown.")

    # 2) Definimos la ventana de ringdown desde el pico hacia adelante
    T_fit = 0.08  # 80 ms de ringdown
    mask_fit = (t >= t_peak) & (t <= t_peak + T_fit)
    if np.sum(mask_fit) < 30:
        raise RuntimeError("Segmento de ringdown demasiado corto para ajuste.")

    t_fit_abs = t[mask_fit]
    h_fit = h[mask_fit]
    # t relativo al inicio del ringdown (para que t=0 sea el pico)
    t_fit = t_fit_abs - t_fit_abs[0]

    # 3) Estimar parámetros iniciales
    A0, f0, tau0, phi0 = estimate_initial_params(t_fit, h_fit)

    # 4) Ajuste no lineal con bounds razonables
    #    Amplitud positiva, tau entre 0.0001 y 1s, frecuencia entre 100 y 5000 Hz
    A_max = max(A0 * 10.0, 1e-6)

    p0 = [A0, f0, tau0, phi0]
    bounds = (
        [0.0, 100.0, 1e-4, -2.0 * np.pi],
        [A_max, 5000.0, 1.0, 2.0 * np.pi],
    )

    try:
        popt, pcov = curve_fit(
            damped_sinusoid,
            t_fit,
            h_fit,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        A_best, f_best, tau_best, phi_best = popt

        # Calculamos un chi2 simple (RMS del residuo)
        model = damped_sinusoid(t_fit, *popt)
        resid = h_fit - model
        chi2 = float(np.mean(resid**2))

        return {
            "event": event,
            "det": det,
            "f_qnm": float(f_best),
            "tau": float(tau_best),
            "A": float(A_best),
            "phi": float(phi_best),
            "t_peak_rel": float(t_peak),
            "chi2": chi2,
            "ok": True,
            "message": "OK",
        }

    except Exception as e:
        return {
            "event": event,
            "det": det,
            "f_qnm": np.nan,
            "tau": np.nan,
            "A": np.nan,
            "phi": np.nan,
            "t_peak_rel": float(t_peak),
            "chi2": np.nan,
            "ok": False,
            "message": f"Error en ajuste: {e}",
        }


def main():
    events_meta = load_events_metadata()
    rows = []

    print("=== MÓDULO C – AJUSTE QNM DETALLADO (versión B PRO) ===\n")
    print(
        "EVENTO     DET   f_QNM [Hz]      tau [s]       chi2        OK?  MENSAJE\n"
        "--------------------------------------------------------------------------"
    )

    for ev in EVENTS:
        gps = float(events_meta[ev]["gps"])
        print(f"\n>>> EVENTO: {ev} (GPS = {gps:.3f})")
        for det in DETECTORS:
            try:
                ts_white = load_white_strain(ev, det)
                result = fit_qnm(ts_white, gps, ev, det)
            except Exception as e:
                result = {
                    "event": ev,
                    "det": det,
                    "f_qnm": np.nan,
                    "tau": np.nan,
                    "A": np.nan,
                    "phi": np.nan,
                    "t_peak_rel": np.nan,
                    "chi2": np.nan,
                    "ok": False,
                    "message": f"Falló antes del ajuste: {e}",
                }

            rows.append(result)

            f_str = (
                f"{result['f_qnm']:10.2f}"
                if np.isfinite(result["f_qnm"])
                else "    nan   "
            )
            tau_str = (
                f"{result['tau']:10.4f}"
                if np.isfinite(result["tau"])
                else "   nan   "
            )
            chi2_str = (
                f"{result['chi2']:10.3e}"
                if np.isfinite(result["chi2"])
                else "   nan    "
            )
            ok_str = "YES" if result["ok"] else " NO"

            print(
                f"{ev:9s} {det:3s} {f_str}  {tau_str}  {chi2_str}  {ok_str}  {result['message']}"
            )

    # Guardar resultados en JSON y CSV
    json_path = os.path.join(RESULTS_DIR, "qnm_moduleC_summary.json")
    csv_path = os.path.join(RESULTS_DIR, "qnm_moduleC_summary.csv")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # CSV sencillo
    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event",
                "det",
                "f_qnm_Hz",
                "tau_s",
                "A",
                "phi_rad",
                "t_peak_rel_s",
                "chi2",
                "ok",
                "message",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["event"],
                    r["det"],
                    r["f_qnm"],
                    r["tau"],
                    r["A"],
                    r["phi"],
                    r["t_peak_rel"],
                    r["chi2"],
                    r["ok"],
                    r["message"],
                ]
            )

    print("\n=== MÓDULO C – COMPLETADO ===")
    print(f"Resumen JSON: {json_path}")
    print(f"Resumen CSV : {csv_path}")


if __name__ == "__main__":
    main()
