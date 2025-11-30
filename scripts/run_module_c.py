import os
import json
import numpy as np

from scipy.signal import tukey
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
import h5py

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


# =============================================
# Cargar metadata events.json
# =============================================
def load_events_metadata(json_path="events.json"):
    with open(json_path, "r") as f:
        return json.load(f)


# =============================================
# Cargar archivo blanco HDF5 (MANUAL)
# =============================================
def load_white_strain(event, det):
    fname = f"data/white/{event}_{det}_white.hdf5"
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    with h5py.File(fname, "r") as f:
        data = f["strain"][:]
        fs = float(f["strain"].attrs["fs"])

    # reconstruimos un "pseudo" TimeSeries:
    class Obj:
        pass

    ts = Obj()
    ts.value = data
    ts.fs = fs
    ts.times = np.arange(len(data)) / fs + 0  # GPS relativo fake

    return ts


# =============================================
# Señal modelo para QNM
# =============================================
def damped_sinusoid(t, A, f, tau, phi):
    return A * np.exp(-t/tau) * np.cos(2*np.pi*f*t + phi)


# =============================================
# Estimar parámetros iniciales
# =============================================
def estimate_initial_params(t, h):
    win = tukey(len(h), 0.2)
    h_win = h * win

    dt = t[1] - t[0]
    freqs = rfftfreq(len(h), dt)
    H = np.abs(rfft(h_win))

    # buscar pico 100–3000 Hz
    band = (freqs >= 100) & (freqs <= 3000)
    if not np.any(band):
        f0 = 1500
    else:
        f0 = freqs[band][np.argmax(H[band])]

    env = np.abs(h)
    env = np.where(env <= 1e-24, 1e-24, env)
    mask = env > 0.1*env.max()

    if np.sum(mask) < 10:
        tau0 = 0.01
    else:
        t_sel = t[mask]
        e_sel = env[mask]
        a, b = np.polyfit(t_sel, np.log(e_sel), 1)
        tau0 = -1/b if b < 0 else 0.01

    return float(env.max()), float(f0), float(tau0), 0.0


# =============================================
# Ajuste de QNM
# =============================================
def fit_qnm(ts, gps):
    fs = ts.fs
    t_abs = ts.times
    h = ts.value

    t = t_abs - gps

    # buscar pico más flexible
    def peak_search(t, h, tmin, tmax):
        mask = (t >= tmin) & (t <= tmax)
        if np.sum(mask) < 10:
            return None, None, None
        hh = h[mask]
        tt = t[mask]
        idx = np.argmax(np.abs(hh))
        return tt[idx], tt, hh

    # ampliamos mucho el rango del merger
    for w in [0.05, 0.10, 0.20]:
        t_peak, ts_seg, hs_seg = peak_search(t, h, -w, w)
        if t_peak is not None:
            break

    if t_peak is None:
        raise RuntimeError("No se pudo localizar un pico claro de ringdown.")

    # ventana ringdown 100 ms
    T_fit = 0.10
    mask = (t >= t_peak) & (t <= t_peak + T_fit)
    if np.sum(mask) < 30:
        raise RuntimeError("Ringdown demasiado corto.")

    t_fit = t[mask]
    h_fit = h[mask]

    # t relativo al inicio
    t_fit = t_fit - t_fit[0]

    # iniciales
    A0, f0, tau0, phi0 = estimate_initial_params(t_fit, h_fit)

    p0 = [A0, f0, tau0, phi0]
    bounds = (
        [0, 100, 1e-4, -2*np.pi],
        [A0*20, 5000, 1.0, 2*np.pi]
    )

    popt, _ = curve_fit(
        damped_sinusoid, t_fit, h_fit,
        p0=p0, bounds=bounds, maxfev=20000
    )

    A, f, tau, phi = popt
    model = damped_sinusoid(t_fit, *popt)
    resid = h_fit - model
    chi2 = float(np.mean(resid**2))

    return f, tau, chi2


# =============================================
# MAIN
# =============================================
def main():
    meta = load_events_metadata()

    print("=== MÓDULO C – AJUSTE QNM DETALLADO (versión B PRO) ===\n")
    print("EVENTO     DET   f_QNM[Hz]   tau[s]     chi2     OK? MENSAJE")
    print("-------------------------------------------------------------------")

    results = []

    for ev in EVENTS:
        gps = float(meta[ev]["gps"])
        print(f"\n>>> EVENTO: {ev} (GPS = {gps:.3f})")

        for det in DETECTORS:

            try:
                ts = load_white_strain(ev, det)
                f, tau, chi2 = fit_qnm(ts, gps)
                ok = True
                msg = "OK"
            except Exception as e:
                f = tau = chi2 = np.nan
                ok = False
                msg = str(e)

            print(f"{ev:9s} {det:3s} "
                  f"{(f if ok else 'nan'):>10} "
                  f"{(tau if ok else 'nan'):>10} "
                  f"{(chi2 if ok else 'nan'):>10} "
                  f"{'YES' if ok else 'NO '} {msg}")

            results.append({
                "event": ev,
                "det": det,
                "f_qnm": float(f) if ok else None,
                "tau": float(tau) if ok else None,
                "chi2": float(chi2) if ok else None,
                "ok": ok,
                "message": msg
            })

    # guardar JSON
    os.makedirs("results", exist_ok=True)
    with open("results/qnm_moduleC_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== MÓDULO C – COMPLETADO ===")


if __name__ == "__main__":
    main()
