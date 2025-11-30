# scripts/download_data.py
#
# Descarga y preprocesa strain real de GWOSC usando GWpy,
# pero hace el blanqueo (whitening) a mano con numpy + scipy
# para no depender de gwpy.whiten (que rompe por 'hanning').

import os
from typing import Tuple

import numpy as np
from scipy.signal import welch

from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

# Carpetas de salida ---------------------------------------------------------

BASE_DIR = os.path.join("data")
RAW_DIR = os.path.join(BASE_DIR, "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "clean")
WHITE_DIR = os.path.join(BASE_DIR, "white")

for d in (BASE_DIR, RAW_DIR, CLEAN_DIR, WHITE_DIR):
    os.makedirs(d, exist_ok=True)


# ==============================
#  GPS del evento (GWOSC)
# ==============================
def get_event_gps(event: str) -> float:
    gps = event_gps(event)
    if isinstance(gps, (list, tuple)):
        gps = gps[0]
    return float(gps)


# ==============================
#  Whitening manual
# ==============================
def manual_whiten(ts: TimeSeries, seglen: float = 4.0) -> TimeSeries:
    """
    Whitening con FFT + PSD tipo Welch, sin usar ts.whiten().
    """
    data = ts.value
    dt = ts.dt.value
    fs = 1.0 / dt

    # PSD con Welch
    nperseg = int(seglen * fs)
    if nperseg > len(data):
        nperseg = len(data) // 2
    freqs_psd, psd = welch(data, fs=fs, nperseg=nperseg)

    # FFT de la señal
    freqs_fft = np.fft.rfftfreq(len(data), dt)
    fft_data = np.fft.rfft(data)

    # Interpolar PSD a las frecuencias de la FFT
    psd_interp = np.interp(freqs_fft, freqs_psd, psd)
    psd_interp = np.where(psd_interp <= 0, np.inf, psd_interp)

    # Whitening en frecuencia
    fft_white = fft_data / np.sqrt(psd_interp)

    # Volver al tiempo
    white_data = np.fft.irfft(fft_white, n=len(data))

    # Crear nuevo TimeSeries blanqueado
    ts_white = TimeSeries(
        white_data,
        sample_rate=ts.sample_rate,
        t0=ts.t0,
    )
    return ts_white


# ==============================
#  Descarga + preproceso
# ==============================
def download_and_preprocess(
    event: str,
    det: str,
    window: float = 8.0,
    pad: float = 8.0,
    f_low: float = 20.0,
    f_high: float = 1024.0,
) -> Tuple[str, str, str]:
    """
    Descarga y procesa strain para un evento y detector.

    Pasos:
      1. GPS del evento vía GWOSC.
      2. Descarga strain real con TimeSeries.fetch_open_data.
      3. Recorta ventana de interés.
      4. Limpia (detrend + bandpass).
      5. Blanquea con manual_whiten().
      6. Guarda RAW, CLEAN y WHITE en HDF5.
    """
    gps = get_event_gps(event)
    t0 = int(gps) - int(window // 2)
    t1 = t0 + int(window)

    print(f"  GPS = {gps:.3f}")
    print(f"  Ventana de análisis: [{t0}, {t1}] s")

    # 1) Descargar datos reales
    print("  Descargando datos con TimeSeries.fetch_open_data(...)")
    ts_full = TimeSeries.fetch_open_data(det, t0 - int(pad), t1 + int(pad), cache=True)

    # 2) Recorte exacto
    ts_raw = ts_full.crop(t0, t1)

    # 3) Limpieza
    print(f"  Limpieza: detrend + bandpass [{f_low}, {f_high}] Hz")
    ts_clean = ts_raw.detrend("constant").bandpass(f_low, f_high)

    # 4) Whitening manual
    print("  Blanqueando con método manual (Welch + FFT)")
    ts_white = manual_whiten(ts_clean, seglen=4.0)

    # 5) Guardar en disco
    base = f"{event}_{det}"
    raw_path = os.path.join(RAW_DIR, f"{base}_raw.hdf5")
    clean_path = os.path.join(CLEAN_DIR, f"{base}_clean.hdf5")
    white_path = os.path.join(WHITE_DIR, f"{base}_white.hdf5")

    print(f"  Guardando RAW   -> {raw_path}")
    ts_raw.write(raw_path, format="hdf5")

    print(f"  Guardando CLEAN -> {clean_path}")
    ts_clean.write(clean_path, format="hdf5")

    print(f"  Guardando WHITE -> {white_path}")
    ts_white.write(white_path, format="hdf5")

    return raw_path, clean_path, white_path
