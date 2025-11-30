import os
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import welch

# ============================================================
# CONFIG
# ============================================================

OUT_RAW   = "data/raw"
OUT_CLEAN = "data/clean"
OUT_WHITE = "data/white"

os.makedirs(OUT_RAW, exist_ok=True)
os.makedirs(OUT_CLEAN, exist_ok=True)
os.makedirs(OUT_WHITE, exist_ok=True)


# ============================================================
# DESCARGA
# ============================================================

def download_strain(event: str, det: str, gps_start: float, gps_end: float):
    """
    Descarga datos reales desde GWOSC usando gwpy.fetch_open_data
    """
    try:
        ts = TimeSeries.fetch_open_data(det, gps_start, gps_end, verbose=False)
        return ts
    except Exception as e:
        print(f"✖ Error descargando {event}/{det}: {e}")
        return None


# ============================================================
# LIMPIEZA
# ============================================================

def clean_strain(ts: TimeSeries):
    """
    Detrend + Bandpass 20–1024 Hz
    """
    try:
        ts2 = ts.detrend("linear")
        ts3 = ts2.bandpass(20, 1024)
        return ts3
    except Exception as e:
        print("✖ Error en limpieza:", e)
        return None


# ============================================================
# BLANQUEO
# ============================================================

def whiten_manual(ts: TimeSeries):
    """
    Blanqueo manual usando Welch + FFT (compatible con SciPy actual)
    """
    try:
        fs = ts.sample_rate.value

        # PSD por Welch
        freqs, psd = welch(ts.value, fs=fs, nperseg=4*fs)

        # FFT
        ft = np.fft.rfft(ts.value)
        ft_white = ft / np.sqrt(psd + 1e-12)

        white = np.fft.irfft(ft_white, n=len(ts.value))

        return TimeSeries(white, sample_rate=ts.sample_rate, t0=ts.t0)
    except Exception as e:
        print("✖ Error blanqueando:", e)
        return None


# ============================================================
# GUARDA (con overwrite=True)
# ============================================================

def save_series(ts: TimeSeries, path: str):
    """
    Guarda un TimeSeries en formato HDF5, permitiendo overwrite.
    """
    try:
        ts.write(path, format="hdf5", path="strain", overwrite=True)
    except Exception as e:
        print(f"✖ Error guardando {path}: {e}")


# ============================================================
# PIPE DE PROCESAMIENTO
# ============================================================

def process_event(event: str, det: str, gps: float):
    gps_start = int(gps) - 4
    gps_end   = int(gps) + 4

    print(f"  GPS = {gps}")
    print(f"  Ventana = [{gps_start}, {gps_end}]")

    # Descargar
    ts_raw = download_strain(event, det, gps_start, gps_end)
    if ts_raw is None:
        return

    # Limpiar
    ts_clean = clean_strain(ts_raw)
    if ts_clean is None:
        return

    # Blanquear
    ts_white = whiten_manual(ts_clean)
    if ts_white is None:
        return

    # === RUTAS DE SALIDA ===
    raw_path   = f"{OUT_RAW}/{event}_{det}_raw.hdf5"
    clean_path = f"{OUT_CLEAN}/{event}_{det}_clean.hdf5"
    white_path = f"{OUT_WHITE}/{event}_{det}_white.hdf5"

    print(f"  Guardando RAW   → {raw_path}")
    save_series(ts_raw, raw_path)

    print(f"  Guardando CLEAN → {clean_path}")
    save_series(ts_clean, clean_path)

    print(f"  Guardando WHITE → {white_path}")
    save_series(ts_white, white_path)

    print("  ✓ COMPLETADO\n")
