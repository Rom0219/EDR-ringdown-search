"""
scripts/run_pipeline.py

Módulo A completo:
 - Descarga datos reales de GWOSC
 - Preprocesamiento básico (whitening + highpass)
 - Guardado en data/clean y data/processed
 - Gráficas LIGO-style

Compatible con Codespaces y GWPy moderno.
"""

import os
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import butter, filtfilt

from scripts.download_data import download_event

# ===============================
# Configuración
# ===============================

EVENTS = [
    "GW150914",
    "GW151226",
    "GW170104",
    "GW170608",
    "GW170814",
    "GW170729",
    "GW190412",
    "GW190521",
    "GW190814",
    "GW200129"
]

DETECTORS = ["H1", "L1"]

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
PROC_DIR = "data/processed"
PLOT_DIR = "plots/pipeline"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ===============================
# Funciones de preprocesamiento
# ===============================

def butter_highpass(data, fs, cutoff=30):
    """Filtro pasa-altos básico."""
    nyq = 0.5 * fs
    norm = cutoff / nyq
    b, a = butter(4, norm, btype='highpass')
    return filtfilt(b, a, data)


def whiten(ts):
    """Whitening usando método interno de GWPy."""
    white = ts.whiten(fftlength=4)
    return white


# ===============================
# Pipeline general por evento
# ===============================

def process_event(event, det):
    """
    - Descarga datos
    - Aplica filtro HP
    - Whitening
    - Guarda datos procesados
    """
    print(f"\n===============================")
    print(f"   PROCESANDO {event} — {det}")
    print(f"===============================")

    # ---------- A1: Descargar datos ----------
    raw_path = download_event(event, det, outdir=RAW_DIR)

    if raw_path is None:
        print(f"✖ No hay datos descargados para {event} / {det}")
        return

    # ---------- A2: cargar serie ----------
    try:
        ts = TimeSeries.read(raw_path)
    except Exception as e:
        print(f"✖ Error leyendo {raw_path}")
        print(e)
        return

    fs = ts.sample_rate.value
    data = ts.value
    t = ts.times.value

    # ---------- A3: filtro highpass ----------
    hp = butter_highpass(data, fs, cutoff=30)
    ts_hp = TimeSeries(hp, times=t)

    # Guardar clean
    clean_path = os.path.join(CLEAN_DIR, f"{event}_{det}_clean.hdf5")
    ts_hp.write(clean_path)
    print(f"✔ Guardado archivo CLEAN: {clean_path}")

    # ---------- A4: whitening ----------
    ts_white = whiten(ts_hp)

    # Guardar processed
    proc_path = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")
    ts_white.write(proc_path)
    print(f"✔ Guardado archivo PROCESSED: {proc_path}")

    # ---------- A5: figuras ----------
    fig = ts_white.plot()
    fig.suptitle(f"{event} — {det} (Whitened)")
    fig.savefig(os.path.join(PLOT_DIR, f"{event}_{det}_whitened.png"))
    fig.close()
    print(f"✔ Figura whitened creada.")


# ===============================
# Pipeline principal
# ===============================

def run_pipeline():
    print("\n============================================")
    print("        EJECUTANDO PIPELINE COMPLETO")
    print("============================================\n")

    for event in EVENTS:
        print(f"\n========== EVENTO: {event} ==========\n")
        print("--- (A1, A2, A3, A4, A5) Procesando datos ---")

        for det in DETECTORS:
            process_event(event, det)

        print(f"\n✔ Módulo A completado para {event}")


if __name__ == "__main__":
    run_pipeline()
