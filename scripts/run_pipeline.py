import os
import numpy as np
from gwpy.timeseries import TimeSeries
from gwosc.api import fetch_event_json
from gwosc.locate import get_urls
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
PROC_DIR = "data/processed"
FIG_DIR = "figures"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------
#     A: LISTA DE EVENTOS
# -----------------------------------------
EVENTS = [
    "GW150914", "GW151226", "GW170104", "GW170608", "GW170814",
    "GW170729", "GW190412", "GW190521", "GW190814", "GW200129"
]

DETECTORS = ["H1", "L1", "V1"]

# -----------------------------------------
#     A1: Obtener GPS del evento
# -----------------------------------------
def get_gps(event):
    try:
        info = fetch_event_json(event)
        return float(info["events"][event]["GPS"])
    except:
        print(f"✖ No se pudo obtener GPS de {event}")
        return None

# -----------------------------------------
#     A2: Descargar datos desde GWOSC
# -----------------------------------------
def download_event(event, det):
    gps = get_gps(event)
    if gps is None:
        return None

    print(f"Descargando {event} ({det}) — GPS {gps}")

    try:
        urls = get_urls(gps - 4, gps + 4, detector=det)
        if len(urls) == 0:
            print("✖ No hay URLs disponibles")
            return None

        url = urls[0]
        print(f"Downloading {url}")

        ts = TimeSeries.read(url, format="hdf5.losc")
        raw_path = os.path.join(RAW_DIR, f"{event}_{det}_raw.hdf5")
        ts.name = f"{event}_{det}_raw"
        ts.write(raw_path, path="/")
        print(f"✔ Guardado raw en {raw_path}")

        return raw_path

    except Exception as e:
        print(f"✖ Error descargando {event}/{det}: {e}")
        return None

# -----------------------------------------
#     A3: Highpass
# -----------------------------------------
def butter_highpass(data, fs, cutoff=30):
    nyq = fs / 2
    b, a = butter(4, cutoff / nyq, btype="high", analog=False)
    return filtfilt(b, a, data)

# -----------------------------------------
#     A4: Whitening (estándar)
# -----------------------------------------
def whiten(ts):
    psd = ts.psd(4)
    white = ts / np.sqrt(psd.interpolate(ts.frequencies))
    return white

# -----------------------------------------
#     A5: Figuras
# -----------------------------------------
def plot_figures(event, det, ts_clean, ts_proc):
    # PSD
    plt.figure(figsize=(10,5))
    psd = ts_clean.psd(4)
    psd.plot()
    plt.title(f"PSD — {event} {det}")
    plt.savefig(f"{FIG_DIR}/{event}_{det}_PSD.png")
    plt.close()

    # whitened
    plt.figure(figsize=(10,5))
    ts_proc.plot()
    plt.title(f"Whitened — {event} {det}")
    plt.savefig(f"{FIG_DIR}/{event}_{det}_WHITENED.png")
    plt.close()

# -----------------------------------------
#    PROCESAR EVENTO
# -----------------------------------------
def process_event(event, det):
    print(f"\n===============================")
    print(f"   PROCESANDO {event} — {det}")
    print(f"===============================\n")

    raw_path = download_event(event, det)
    if raw_path is None:
        return

    # Leer raw
    ts = TimeSeries.read(raw_path)
    data = ts.value
    t = ts.times.value
    fs = ts.sample_rate.value

    # ---------- Highpass ----------
    hp = butter_highpass(data, fs, cutoff=30)
    ts_hp = TimeSeries(hp, times=t)
    ts_hp.name = f"{event}_{det}_clean"
    clean_path = os.path.join(CLEAN_DIR, f"{event}_{det}_clean.hdf5")
    ts_hp.write(clean_path, path="/")
    print(f"✔ Guardado CLEAN: {clean_path}")

    # ---------- Whitening ----------
    ts_white = whiten(ts_hp)
    ts_white.name = f"{event}_{det}_processed"
    proc_path = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")
    ts_white.write(proc_path, path="/")
    print(f"✔ Guardado PROCESSED: {proc_path}")

    # ---------- Figuras ----------
    plot_figures(event, det, ts_hp, ts_white)

# -----------------------------------------
#     PIPELINE GENERAL
# -----------------------------------------
def run_pipeline():
    print("\n============================================")
    print("        EJECUTANDO PIPELINE COMPLETO")
    print("============================================\n")

    for event in EVENTS:
        print(f"\n========== EVENTO: {event} ==========\n")

        for det in DETECTORS:
            print("\n--- (A1, A2, A3, A4, A5) Procesando datos ---\n")
            process_event(event, det)

    print("\n============================================")
    print("   PIPELINE MÓDULO A COMPLETO")
    print("============================================\n")

if __name__ == "__main__":
    run_pipeline()
