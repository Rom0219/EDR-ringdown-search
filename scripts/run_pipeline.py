import os
import json
import requests
from gwpy.timeseries import TimeSeries
from scipy.signal import butter, filtfilt
import numpy as np

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
PROC_DIR = "data/processed"

for d in [RAW_DIR, CLEAN_DIR, PROC_DIR]:
    os.makedirs(d, exist_ok=True)

EVENTS = [
    "GW150914", "GW151226", "GW170104", "GW170608",
    "GW170814", "GW170729", "GW190412", "GW190521",
    "GW190814", "GW200129"
]

DETECTORS = ["H1", "L1"]

API = "https://www.gw-openscience.org/eventapi/json/event/"

def get_losc_url(event, det):
    url = API + event + "/"
    resp = requests.get(url)
    data = resp.json()
    for entry in data[event]["detector"]:
        if entry["detector"] == det:
            return entry["frame"]["hdf5"]
    return None

def download_file(url, path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def butter_highpass(data, fs, cutoff=30, order=4):
    nyq = fs / 2
    norm = cutoff / nyq
    b, a = butter(order, norm, btype="high")
    return filtfilt(b, a, data)

def whiten_manual(ts, fft=4):
    psd = ts.psd(fft)
    interp = psd.interpolate(ts.frequencies)
    return ts / np.sqrt(interp)

def process_event(event, det):
    print(f"\n==== {event} — {det} ====")

    losc_url = get_losc_url(event, det)
    if losc_url is None:
        print("✖ No se encontró URL LOSC")
        return

    raw_path = os.path.join(RAW_DIR, f"{event}_{det}_raw.hdf5")

    print("Descargando:", losc_url)
    download_file(losc_url, raw_path)
    print("✔ Archivo descargado")

    ts = TimeSeries.read(raw_path)

    fs = ts.sample_rate.value
    hp = butter_highpass(ts.value, fs)
    clean = TimeSeries(hp, times=ts.times)
    clean_path = os.path.join(CLEAN_DIR, f"{event}_{det}_clean.hdf5")
    clean.write(clean_path, path="/")
    print("✔ Clean generado")

    white = whiten_manual(clean)
    proc_path = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")
    white.write(proc_path, path="/")
    print("✔ Whitening generado")

def run():
    print("\n=== PIPELINE LOSC ===\n")
    for ev in EVENTS:
        for det in DETECTORS:
            process_event(ev, det)
    print("\n=== FIN ===")

if __name__ == "__main__":
    run()
