import os
from gwpy.timeseries import TimeSeries
from gwpy.signal.filter_design import bandpass
from matplotlib import pyplot as plt

from scripts.download_data import download_strain, load_strain

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
    "GW200129",
]

DETECTORS = ["H1", "L1"]

CLEAN_DIR = "data/clean"
SPEC_DIR = "output/spectrograms"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(SPEC_DIR, exist_ok=True)


def process_event(event, det):
    print(f"\n==== {event} — {det} ====")

    # 1. Descargar archivo
    path = download_strain(event, det)
    if not path:
        print("✖ No se pudo descargar")
        return

    # 2. Cargar señal
    ts = load_strain(path)
    if ts is None:
        return

    # 3. Bandpass + Whitening
    ts_bp = ts.bandpass(20, 1000)
    ts_white = ts_bp.whiten()  # ← método correcto

    # 4. Guardar señal procesada
    out_clean = f"{CLEAN_DIR}/{event}_{det}_white.hdf5"
    ts_white.write(out_clean, format="hdf5")
    print(f"✓ Señal procesada guardada en {out_clean}")

    # 5. Generar espectrograma
    out_fig = f"{SPEC_DIR}/{event}_{det}_spectrogram.png"
    sg = ts_white.spectrogram(fftlength=4, overlap=2)
    fig = sg.plot(norm="log")
    fig.savefig(out_fig)
    plt.close(fig.fig)
    print(f"✓ Espectrograma guardado en {out_fig}")


def run():
    print("\n=== PIPELINE GWOSC — INICIO ===")

    for event in EVENTS:
        print(f"\n>>> EVENTO: {event}")
        for det in DETECTORS:
            try:
                process_event(event, det)
            except Exception as e:
                print(f"✖ Error procesando {event}/{det}: {e}")

    print("\n=== PIPELINE COMPLETO ===")


if __name__ == "__main__":
    run()
