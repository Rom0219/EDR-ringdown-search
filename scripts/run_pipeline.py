"""
Full Module A Pipeline:
 - Download raw + clean strain
 - Preprocess (highpass, notches, whitening)
 - Generate plots (ASD, whitened, spectrogram)
"""

import json
from scripts.download_data import download_event
from scripts.preprocess import preprocess
from scripts.visualize import plot_asd, plot_whitened, plot_spectrogram

DETECTORS = ["H1", "L1", "V1"]

def run_pipeline():
    print("\n============================================")
    print("        EJECUTANDO PIPELINE COMPLETO")
    print("============================================\n")

    # Cargar lista de eventos
    with open("events.json") as f:
        events = json.load(f)["events"]

    for event in events:
        print(f"\n========== EVENTO: {event} ==========")

        # 1) Descargar datos RAW + CLEAN
        print("\n--- (A1 & A2) Descargando datos ---")
        download_event(event)

        # 2) Preprocesamiento
        print("\n--- (A3) Preprocesando ---")
        for det in DETECTORS:
            preprocess(det, event)

        # 3) Visualizaciones
        print("\n--- (A4 & A5) Generando figuras ---")
        for det in DETECTORS:
            plot_asd(det, event)
            plot_whitened(det, event)
            plot_spectrogram(det, event)

    print("\n============================================")
    print("   PIPELINE MÓDULO A COMPLETO (NO CORRER)")
    print("============================================\n")


if __name__ == "__main__":
    run_pipeline()
