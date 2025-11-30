# scripts/run_pipeline.py
#
# Pipeline mínimo: descarga y preprocesa strain real de 10 eventos
# usando sólo GWOSC + GWpy (fetch_open_data).

from scripts.download_data import download_and_preprocess

# 10 eventos que queremos analizar
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

DETECTORS = ["H1", "L1"]  # Hanford y Livingston


def main() -> None:
    print("=== PIPELINE GWOSC + GWpy (fetch_open_data) ===\n")

    for event in EVENTS:
        print(f">>> EVENTO: {event}")
        for det in DETECTORS:
            print(f"\n===== {event} — {det} =====")
            try:
                raw_path, clean_path, white_path = download_and_preprocess(event, det)
                print("  ✔ Descarga y preprocesado completados.")
                print(f"    RAW   : {raw_path}")
                print(f"    CLEAN : {clean_path}")
                print(f"    WHITE : {white_path}")
            except Exception as e:
                print(f"  ✖ Error procesando {event}/{det}: {e}")

    print("\n=== PIPELINE COMPLETO ===")


if __name__ == "__main__":
    main()
