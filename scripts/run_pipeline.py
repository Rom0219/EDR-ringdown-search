import os
from gwpy.timeseries import TimeSeries
from scripts.download_data import download_strain, load_strain

# ============================================
# LISTA FINAL DE EVENTOS (10)
# ============================================
EVENTS = {
    "GW150914":   {"gps": 1126259462, "run": "O1"},
    "GW151226":   {"gps": 1135136350, "run": "O1"},
    "GW170104":   {"gps": 1167559936, "run": "O2"},
    "GW170608":   {"gps": 1180922494, "run": "O2"},
    "GW170814":   {"gps": 1186741861, "run": "O2"},
    "GW170729":   {"gps": 1185389807, "run": "O2"},
    "GW170823":   {"gps": 1187529256, "run": "O2"},
    "GW190412":   {"gps": 1239082262, "run": "O3"},
    "GW190521":   {"gps": 1242442967, "run": "O3"},
    "GW190814":   {"gps": 1249852257, "run": "O3"},
}

DET_LIST = ["H1", "L1"]

RAW_PATH   = "data/raw"
CLEAN_PATH = "data/clean"
WHITE_PATH = "data/white"

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(CLEAN_PATH, exist_ok=True)
os.makedirs(WHITE_PATH, exist_ok=True)


# =====================================================
# LIMPIEZA  (filtro pasa banda + notch 60Hz)
# =====================================================
def clean_strain(ts):
    try:
        ts = ts.bandpass(20, 500)
        ts = ts.notch(60)
        return ts
    except Exception as e:
        print("✖ Error limpiando:", e)
        return None


# =====================================================
# BLANQUEADO
# =====================================================
def whiten_strain(ts):
    try:
        white = ts.whiten(2, 2)
        return white
    except Exception as e:
        print("✖ Error blanqueando:", e)
        return None


# =====================================================
# PROCESAR EVENTO COMPLETO
# =====================================================
def process_event(event_name, det, gps, obs_run):
    print(f"\n===== {event_name} — {det} =====")

    # (1) DESCARGA
    raw_path = download_strain(event_name, det, gps, obs_run)
    if raw_path is None:
        print("✖ No se pudo descargar.\n")
        return

    # (2) CARGAR
    ts_raw = load_strain(raw_path)
    if ts_raw is None:
        print("✖ No se pudo cargar el strain.\n")
        return

    # (3) LIMPIEZA
    ts_clean = clean_strain(ts_raw)
    if ts_clean is None:
        print("✖ No se pudo limpiar.\n")
        return

    clean_out = os.path.join(CLEAN_PATH, f"{event_name}_{det}_clean.hdf5")
    ts_clean.write(clean_out, format="hdf5")
    print("✔ Limpio guardado:", clean_out)

    # (4) BLANQUEADO
    ts_white = whiten_strain(ts_clean)
    if ts_white is None:
        print("✖ No se pudo blanquear.\n")
        return

    white_out = os.path.join(WHITE_PATH, f"{event_name}_{det}_white.hdf5")
    ts_white.write(white_out, format="hdf5")
    print("✔ Blanqueado guardado:", white_out)


# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    print("\n=== PIPELINE COMPLETO GWOSC ===\n")

    for event_name, info in EVENTS.items():
        gps = info["gps"]
        run = info["run"]

        print(f"\n>>> EVENTO: {event_name}\n")

        for det in DET_LIST:
            process_event(event_name, det, gps, run)

    print("\n=== PIPELINE TERMINADO ===\n")


if __name__ == "__main__":
    main()
