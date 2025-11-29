import os
import h5py
import requests
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from gwosc.api import event_json

RAW_DIR = "data/raw"

os.makedirs(RAW_DIR, exist_ok=True)

# ------------------------------------------
# Obtener URLs reales desde la API moderna
# ------------------------------------------
def get_strain_url(event, detector):
    """
    Devuelve la URL HDF5 oficial de strain LOSC v1/v2/v3 según GWOSC.
    """
    meta = event_json(event)
    if "strain" not in meta:
        raise ValueError(f"No hay strain para {event}")

    # Filtrar strain del detector correcto
    for entry in meta["strain"]:
        if entry["ifo"] == detector:
            return entry["url"]

    raise ValueError(f"No hay strain para {event}/{detector}")


# ------------------------------------------
# Descargar archivo
# ------------------------------------------
def download_strain(event, detector):
    out_path = f"{RAW_DIR}/{event}_{detector}.hdf5"

    if os.path.exists(out_path):
        print(f"✓ Ya existe {out_path}")
        return out_path

    try:
        print(f"Obteniendo URL oficial para {event}/{detector}...")
        url = get_strain_url(event, detector)
    except Exception as e:
        print(f"✖ Error obteniendo URL para {event}/{detector}: {e}")
        return None

    print(f"Descargando:\n{url}")

    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        print(f"✖ Error HTTP {resp.status_code}")
        return None

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"✓ Archivo guardado en {out_path}")
    return out_path


# ------------------------------------------
# Cargar TimeSeries desde HDF5
# ------------------------------------------
def load_strain(path):
    try:
        ts = TimeSeries.read(path, format="hdf5.gwosc")
        print(f"✓ Señal cargada correctamente ({len(ts)} muestras)")
        return ts
    except Exception as e:
        print(f"✖ Error leyendo archivo {path}: {e}")
        return None
