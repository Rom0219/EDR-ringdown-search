"""
scripts/download_data.py

Descarga datos de LIGO usando GWPy (método moderno).
"""

import os
from gwpy.timeseries import TimeSeries

# Tabla oficial de tiempos GPS
GPS_TIMES = {
    "GW150914": 1126259462.4,
    "GW151226": 1135136350.6,
    "GW170104": 1167559936.6,
    "GW170608": 1180922494.5,
    "GW170814": 1186741861.5,
    "GW170729": 1185389807.3,
    "GW190412": 1239082262.2,
    "GW190521": 1242442967.4,
    "GW190814": 1249852257.0,
    "GW200129": 1264316115.4
}

def download_event(event, det, outdir="data/raw"):

    if event not in GPS_TIMES:
        print(f"✖ Evento {event} no está en la tabla GPS.")
        return None

    gps = GPS_TIMES[event]
    duration = 8  # segundos alrededor del evento
    start = gps - duration/2
    end   = gps + duration/2

    print(f"\nDescargando {event} ({det}) — GPS {gps}")

    try:
        ts = TimeSeries.fetch_open_data(
            det,
            start,
            end,
            sample_rate=4096,
            cache=True,
            verbose=True
        )

        # Crear carpeta si no existe
        os.makedirs(outdir, exist_ok=True)

        # Guardar
        outpath = os.path.join(outdir, f"{event}_{det}_raw.hdf5")
        ts.write(outpath)
        print(f"✔ Guardado en {outpath}")

        return outpath

    except Exception as e:
        print(f"✖ Error descargando {event} / {det}")
        print(e)
        return None
