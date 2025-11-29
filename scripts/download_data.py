from gwosc import datasets
from gwpy.timeseries import TimeSeries
import json, os

DATA_DIR = "data"

def download_event(event, duration=16):
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"\n=== Descargando {event} ===")

    # Get event metadata
    try:
        event_info = datasets.event_gps(event)
        t0 = event_info[event]
    except:
        print(f"No se pudo obtener el GPS de {event}")
        return
    
    start = t0 - duration
    end = t0 + duration

    for det in ["H1", "L1", "V1"]:
        try:
            ts = TimeSeries.fetch_open_data(det, start, end)
            outname = f"{DATA_DIR}/{event}_{det}.hdf5"
            ts.write(outname)
            print("Guardado:", outname)
        except Exception as e:
            print(f"{det} no disponible:", e)


if __name__ == "__main__":
    with open("events.json") as f:
        events = json.load(f)["events"]

    for ev in events:
        download_event(ev)
