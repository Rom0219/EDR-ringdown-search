# download_data.py
from gwosc import datasets, event
from gwosc.datasets import get_urls
from gwpy.timeseries import TimeSeries
import os

def download_event_strain(event_name, outdir='data', duration=16, sample_rate=4096):
    os.makedirs(outdir, exist_ok=True)
    ev = event.Event(event_name)
    t0 = ev['t0']  # GPS time of coalescence
    start = int(t0 - duration//2)
    end   = int(t0 + duration//2)
    # use gwpy to fetch open data (auto chooses host)
    for det in ['H1','L1','V1']:
        try:
            ts = TimeSeries.fetch_open_data(det, start, end, sample_rate=sample_rate)
            fname = os.path.join(outdir, f"{event_name}_{det}_{start}_{end}.hdf")
            ts.write(fname, format='hdf5')
            print("Saved", fname)
        except Exception as e:
            print("No data for", det, e)

if __name__=='__main__':
    events = ["GW150914","GW151226"]  # expand with the 10 events
    for e in events:
        download_event_strain(e)

