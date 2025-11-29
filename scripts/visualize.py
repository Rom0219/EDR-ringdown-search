"""
Visualization module for processed LIGO data:
 - raw strain plot
 - whitened strain plot
 - ASD (Amplitude Spectral Density)
 - Spectrogram
"""

import os
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
PROC_DIR = os.path.join(DATA_DIR, "processed")
PLOT_DIR = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)


def plot_asd(det, event, source="processed"):
    """Generate ASD plot."""
    infile = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")

    if not os.path.exists(infile):
        print(f"✖ No existe archivo {infile}")
        return

    ts = TimeSeries.read(infile)

    plot = Plot(ts.asd(fftlength=4))
    plot.title = f"ASD — {event} — {det}"
    out = os.path.join(PLOT_DIR, f"{event}_{det}_ASD.png")
    plot.save(out)
    print(f"✔ ASD guardado: {out}")


def plot_whitened(det, event):
    """Plot whitened strain."""
    infile = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")
    if not os.path.exists(infile):
        print(f"✖ No existe archivo {infile}")
        return

    ts = TimeSeries.read(infile)

    plot = Plot(ts)
    plot.title = f"Whitened strain — {event} — {det}"
    out = os.path.join(PLOT_DIR, f"{event}_{det}_whitened.png")
    plot.save(out)
    print(f"✔ Whitened guardado: {out}")


def plot_spectrogram(det, event):
    """Spectrogram plot."""
    infile = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")
    if not os.path.exists(infile):
        print(f"✖ No existe archivo {infile}")
        return

    ts = TimeSeries.read(infile)

    plot = ts.spectrogram2(fftlength=0.5, overlap=0.25).plot()
    plot[0].set_title(f"Spectrogram — {event} — {det}")

    out = os.path.join(PLOT_DIR, f"{event}_{det}_spectrogram.png")
    plot[0].figure.savefig(out)
    print(f"✔ Spectrogram guardado: {out}")
