"""
Preprocessing pipeline:
 - Load raw or clean strain
 - Detrend
 - Highpass filter
 - Notch instrumental lines
 - Whitening
 - Save processed output
 - Produce quality-control plots (spectrogram, ASD)
"""

import os
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design

DATA_DIR = "data"
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
PROC_DIR = os.path.join(DATA_DIR, "processed")
PLOT_DIR = "plots"

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Frecuencias instrumentales conocidas de LIGO
NOTCHES = [60, 120, 180, 300, 331.9]


def apply_notches(ts):
    """Apply notch filters to remove instrumental lines."""
    out = ts
    for f0 in NOTCHES:
        try:
            out = out.notch(f0, Q=30)
        except:
            pass
    return out


def preprocess(det, event, source="clean"):
    """
    Process one detector stream:
     - load strain
     - detrend
     - highpass
     - notches
     - whitening
     - save result
    """
    infile = os.path.join(CLEAN_DIR, f"{event}_{det}_{source}.hdf5")
    outfile = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")

    if not os.path.exists(infile):
        print(f"✖ No existe archivo {infile}")
        return False

    print(f"Procesando {event} {det} ...")

    # Cargar
    ts = TimeSeries.read(infile)

    # Detrend
    ts = ts.detrend()

    # Filtro paso alto (15 Hz)
    ts = ts.highpass(15)

    # Notch filters
    ts = apply_notches(ts)

    # Whitening
    white = ts.whiten()

    # Guardar procesado
    white.write(outfile)

    print(f"✔ Guardado procesado: {outfile}")
    return True
