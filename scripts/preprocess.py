from gwpy.timeseries import TimeSeries
import numpy as np

def preprocess(filename, f_low=20):
    ts = TimeSeries.read(filename)

    ts = ts.detrend()
    ts = ts.highpass(f_low)
    white = ts.whiten()

    return white

