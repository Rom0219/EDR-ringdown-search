import numpy as np
from pycbc.filter import matched_filter
from pycbc.types import TimeSeries

def compute_snr(strain, template, sample_rate):

    data = TimeSeries(strain, delta_t=1/sample_rate)
    temp = TimeSeries(template, delta_t=1/sample_rate)

    snr = matched_filter(temp, data)
    peak_snr = max(abs(snr))

    return peak_snr
