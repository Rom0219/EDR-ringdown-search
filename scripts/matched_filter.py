# matched_filter.py
import numpy as np
from scipy.signal import fftconvolve
from gwpy.signal import filter_design

def whiten(strain, psd, fs):
    # simple whitening: divide by sqrt(psd) in freq domain
    # but better use gwpy.timeseries.whiten() or pycbc.filter.whiten
    from gwpy.timeseries import TimeSeries
    ts = TimeSeries(strain, sample_rate=fs)
    return ts.whiten(window='hann', method='median', beta=4).value

def matched_filter_whitened(data, template):
    # data and template must be whitened and same sampling
    corr = fftconvolve(data, template[::-1], mode='same')
    # estimate SNR time series: corr / sqrt(<template,template>)
    norm = np.sqrt(np.sum(template**2))
    snr = corr / norm
    return snr

