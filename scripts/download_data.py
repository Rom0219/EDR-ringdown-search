import os
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import welch
import h5py

# ======================================================
# 1) BLANQUEO MANUAL CORREGIDO — ESTABLE (LIGO-LIKE)
# ======================================================

def whiten_manual(strain, fs, seglen=4):
    """
    Whitening robusto estilo LIGO:
    - PSD con Welch
    - FFT
    - División por sqrt(PSD/2)
    - iFFT
    """

    N = len(strain)
    dt = 1.0 / fs

    # FFT
    freq = np.fft.rfftfreq(N, dt)
    fft_data = np.fft.rfft(strain)

    # PSD Welch
    nperseg = int(seglen * fs)
    freqs_psd, psd = welch(
        strain,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )

    # evitar ceros
    psd = np.where(psd <= 1e-30, 1e-30, psd)

    # interpolar PSD al grid FFT
    psd_interp = np.interp(freq, freqs_psd, psd)

    # whitening correcto
    white_fft = fft_data / np.sqrt(psd_interp / 2.0)

    # señal whitened
    white = np.fft.irfft(white_fft, n=N)

    return white


# ======================================================
# 2) GUARDADO SEGURO (HDF5 compatible con GWpy)
# ======================================================

def save_timeseries_safe(filename, data, fs):
    """
    Guarda un vector como HDF5 con atributos fs.
    Compatible con nuestra lectura manual.
    """
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)

    with h5py.File(filename, "w") as f:
        dset = f.create_dataset("strain", data=data)
        dset.attrs["fs"] = fs


# ======================================================
# 3) DESCARGA + PROCESAMIENTO COMPLETO
# ======================================================

def download_and_preprocess(event_name, detector, gps,
                            t_pre=4, t_post=4,
                            fmin=20, fmax=1024):
    """
    Descarga de GWOSC con GWpy y procesamiento completo.
    """

    print(f"  GPS = {gps}")
    t0 = gps - t_pre
    t1 = gps + t_post
    print(f"  Ventana = [{t0}, {t1}]")

    # -----------------------
    # 1) DESCARGA
    # -----------------------
    try:
        strain = TimeSeries.fetch_open_data(detector, t0, t1, cache=False)
    except Exception as e:
        print(f"✖ Error descargando datos: {e}")
        return None

    fs = strain.sample_rate.value

    # -----------------------
    # 2) LIMPIEZA
    # -----------------------
    strain_clean = strain.detrend().bandpass(fmin, fmax)

    # -----------------------
    # 3) BLANQUEO MANUAL
    # -----------------------
    try:
        strain_white = whiten_manual(strain_clean.value, fs)
    except Exception as e:
        print(f"✖ Error blanqueando: {e}")
        return None

    # -----------------------
    # 4) GUARDADO
    # -----------------------

    save_timeseries_safe(
        f"data/raw/{event_name}_{detector}_raw.hdf5",
        strain.value, fs
    )

    save_timeseries_safe(
        f"data/clean/{event_name}_{detector}_clean.hdf5",
        strain_clean.value, fs
    )

    save_timeseries_safe(
        f"data/white/{event_name}_{detector}_white.hdf5",
        strain_white, fs
    )

    print("  ✓ Datos procesados y guardados")

    return strain_white
