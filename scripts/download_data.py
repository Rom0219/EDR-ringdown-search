import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import welch
import h5py
import os

# ======================================================
# 1) BLANQUEO MANUAL CORREGIDO — IGUAL A COLAB
# ======================================================

def whiten_manual(strain, fs, fftlength=4.0, overlap=2.0):
    """
    Whitening igual al comportamiento de GWpy:
    1) FFT de la señal
    2) PSD con Welch
    3) Interpolación del PSD a la malla FFT
    4) División en frecuencia
    5) Transformada inversa
    """

    N = len(strain)
    freqs = np.fft.rfftfreq(N, 1/fs)
    fft_data = np.fft.rfft(strain)

    # PSD usando Welch
    freqs_psd, psd = welch(
        strain,
        fs=fs,
        nperseg=int(fftlength * fs),
        noverlap=int(overlap * fs)
    )

    # Interpolación del PSD
    psd_interp = np.interp(freqs, freqs_psd, psd)

    # Blanqueo
    white_fft = fft_data / np.sqrt(psd_interp / 2.0)

    # Transformada inversa
    white_time = np.fft.irfft(white_fft, n=N)

    return white_time


# ======================================================
# 2) GUARDADO SEGURO (evita FileExists)
# ======================================================

def save_timeseries_safe(filename, data, fs):
    """
    Guarda un vector como HDF5 asegurando compatibilidad con GWpy.
    """
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)

    with h5py.File(filename, "w") as f:
        dset = f.create_dataset("strain", data=data)
        dset.attrs["fs"] = fs


# ======================================================
# 3) DESCARGA + PROCESAMIENTO COMPLETO
# ======================================================

def download_and_preprocess(event_name, detector, gps, t_pre=4, t_post=4,
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
        strain.value,
        fs
    )

    save_timeseries_safe(
        f"data/clean/{event_name}_{detector}_clean.hdf5",
        strain_clean.value,
        fs
    )

    save_timeseries_safe(
        f"data/white/{event_name}_{detector}_white.hdf5",
        strain_white,
        fs
    )

    print("  ✓ Datos procesados y guardados")
    return strain_white
