import numpy as np

def damped_sine(t, f0, tau, phase=0, amp=1):
    return amp * np.exp(-t/tau) * np.cos(2*np.pi*f0*t + phase)

def template(fs, duration, f0, tau):
    t = np.arange(0, duration, 1/fs)
    return t, damped_sine(t, f0, tau)
