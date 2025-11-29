# ringdown_templates.py
import numpy as np

def damped_sinusoid(t, f0, tau, phase=0.0, amplitude=1.0):
    return amplitude * np.exp(-t/tau) * np.cos(2*np.pi*f0*t + phase) * (t>=0)

def generate_template(fs, duration, f0, tau, phase=0.0, amplitude=1.0):
    t = np.arange(0, duration, 1.0/fs)
    return t, damped_sinusoid(t, f0, tau, phase, amplitude)

