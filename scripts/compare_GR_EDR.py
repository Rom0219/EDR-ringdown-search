import numpy as np

# GR QNM expected approximate values
def GR_f0(M_solar):
    return 250 / (M_solar / 60)

def GR_tau(M_solar):
    return 4e-3 * (M_solar / 60)

# EDR: shift δω/ω estimated
def EDR_f0(M_solar, shift=0.01):
    return GR_f0(M_solar) * (1 + shift)

def EDR_tau(M_solar, shift=0.01):
    return GR_tau(M_solar) * (1 - shift)
