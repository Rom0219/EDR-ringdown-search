"""
scripts/edr_field_params.py

Traduce los parámetros del ajuste EDR FULL (fit_edr_full.py) a
parámetros físicos efectivos de la teoría EDR-Field.

Entrada:
  params_fit = [
      A22, A33, A21,
      d_om22, d_tau22,
      d_om33, d_tau33,
      d_om21, d_tau21,
      phi22, phi33, phi21,
      t0
  ]

Salida (objeto EDRFieldParams):
  - spiral_intensity      (I_sp): intensidad global de rotación del campo
  - radial_scale          (R_eff): escala radial efectiva del vórtice
  - effective_viscosity   (nu_eff): “viscosidad” gravitacional efectiva
  - multipole_anisotropy  (A_lm): anisotropía entre modos 22, 33, 21
  - mode_coupling         (C_modes): medida de acoplamiento de modos

Este mapeo es FENOMENOLÓGICO:
 - No reemplaza la derivación completa desde la ecuación EDR.
 - Sirve como capa intermedia para interpretar resultados QNM
   en términos de la física de EDR-Field.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class EDRFieldParams:
    spiral_intensity: float      # I_sp  (dimensiónless)
    radial_scale: float          # R_eff (dimensiónless, en unidades de M)
    effective_viscosity: float   # nu_eff (dimensiónless)
    multipole_anisotropy: float  # A_lm
    mode_coupling: float         # C_modes


def infer_edr_field_params(params_fit):
    """
    Recibe el vector de parámetros de fit_edr_full y devuelve
    un objeto EDRFieldParams con una interpretación física efectiva.

    params_fit =
      [A22, A33, A21,
       d_om22, d_tau22,
       d_om33, d_tau33,
       d_om21, d_tau21,
       phi22, phi33, phi21,
       t0]
    """

    (
        A22, A33, A21,
        d_om22, d_tau22,
        d_om33, d_tau33,
        d_om21, d_tau21,
        phi22, phi33, phi21,
        t0,
    ) = params_fit

    # ================================
    # 1) Intensidad de rotación global
    # ================================
    # Usamos el modo 22 (dominante) como proxy principal de la
    # intensidad rotacional del campo:
    #
    #   I_sp ~ 1 + d_om22
    #
    # Si d_om22 > 0 => campo EDR rota más “rápido” que GR.
    I_sp = 1.0 + d_om22

    # ================================
    # 2) Escala radial efectiva
    # ================================
    # Tomamos el promedio de desviaciones de tau como indicador
    # de qué tan “compacto / extendido” es el vórtice:
    #
    #   R_eff ~ 1 - <d_tau>   (si tau baja, el sistema es más compacto)
    #
    d_tau_mean = (d_tau22 + d_tau33 + d_tau21) / 3.0
    R_eff = 1.0 - d_tau_mean

    # ================================
    # 3) Viscosidad efectiva del campo
    # ================================
    # Interpretamos la reducción de tau como aumento de disipación:
    #
    #   nu_eff ~ - d_tau_mean
    #
    # Si d_tau_mean < 0 => tau más corto => más “viscoso”.
    nu_eff = -d_tau_mean

    # ================================
    # 4) Anisotropía multipolar
    # ================================
    # Medida de cuán distinto responde cada modo en frecuencia:
    #
    #   A_lm ~ var(d_om22, d_om33, d_om21)
    #
    d_oms = np.array([d_om22, d_om33, d_om21])
    A_lm = float(np.var(d_oms))

    # ================================
    # 5) Acoplamiento de modos
    # ================================
    # Relación entre amplitudes subdominantes y dominante:
    #
    #   C_modes ~ (A33 + A21) / A22
    #
    # Si A22 es muy pequeño, ponemos un seguro.
    if abs(A22) > 1e-6:
        C_modes = float((A33 + A21) / A22)
    else:
        C_modes = 0.0

    return EDRFieldParams(
        spiral_intensity=I_sp,
        radial_scale=R_eff,
        effective_viscosity=nu_eff,
        multipole_anisotropy=A_lm,
        mode_coupling=C_modes,
    )
