"""
summarize_edr.py

Lee TODOS los JSON en results/edr_full_pipeline/,
filtra los casos confiables y calcula estadísticas
del núcleo EDR (d_om22, d_tau22, A22).

Ejecutar así:
    python3 -m scripts.summarize_edr
"""

import os
import json
import numpy as np

RESULTS_DIR = "results/edr_full_pipeline"


def cargar_resultados():
    data = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fname), "r") as f:
                d = json.load(f)
                d["file"] = fname
                data.append(d)
    return data


def es_confiable(entry):
    p = entry["edr_full_params"]
    
    A22 = p["A22"]
    d_om22 = p["d_om22"]
    d_tau22 = p["d_tau22"]

    # Filtros básicos de confiabilidad
    if A22 < 0.03:        # muy débil → no confiable
        return False
    if abs(d_om22) >= 0.48:   # pegado al borde
        return False
    if abs(d_tau22) >= 0.48:  # pegado al borde
        return False

    return True


def resumen_valores(valores):
    if len(valores) == 0:
        return None
    arr = np.array(valores)
    return {
        "promedio": float(arr.mean()),
        "desviacion": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "cuenta": len(arr)
    }


def main():
    results = cargar_resultados()

    d_om22_vals = []
    d_tau22_vals = []
    A22_vals = []
    usados = []

    for r in results:
        if es_confiable(r):
            p = r["edr_full_params"]
            d_om22_vals.append(p["d_om22"])
            d_tau22_vals.append(p["d_tau22"])
            A22_vals.append(p["A22"])
            usados.append(r["file"])

    print("\n===================== ")
    print("  RESUMEN EDR (filtrado) ")
    print("=====================\n")

    print("Archivos usados (confiables):")
    for u in usados:
        print(" -", u)

    print("\n--- Estadísticas ---")
    print("A22:", resumen_valores(A22_vals))
    print("d_om22:", resumen_valores(d_om22_vals))
    print("d_tau22:", resumen_valores(d_tau22_vals))

    print("\n✔ Listo.")


if __name__ == "__main__":
    main()
