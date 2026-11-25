# -*- coding: utf-8 -*-
"""
Script 03 - Aislamiento por Inundaciones: puntos de muestreo cada 15 m
======================================================================
Fuente: rnc2025Edomex.gpkg (capas 10km_red_vial_buffer y 20km_red_vial_buffer).

Genera puntos de muestreo cada <= 15 m sobre los tramos de la red vial y
muestra el conteo final.

Para que los resultados sean comparables con el script 02, se reutilizan sus
funciones y criterios (se importan de 02_red_vial_edomex.py, que debe estar en
la misma carpeta):
  - Exclusión de tramos: ESTATUS = 'Deshabilitado', CONDICION = 'En
    construcción - cerrado' o CIRCULA = 'Cerrada en ambos sentidos'.
  - Se muestrean solo los tramos de la componente principal (componente
    conexa más grande del grafo no dirigido de uniones de la RNC).
  - Por tramo de longitud L (elipsoidal GRS80): n = max(1, ceil(L / 15))
    puntos en los centros de n partes iguales -> espaciado <= 15 m, al menos
    un punto por tramo y sin puntos duplicados en las uniones.

Sobre el intervalo de 15 m: es la mitad del píxel del ráster de inundación
tr100_prcent.tif (30 m). Con 30 m, un tramo puede cruzar un píxel por una
esquina sin que caiga ningún punto en él; con 15 m todo cruce de más de 15 m
dentro de un píxel queda muestreado, a costa de duplicar el número de puntos.

Uso:
  python 03_puntos_muestreo_15m.py
  python 03_puntos_muestreo_15m.py --capa 20km_red_vial_buffer
  python 03_puntos_muestreo_15m.py --gpkg RUTA\\rnc2025Edomex.gpkg

Requisitos: numpy, networkx.
Salidas (carpeta salidas\\):
  03_puntos_muestreo_15m_<capa>.csv.gz   (fid, ID_RED, n, lon, lat; EPSG:6365)
  03_resumen_puntos_muestreo_15m.csv
"""

import argparse
import csv
import gzip
import importlib.util
import sqlite3
import time
from pathlib import Path

import networkx as nx
import numpy as np

AQUI = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("red02", AQUI / "02_red_vial_edomex.py")
red02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(red02)

INTERVALO = 15.0                                           # m
SALIDAS = AQUI / "salidas"


def excluido(est, cond, circ) -> bool:
    E = red02.EXCLUSION
    return est in E["ESTATUS"] or cond in E["CONDICION"] or circ in E["CIRCULA"]


def procesar_capa(con, capa, intervalo):
    t0 = time.time()
    filas = con.execute(
        f'SELECT fid, ID_RED, ESTATUS, CONDICION, CIRCULA, UNION_INI, UNION_FIN, geom '
        f'FROM "{capa}"').fetchall()
    validos = [f for f in filas if not excluido(f[2], f[3], f[4])]

    # Componente principal
    G = nx.MultiGraph()
    for fid, _, _, _, _, u, v, _ in validos:
        G.add_edge(u, v, key=fid)
    principal = max(nx.connected_components(G), key=len)
    fids_princ = {k for u, v, k in G.subgraph(principal).edges(keys=True)}

    # Muestreo
    SALIDAS.mkdir(exist_ok=True)
    ruta = SALIDAS / f"03_puntos_muestreo_15m_{capa}.csv.gz"
    n_pts, long_m, por_tramo = 0, 0.0, []
    with gzip.open(ruta, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fid", "ID_RED", "n", "lon", "lat"])
        for fid, idr, *_, g in validos:
            if fid not in fids_princ:
                continue
            L, pts = red02.puntos_a_lo_largo(red02.gpkg_a_lineas(g), intervalo)
            for i, (x, y) in enumerate(pts):
                w.writerow([fid, idr, i, f"{x:.7f}", f"{y:.7f}"])
            n_pts += len(pts)
            long_m += L
            por_tramo.append(len(pts))

    por_tramo = np.array(por_tramo)
    print("\n" + "=" * 66)
    print(f" PUNTOS DE MUESTREO CADA <= {intervalo:g} m - {capa}")
    print("=" * 66)
    print(f"  Tramos en la capa                       : {len(filas):>14,d}")
    print(f"  Tramos excluidos                        : {len(filas) - len(validos):>14,d}")
    print(f"  Tramos muestreados (comp. principal)    : {len(fids_princ):>14,d}")
    print(f"  Longitud muestreada (km)                : {long_m / 1000:>14,.1f}")
    print(f"  PUNTOS DE MUESTREO (conteo final)       : {n_pts:>14,d}")
    print(f"  Espaciado medio efectivo (m)            : {long_m / n_pts:>14,.2f}")
    print(f"  Puntos por tramo: mín {por_tramo.min()}, mediana {np.median(por_tramo):.0f}, "
          f"máx {por_tramo.max():,d}; tramos con 1 solo punto (< 15 m): "
          f"{(por_tramo == 1).sum():,d}")
    print(f"  Guardado en: {ruta}   [{time.time() - t0:.0f} s]")

    return {"capa": capa, "intervalo_m": intervalo, "tramos_capa": len(filas),
            "tramos_excluidos": len(filas) - len(validos),
            "tramos_muestreados": len(fids_princ),
            "longitud_muestreada_km": round(long_m / 1000, 3),
            "puntos_muestreo": n_pts,
            "espaciado_medio_m": round(long_m / n_pts, 3),
            "tramos_con_un_punto": int((por_tramo == 1).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpkg", type=Path, default=red02.GPKG_DEFECTO)
    ap.add_argument("--capa", choices=red02.CAPAS, action="append",
                    help="capa a procesar (por defecto, ambas)")
    ap.add_argument("--intervalo", type=float, default=INTERVALO)
    args = ap.parse_args()
    if not args.gpkg.exists():
        raise SystemExit(f"No existe: {args.gpkg}")
    print(f"Fuente: {args.gpkg}")
    con = sqlite3.connect(f"file:{args.gpkg.as_posix()}?mode=ro", uri=True)

    res = [procesar_capa(con, c, args.intervalo) for c in (args.capa or red02.CAPAS)]

    ruta = SALIDAS / "03_resumen_puntos_muestreo_15m.csv"
    with open(ruta, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print(f"\nResumen guardado en: {ruta}")


if __name__ == "__main__":
    main()
