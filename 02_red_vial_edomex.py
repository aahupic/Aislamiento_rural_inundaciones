# -*- coding: utf-8 -*-
"""
Script 02 - Aislamiento por Inundaciones: red vial del Estado de México
=======================================================================
Fuente: rnc2025Edomex.gpkg (Red Nacional de Caminos 2025, INEGI), recortada
al Estado de México con área de amortiguamiento. El GeoPackage trae dos capas
de tramos:
    10km_red_vial_buffer  -> Edomex + 10 km
    20km_red_vial_buffer  -> Edomex + 20 km
Por defecto se procesan las dos (se puede elegir una con --capa).

Indicadores:
  1. Tramos excluidos (cerrados, deshabilitados o en construcción)
     Tramos / aristas del grafo (total y en la componente principal)
  2. Nodos (intersecciones) de la componente principal
  3. Porcentaje de nodos en la componente principal
  4. Longitud total de la red del grafo
  5. Puntos de muestreo sobre los tramos

Criterios (explícitos para que puedan debatirse):
  - Exclusión: un tramo sale del grafo si cumple AL MENOS uno de:
        ESTATUS   = 'Deshabilitado'
        CONDICION = 'En construcción - cerrado'
        CIRCULA   = 'Cerrada en ambos sentidos'
    (TIPO_VIAL = 'Cerrada' es un tipo de calle -calle cerrada / sin salida-,
    no un cierre a la circulación, por lo que NO se excluye.)
  - Grafo: nodos = uniones de la RNC (UNION_INI, UNION_FIN); aristas = tramos
    válidos (multigrafo: se conservan tramos paralelos).
  - Componente principal: componente conexa más grande del grafo NO dirigido
    (conectividad física). Como referencia se reporta también la componente
    fuertemente conexa más grande del grafo dirigido según CIRCULA
    ('Un sentido' se toma en el sentido de digitalización UNION_INI ->
    UNION_FIN, supuesto a verificar con la documentación de la RNC).
  - Longitud: se calcula de la geometría (EPSG:6365, grados) con distancias
    elipsoidales GRS80 por segmento, y se compara con el campo LONGITUD.
  - Muestreo: por cada tramo de la componente principal con longitud L,
    n = max(1, ceil(L / INTERVALO)) puntos en los centros de n partes
    iguales. Así el espaciado es <= INTERVALO, todo tramo recibe al menos un
    punto y no se duplican puntos en las uniones compartidas.
    INTERVALO = 30 m por defecto, igual al tamaño de píxel del ráster de
    inundación tr100_prcent.tif (30 m): un espaciado mayor puede saltarse
    franjas inundadas de un píxel; uno menor repite píxeles sin ganar
    información. Se puede cambiar con --intervalo.

Uso:
  python 02_red_vial_edomex.py
  python 02_red_vial_edomex.py --capa 20km_red_vial_buffer --intervalo 30
  python 02_red_vial_edomex.py --gpkg RUTA\\rnc2025Edomex.gpkg

Requisitos: numpy, networkx (lee el GeoPackage con sqlite3; no requiere GDAL).
Salidas (carpeta salidas\\):
  02_resumen_red_vial_edomex.csv
  02_puntos_muestreo_<capa>.csv.gz   (fid, ID_RED, n, lon, lat)
"""

import argparse
import csv
import gzip
import math
import sqlite3
import struct
import time
from pathlib import Path

import networkx as nx
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent          # D:\ARTUAEM
GPKG_DEFECTO = BASE_DIR / "rnc2025Edomex.gpkg"
SALIDAS = Path(__file__).resolve().parent / "salidas"
CAPAS = ["10km_red_vial_buffer", "20km_red_vial_buffer"]
INTERVALO_DEFECTO = 30.0                                   # m

EXCLUSION = {
    "ESTATUS": {"Deshabilitado"},
    "CONDICION": {"En construcción - cerrado"},
    "CIRCULA": {"Cerrada en ambos sentidos"},
}

# Elipsoide GRS80
A = 6378137.0
F = 1 / 298.257222101
E2 = F * (2 - F)


# ----------------------------------------------------------------------------
# Lectura de geometrías GeoPackage (WKB) sin GDAL
# ----------------------------------------------------------------------------
_ENV_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def gpkg_a_lineas(blob: bytes):
    """Devuelve lista de arreglos (k,2) lon/lat de una (Multi)LineString."""
    flags = blob[3]
    off = 8 + _ENV_BYTES[(flags >> 1) & 7]
    return _wkb(memoryview(blob), off)[0]


def _wkb(b, off):
    orden = "<" if b[off] == 1 else ">"
    tipo = struct.unpack_from(orden + "I", b, off + 1)[0]
    off += 5
    if tipo & 0x80000000:                                  # EWKB con Z
        base, dims = tipo & 0xFF, 3
    else:                                                  # ISO: 0=XY 1=XYZ 2=XYM 3=XYZM
        base, dims = tipo % 1000, {0: 2, 1: 3, 2: 3, 3: 4}[tipo // 1000]
    if base == 2:                                          # LineString
        n = struct.unpack_from(orden + "I", b, off)[0]
        off += 4
        pts = np.frombuffer(b, dtype=orden + "f8", count=n * dims, offset=off).reshape(n, dims)[:, :2]
        return [pts], off + 8 * n * dims
    if base == 5:                                          # MultiLineString
        n = struct.unpack_from(orden + "I", b, off)[0]
        off += 4
        partes = []
        for _ in range(n):
            p, off = _wkb(b, off)
            partes.extend(p)
        return partes, off
    raise ValueError(f"Tipo WKB no soportado: {tipo}")


# ----------------------------------------------------------------------------
# Geodesia: longitud elipsoidal por segmento (precisa para segmentos cortos)
# ----------------------------------------------------------------------------
def longitudes_segmentos(xy: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(xy[:, 0]), np.radians(xy[:, 1])
    latm = (lat[:-1] + lat[1:]) / 2
    s = np.sin(latm)
    w = np.sqrt(1 - E2 * s * s)
    M = A * (1 - E2) / w**3                                # radio meridiano
    N = A / w                                              # radio primer vertical
    dy = M * np.diff(lat)
    dx = N * np.cos(latm) * np.diff(lon)
    return np.hypot(dx, dy)


def puntos_a_lo_largo(partes, intervalo):
    """Puntos en los centros de n partes iguales del tramo (n = ceil(L/intervalo))."""
    segs = [(p, longitudes_segmentos(p)) for p in partes if len(p) > 1]
    L = sum(float(d.sum()) for _, d in segs)
    if L == 0:
        p = partes[0][0]
        return L, np.array([[p[0], p[1]]])
    n = max(1, math.ceil(L / intervalo))
    objetivos = (np.arange(n) + 0.5) * (L / n)
    # Concatenar partes como una sola polilínea medida (sin contar saltos entre partes)
    xy = np.vstack([p for p, _ in segs])
    acum, inicio = [], 0.0
    for _, dd in segs:
        c = inicio + np.r_[0.0, np.cumsum(dd)]
        acum.append(c)
        inicio = c[-1]
    acum = np.concatenate(acum)
    lon = np.interp(objetivos, acum, xy[:, 0])
    lat = np.interp(objetivos, acum, xy[:, 1])
    return L, np.column_stack([lon, lat])


# ----------------------------------------------------------------------------
# Proceso de una capa
# ----------------------------------------------------------------------------
def procesar_capa(con, capa, intervalo):
    t0 = time.time()
    print("\n" + "=" * 70)
    print(f" RED VIAL - {capa}")
    print("=" * 70)

    filas = con.execute(
        f'SELECT fid, ID_RED, ESTATUS, CONDICION, CIRCULA, UNION_INI, UNION_FIN, '
        f'LONGITUD, geom FROM "{capa}"').fetchall()
    total = len(filas)

    # 1. Exclusiones
    motivo = {k: 0 for k in EXCLUSION}
    validos, excluidos = [], 0
    for f in filas:
        _, _, est, cond, circ, *_ = f
        flags = {"ESTATUS": est in EXCLUSION["ESTATUS"],
                 "CONDICION": cond in EXCLUSION["CONDICION"],
                 "CIRCULA": circ in EXCLUSION["CIRCULA"]}
        for k, v in flags.items():
            motivo[k] += v
        if any(flags.values()):
            excluidos += 1
        else:
            validos.append(f)

    # 2. Grafo
    G = nx.MultiGraph()
    D = nx.DiGraph()
    long_geom = {}
    for fid, idr, _, _, circ, u, v, L, g in validos:
        partes = gpkg_a_lineas(g)
        lg = float(sum(longitudes_segmentos(p).sum() for p in partes if len(p) > 1))
        long_geom[fid] = lg
        G.add_edge(u, v, key=fid, fid=fid, longitud=lg, longitud_campo=L)
        D.add_edge(u, v)
        if circ != "Un sentido":
            D.add_edge(v, u)

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    principal = comps[0]
    Gp = G.subgraph(principal)
    scc = max(nx.strongly_connected_components(D), key=len)

    n_nodos = G.number_of_nodes()
    n_princ = len(principal)
    pct = 100 * n_princ / n_nodos
    long_total = sum(d["longitud"] for *_, d in G.edges(data=True)) / 1000
    long_princ = sum(d["longitud"] for *_, d in Gp.edges(data=True)) / 1000
    long_campo = sum(d["longitud_campo"] for *_, d in G.edges(data=True)) / 1000

    # 5. Puntos de muestreo (tramos de la componente principal)
    fids_princ = {d["fid"] for *_, d in Gp.edges(data=True)}
    SALIDAS.mkdir(exist_ok=True)
    ruta_pts = SALIDAS / f"02_puntos_muestreo_{capa}.csv.gz"
    n_pts = 0
    with gzip.open(ruta_pts, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fid", "ID_RED", "n", "lon", "lat"])
        for fid, idr, *_, g in validos:
            if fid not in fids_princ:
                continue
            _, pts = puntos_a_lo_largo(gpkg_a_lineas(g), intervalo)
            for i, (x, y) in enumerate(pts):
                w.writerow([fid, idr, i, f"{x:.7f}", f"{y:.7f}"])
            n_pts += len(pts)

    # Impresión
    print(f"  Tramos en la capa                         : {total:>14,d}")
    print(f"  1. Tramos excluidos                       : {excluidos:>14,d}  ({100*excluidos/total:.2f} %)")
    print(f"       ESTATUS = Deshabilitado              : {motivo['ESTATUS']:>14,d}")
    print(f"       CONDICION = En construcción-cerrado  : {motivo['CONDICION']:>14,d}")
    print(f"       CIRCULA = Cerrada en ambos sentidos  : {motivo['CIRCULA']:>14,d}")
    print(f"     Tramos / aristas del grafo             : {G.number_of_edges():>14,d}")
    print(f"       en la componente principal           : {Gp.number_of_edges():>14,d}  "
          f"({100*Gp.number_of_edges()/G.number_of_edges():.2f} %)")
    print(f"     Nodos del grafo                        : {n_nodos:>14,d}")
    print(f"     Componentes conexas                    : {len(comps):>14,d}")
    print(f"  2. Nodos en la componente principal       : {n_princ:>14,d}")
    print(f"  3. % de nodos en la componente principal  : {pct:>13.2f} %")
    print(f"  4. Longitud total de la red del grafo (km): {long_total:>14,.1f}")
    print(f"       en la componente principal (km)      : {long_princ:>14,.1f}")
    print(f"       control: suma del campo LONGITUD (km): {long_campo:>14,.1f}  "
          f"(dif. {100*(long_total-long_campo)/long_campo:+.2f} %)")
    print(f"  5. Puntos de muestreo (cada <= {intervalo:g} m)    : {n_pts:>14,d}")
    print(f"       sobre {len(fids_princ):,d} tramos de la componente principal")
    print(f"     Referencia dirigida: nodos en la mayor componente fuertemente conexa: "
          f"{len(scc):,d} ({100*len(scc)/n_nodos:.2f} %)")
    print(f"     Segunda componente más grande: {len(comps[1]) if len(comps) > 1 else 0:,d} nodos")
    print(f"  Puntos guardados en: {ruta_pts}   [{time.time()-t0:.0f} s]")

    return {
        "capa": capa,
        "tramos_capa": total,
        "tramos_excluidos": excluidos,
        "excl_deshabilitado": motivo["ESTATUS"],
        "excl_en_construccion_cerrado": motivo["CONDICION"],
        "excl_cerrada_ambos_sentidos": motivo["CIRCULA"],
        "aristas_grafo": G.number_of_edges(),
        "aristas_componente_principal": Gp.number_of_edges(),
        "nodos_grafo": n_nodos,
        "componentes": len(comps),
        "nodos_componente_principal": n_princ,
        "pct_nodos_componente_principal": round(pct, 4),
        "longitud_grafo_km": round(long_total, 3),
        "longitud_componente_principal_km": round(long_princ, 3),
        "longitud_campo_LONGITUD_km": round(long_campo, 3),
        "intervalo_muestreo_m": intervalo,
        "puntos_muestreo": n_pts,
        "nodos_mayor_comp_fuerte_dirigida": len(scc),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpkg", type=Path, default=GPKG_DEFECTO)
    ap.add_argument("--capa", choices=CAPAS, action="append",
                    help="capa a procesar (por defecto, ambas)")
    ap.add_argument("--intervalo", type=float, default=INTERVALO_DEFECTO,
                    help="espaciado máximo entre puntos de muestreo, en metros (defecto 30)")
    args = ap.parse_args()

    if not args.gpkg.exists():
        raise SystemExit(f"No existe: {args.gpkg}")
    print(f"Fuente: {args.gpkg}")
    con = sqlite3.connect(f"file:{args.gpkg.as_posix()}?mode=ro", uri=True)

    resultados = [procesar_capa(con, c, args.intervalo) for c in (args.capa or CAPAS)]

    SALIDAS.mkdir(exist_ok=True)
    ruta = SALIDAS / "02_resumen_red_vial_edomex.csv"
    with open(ruta, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resultados[0].keys()))
        w.writeheader()
        w.writerows(resultados)
    print(f"\nResumen guardado en: {ruta}")


if __name__ == "__main__":
    main()
