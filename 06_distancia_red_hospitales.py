# -*- coding: utf-8 -*-
"""
Script 06 - Aislamiento por Inundaciones: distancia por red al hospital más cercano
==================================================================================

OBJETIVO
--------
Calcular, para cada localidad del Estado de México, la DISTANCIA MÍNIMA POR LA
RED VIAL (km) al hospital público más cercano y al hospital privado más
cercano, con el mismo método que usan las herramientas de Análisis de redes de
QGIS (algoritmo de Dijkstra), y reportar intervalo (mínimo-máximo), media y
mediana.

FUENTES
-------
  rnc2025Edomex.gpkg                 Red Nacional de Caminos 2025 (INEGI),
                                     capa 20km_red_vial_buffer (Edomex + 20 km)
                                     y polígono 20km_buffer.
  inegi\\iter_15csv20.zip             ITER 2020 (INEGI): localidades, coordenadas
                                     y población.
  ESTABLECIMIENTO_SALUD_202609.xlsx  CLUES (Secretaría de Salud): ubicación de
                                     los hospitales. Es necesaria porque ni la
                                     RNC ni el ITER contienen hospitales.

EQUIVALENCIA CON QGIS
---------------------
En QGIS el cálculo corresponde a:
  Caja de herramientas > Análisis de redes > "Ruta más corta (capa a punto)" o
  "Ruta más corta (punto a capa)" / "Área de servicio", con:
    - Capa de red: tramos de la RNC (filtrados como se indica abajo).
    - Tipo de ruta a calcular: "Más corta" (costo = longitud).
    - Dirección: "Ambas direcciones" (valor por defecto; no se usa campo de
      dirección).
    - Tolerancia de topología: 0.
QGIS hace internamente lo siguiente, y este script lo replica paso a paso:
  1. Construye un grafo con los vértices de las líneas: los vértices que
     coinciden se unen en un mismo nodo. En la RNC los tramos ya están
     cortados en cada intersección y las uniones vienen identificadas
     (UNION_INI / UNION_FIN), así que se usan esas claves: es el mismo grafo
     (se verificó que los extremos con la misma clave coinciden en > 99.98 %
     de los casos).
  2. "Ajusta" (snap) cada punto de origen y destino al PUNTO MÁS CERCANO
     SOBRE LA LÍNEA más cercana (no al nodo más cercano), partiendo ese tramo
     en dos.
  3. Ejecuta Dijkstra con costo = longitud del tramo. Con proyecto en grados
     (EPSG:6365) QGIS mide longitudes sobre el elipsoide; aquí también
     (GRS80, funciones del script 02).
  4. El costo que reporta QGIS es SOLO la parte recorrida sobre la red: no
     incluye la distancia recta del punto a la línea. Este script reporta esa
     distancia por red (columna principal, igual que QGIS) y, por separado,
     la distancia de ajuste y el total (ajuste localidad + red + ajuste
     hospital), por si se quiere considerar el tramo fuera de la red.
Hacerlo en Python en lugar de QGIS permite resolver las 4,894 localidades y
todos los hospitales de una sola vez (Dijkstra multi-origen), cosa que en QGIS
requeriría correr la herramienta localidad por localidad o hospital por
hospital.

CRITERIOS (explícitos para que puedan debatirse)
-----------------------------------------------
  Red vial
    - Se excluyen tramos con ESTATUS = 'Deshabilitado', CONDICION = 'En
      construcción - cerrado' o CIRCULA = 'Cerrada en ambos sentidos' (mismos
      criterios del script 02).
    - Se usa la componente conexa principal del grafo (99.8 % de los nodos):
      un punto ajustado a una "isla" de la red no tendría ruta a ningún
      hospital. Los puntos se ajustan a la línea más cercana DE ESA
      componente.
    - Capa de 20 km: permite rutas que salen del Edomex y regresan, y llegar
      a hospitales de la CDMX y estados vecinos.
  Hospitales (CLUES; mismos criterios del script 05)
    - 'NOMBRE TIPO ESTABLECIMIENTO' = 'DE HOSPITALIZACIÓN' y 'ESTATUS DE
      OPERACION' = 'EN OPERACION', con coordenadas.
    - Privado = institución 'SERVICIOS MEDICOS PRIVADOS'; público = el resto.
    - Se usan los hospitales de cualquier entidad que caen dentro del
      polígono 20km_buffer: la población cerca del límite estatal puede
      atenderse al otro lado.
  Localidades (ITER; mismos criterios del script 04)
    - Se excluyen los renglones agregados (LOC 0000, 9998, 9999). Se calculan
      todas las localidades con coordenadas y se reporta también el
      subconjunto rural (< 2,500 hab.).
  Estadísticos
    - Intervalo = [mínimo, máximo]; media y mediana por localidad (cada
      localidad cuenta igual).
    - Ponderados por habitantes (POBTOT): media ponderada = suma(d*pob)/suma(pob)
      y mediana ponderada = distancia d tal que la mitad de los habitantes
      está a d km o menos. Describen la distancia que recorre la PERSONA
      típica, no la localidad típica. El intervalo no cambia al ponderar.
      Localidades con POBTOT no numérico (confidencialidad) pesan 0.

MÉTODO DE AJUSTE (detalle)
--------------------------
  - Para buscar la línea más cercana se trabaja en metros, en la Cónica
    Conforme de Lambert de INEGI (función lcc del script 05).
  - Cada segmento (par de vértices consecutivos) se subdivide virtualmente en
    piezas de <= 50 m; sus centros forman un índice espacial (KD-tree). Para
    cada punto se toman las K piezas más cercanas y se calcula la proyección
    EXACTA del punto sobre cada segmento padre; se elige la menor distancia.
    Con piezas de 50 m y K = 32, el segmento verdaderamente más cercano
    siempre está entre los candidatos salvo casos patológicos.
  - La posición del punto ajustado se expresa como fracción f de la longitud
    del tramo (0 = UNION_INI, 1 = UNION_FIN) y se convierte a metros
    elipsoidales: s = f * L.

CÁLCULO DE LA DISTANCIA (detalle)
---------------------------------
  - Dijkstra multi-origen: se agrega un nodo ficticio S unido a los dos
    extremos del tramo donde cae cada hospital, con peso igual a la distancia
    del hospital ajustado a cada extremo (f*L y (1-f)*L). La distancia de S a
    cualquier nodo es la distancia por red al hospital más cercano.
  - Para una localidad ajustada en el tramo (u, v) a la posición s:
        d = min( d(u) + s , d(v) + (L - s) )
    y, si en ese mismo tramo hay un hospital en s_h, también |s - s_h|.
  - Entre dos nodos con tramos paralelos Dijkstra usa el más corto.

USO
---
  python 06_distancia_red_hospitales.py
  python 06_distancia_red_hospitales.py --gpkg RUTA --iter RUTA --clues RUTA

REQUISITOS
----------
  numpy, pandas, networkx, scipy, matplotlib, openpyxl, Pillow.
  En la misma carpeta: 02_red_vial_edomex.py, 04_localidades_iter_edomex.py y
  05_inundacion_caminos_hospitales.py (se reutilizan sus funciones).

SALIDAS (carpeta salidas\\)
--------------------------
  06_distancia_localidades_hospital.csv   una fila por localidad: distancia por
      red (km) al hospital público y privado más cercano, distancia de ajuste
      (m), total, CLUES del hospital más cercano y bandera rural.
  06_resumen_distancias.csv               intervalo, media y mediana.
"""

import argparse
import importlib.util
import math
import sqlite3
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parent                                          # D:\ARTUAEM
SALIDAS = AQUI / "salidas"


def _importar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


red02 = _importar("red02", "02_red_vial_edomex.py")         # lectura GPKG, longitudes, exclusiones
loc04 = _importar("loc04", "04_localidades_iter_edomex.py") # lectura ITER, coordenadas DMS
s05 = _importar("s05", "05_inundacion_caminos_hospitales.py")  # proyección LCC, polígono buffer

RUTAS = {
    "gpkg": BASE / "rnc2025Edomex.gpkg",
    "iter": BASE / "inegi" / "iter_15csv20.zip",
    "clues": BASE / "ESTABLECIMIENTO_SALUD_202609.xlsx",
}
CAPA_RED = "20km_red_vial_buffer"
CAPA_BUFFER = "20km_buffer"
PIEZA_M = 50.0        # tamaño máximo de las piezas del índice espacial (m)
K_CANDIDATOS = 32     # segmentos candidatos evaluados por punto


# ============================================================================
# 1. Red vial: tramos válidos, grafo y componente principal
# ============================================================================
def cargar_red(ruta_gpkg):
    """Lee los tramos válidos y devuelve listas paralelas por tramo.

    Regresa un diccionario con:
      u, v     : uniones inicial y final (nodos del grafo)
      L        : longitud elipsoidal del tramo (m)
      partes   : lista de arreglos lon/lat de cada parte de la geometría
      excluidos: número de tramos excluidos
    """
    con = sqlite3.connect(f"file:{Path(ruta_gpkg).as_posix()}?mode=ro", uri=True)
    E = red02.EXCLUSION
    u, v, L, partes, excl = [], [], [], [], 0
    for est, cond, circ, ui, vi, g in con.execute(
            f'SELECT ESTATUS, CONDICION, CIRCULA, UNION_INI, UNION_FIN, geom FROM "{CAPA_RED}"'):
        if est in E["ESTATUS"] or cond in E["CONDICION"] or circ in E["CIRCULA"]:
            excl += 1
            continue
        p = [q for q in red02.gpkg_a_lineas(g) if len(q) > 1]
        if not p:
            continue
        u.append(ui)
        v.append(vi)
        L.append(float(sum(red02.longitudes_segmentos(q).sum() for q in p)))
        partes.append(p)
    buf = con.execute(f'SELECT geom FROM "{CAPA_BUFFER}"').fetchone()[0]
    return {"u": np.array(u), "v": np.array(v), "L": np.array(L), "partes": partes,
            "excluidos": excl, "buffer": s05.anillos_gpkg_poligono(buf)}


def grafo_principal(red):
    """Grafo no dirigido (costo = longitud mínima entre tramos paralelos) y
    máscara de tramos que pertenecen a su componente conexa principal."""
    G = nx.Graph()
    for ui, vi, Li in zip(red["u"], red["v"], red["L"]):
        if ui == vi:                    # lazos: no aportan rutas
            continue
        if not G.has_edge(ui, vi) or G[ui][vi]["w"] > Li:
            G.add_edge(ui, vi, w=Li)
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()
    en_principal = np.array([(a in comp) and (a != b) for a, b in zip(red["u"], red["v"])])
    return G, en_principal


# ============================================================================
# 2. Índice espacial de segmentos y ajuste de puntos a la línea más cercana
# ============================================================================
class IndiceSegmentos:
    """Ajuste punto -> punto más cercano sobre la red (como QGIS)."""

    def __init__(self, red, mascara_tramos):
        x0, y0, x1, y1, tramo, s0 = [], [], [], [], [], []
        self.L_lcc = np.zeros(len(red["L"]))
        for i in np.flatnonzero(mascara_tramos):
            acum = 0.0
            for q in red["partes"][i]:
                X, Y = s05.lcc(q[:, 0], q[:, 1])
                dl = np.hypot(np.diff(X), np.diff(Y))
                x0.append(X[:-1]); y0.append(Y[:-1]); x1.append(X[1:]); y1.append(Y[1:])
                tramo.append(np.full(len(dl), i))
                s0.append(acum + np.r_[0.0, np.cumsum(dl)[:-1]])
                acum += dl.sum()
            self.L_lcc[i] = acum
        self.x0, self.y0 = np.concatenate(x0), np.concatenate(y0)
        self.x1, self.y1 = np.concatenate(x1), np.concatenate(y1)
        self.tramo, self.s0 = np.concatenate(tramo), np.concatenate(s0)
        self.len = np.hypot(self.x1 - self.x0, self.y1 - self.y0)

        # Piezas virtuales de <= PIEZA_M m para el KD-tree
        k = np.maximum(1, np.ceil(self.len / PIEZA_M)).astype(np.int64)
        seg = np.repeat(np.arange(len(k)), k)
        pos = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k) + 0.5) / np.repeat(k, k)
        px = self.x0[seg] + pos * (self.x1[seg] - self.x0[seg])
        py = self.y0[seg] + pos * (self.y1[seg] - self.y0[seg])
        self.pieza_seg = seg
        self.arbol = cKDTree(np.column_stack([px, py]))

    def ajustar(self, x, y):
        """Devuelve (tramo, fracción f en [0,1], distancia de ajuste en m)."""
        pts = np.column_stack([x, y])
        _, idx = self.arbol.query(pts, k=K_CANDIDATOS)
        cand = self.pieza_seg[idx]                                  # (n, K)
        ax, ay = self.x0[cand], self.y0[cand]
        bx, by = self.x1[cand], self.y1[cand]
        dx, dy = bx - ax, by - ay
        l2 = np.where(dx * dx + dy * dy == 0, 1e-12, dx * dx + dy * dy)
        t = np.clip(((pts[:, :1] - ax) * dx + (pts[:, 1:] - ay) * dy) / l2, 0, 1)
        d = np.hypot(ax + t * dx - pts[:, :1], ay + t * dy - pts[:, 1:])
        j = np.argmin(d, axis=1)
        r = np.arange(len(pts))
        seg = cand[r, j]
        tr = self.tramo[seg]
        f = (self.s0[seg] + t[r, j] * self.len[seg]) / self.L_lcc[tr]
        return tr, np.clip(f, 0, 1), d[r, j]


# ============================================================================
# 3. Dijkstra multi-origen desde los hospitales
# ============================================================================
def distancias_red(G, red, hosp_tramo, hosp_f, loc_tramo, loc_f, hosp_ids):
    """Distancia por red (m) de cada localidad al hospital más cercano y el
    identificador de ese hospital."""
    S = "__S__"
    g2 = G.copy()
    U, V, L = red["u"], red["v"], red["L"]
    # Conectar el origen ficticio a ambos extremos del tramo de cada hospital.
    # Se guarda qué hospital da la conexión más corta a cada extremo.
    via = {}
    for tr, f, hid in zip(hosp_tramo, hosp_f, hosp_ids):
        for nodo, w in ((U[tr], f * L[tr]), (V[tr], (1 - f) * L[tr])):
            if not g2.has_edge(S, nodo) or g2[S][nodo]["w"] > w:
                g2.add_edge(S, nodo, w=w)
                via[nodo] = hid
    dist, rutas = nx.single_source_dijkstra(g2, S, weight="w")
    origen = {n: via[r[1]] for n, r in rutas.items() if len(r) > 1}

    dU = np.array([dist.get(n, np.inf) for n in U[loc_tramo]])
    dV = np.array([dist.get(n, np.inf) for n in V[loc_tramo]])
    s = loc_f * L[loc_tramo]
    a, b = dU + s, dV + (L[loc_tramo] - s)
    d = np.minimum(a, b)
    hid = np.where(a <= b, [origen.get(n) for n in U[loc_tramo]], [origen.get(n) for n in V[loc_tramo]])

    # Hospital en el mismo tramo que la localidad: distancia directa sobre el tramo
    mismo = {}
    for tr, f, h in zip(hosp_tramo, hosp_f, hosp_ids):
        mismo.setdefault(tr, []).append((f, h))
    for i, tr in enumerate(loc_tramo):
        for f, h in mismo.get(tr, []):
            dd = abs(loc_f[i] - f) * L[tr]
            if dd < d[i]:
                d[i], hid[i] = dd, h
    return d, hid


# ============================================================================
# 4. Estadísticos
# ============================================================================
def estadisticos(d_km, pob):
    """Intervalo, media y mediana por localidad y ponderadas por habitantes.

    La mediana ponderada es el primer valor de d (ordenado) en el que la
    población acumulada alcanza el 50 % del total.
    """
    ok = np.isfinite(d_km)
    x, w = d_km[ok], pob[ok]
    o = np.argsort(x)
    acum = np.cumsum(w[o]) / w.sum()
    return {"n": int(ok.sum()), "habitantes": int(w.sum()), "sin_ruta": int((~ok).sum()),
            "min_km": x.min(), "max_km": x.max(),
            "media_km": x.mean(), "mediana_km": float(np.median(x)),
            "media_ponderada_hab_km": float(np.average(x, weights=w)),
            "mediana_ponderada_hab_km": float(x[o][np.searchsorted(acum, 0.5)])}


# ============================================================================
# 5. Programa principal
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUTAS.items():
        ap.add_argument(f"--{k}", type=Path, default=v)
    a = ap.parse_args()
    for k in RUTAS:
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()

    print("1/5 Leyendo red vial...")
    red = cargar_red(a.gpkg)
    G, en_principal = grafo_principal(red)

    print("2/5 Construyendo índice espacial de segmentos...")
    idx = IndiceSegmentos(red, en_principal)

    print("3/5 Leyendo localidades (ITER) y hospitales (CLUES)...")
    it = loc04.leer_iter(a.iter)
    it = it[~it["LOC"].isin(loc04.LOC_AGREGADOS)].copy()
    it["pob"] = pd.to_numeric(it["POBTOT"], errors="coerce").fillna(0)
    it["lon"] = it["LONGITUD"].map(loc04.dms_a_decimal)
    it["lat"] = it["LATITUD"].map(loc04.dms_a_decimal)
    it = it[it["lon"].notna() & it["lat"].notna()].reset_index(drop=True)
    it["CVEGEO"] = it["ENTIDAD"] + it["MUN"] + it["LOC"]
    it["rural"] = it["pob"] < loc04.UMBRAL_RURAL
    it["x"], it["y"] = s05.lcc(it["lon"], it["lat"])

    cl = pd.read_excel(a.clues, sheet_name=0, dtype=str)
    h = cl[(cl["NOMBRE TIPO ESTABLECIMIENTO"] == "DE HOSPITALIZACIÓN")
           & (cl["ESTATUS DE OPERACION"] == "EN OPERACION")].copy()
    h["lat"] = pd.to_numeric(h["LATITUD"], errors="coerce")
    h["lon"] = pd.to_numeric(h["LONGITUD"], errors="coerce")
    h = h[h["lat"].notna() & h["lon"].notna()].copy()
    h["tipo"] = np.where(h["NOMBRE DE LA INSTITUCION"] == "SERVICIOS MEDICOS PRIVADOS", "Privado", "Público")
    h["x"], h["y"] = s05.lcc(h["lon"], h["lat"])
    dentro = np.zeros(len(h), bool)
    for anillo in red["buffer"]:
        dentro |= MplPath(anillo).contains_points(h[["x", "y"]].to_numpy())
    h = h[dentro].reset_index(drop=True)

    print("4/5 Ajustando puntos a la red...")
    it["tramo"], it["f"], it["ajuste_m"] = idx.ajustar(it["x"], it["y"])
    h["tramo"], h["f"], h["ajuste_m"] = idx.ajustar(h["x"], h["y"])

    print("5/5 Dijkstra multi-origen...")
    out = it[["CVEGEO", "NOM_MUN", "NOM_LOC", "lon", "lat", "POBTOT", "rural", "ajuste_m"]].copy()
    out = out.rename(columns={"ajuste_m": "ajuste_localidad_m"})
    resumen = []
    pob = it["pob"].to_numpy()
    for tipo in ("Público", "Privado"):
        hs = h[h["tipo"] == tipo]
        d, hid = distancias_red(G, red, hs["tramo"].to_numpy(), hs["f"].to_numpy(),
                                it["tramo"].to_numpy(), it["f"].to_numpy(), hs["CLUES"].to_numpy())
        aj_h = hs.set_index("CLUES")["ajuste_m"]
        aj_h = aj_h[~aj_h.index.duplicated()]
        clave = tipo.lower().replace("ú", "u")
        out[f"km_red_{clave}"] = np.round(d / 1000, 4)
        out[f"clues_{clave}"] = hid
        out[f"ajuste_hospital_{clave}_m"] = np.round(aj_h.reindex(hid).to_numpy(), 1)
        out[f"km_total_{clave}"] = np.round(
            (d + it["ajuste_m"].to_numpy() + aj_h.reindex(hid).fillna(0).to_numpy()) / 1000, 4)
        for nombre, m in (("Todas", np.ones(len(it), bool)), ("Rurales (< 2,500 hab.)", it["rural"].to_numpy())):
            e = estadisticos(d[m] / 1000, pob[m])
            resumen.append({"hospital": tipo, "localidades": nombre, **e})
    out["ajuste_localidad_m"] = out["ajuste_localidad_m"].round(1)

    # ------------------------------------------------------------------ salida
    print("\n" + "=" * 78)
    print(" DISTANCIA POR RED AL HOSPITAL MÁS CERCANO (Dijkstra, costo = longitud)")
    print("=" * 78)
    print(f"  Tramos válidos: {len(red['L']):,d} (excluidos {red['excluidos']:,d}); "
          f"nodos de la componente principal: {G.number_of_nodes():,d}")
    print(f"  Hospitales dentro de 20km_buffer: {(h.tipo == 'Público').sum():,d} públicos, "
          f"{(h.tipo == 'Privado').sum():,d} privados")
    print(f"  Localidades: {len(it):,d} (rurales {int(it.rural.sum()):,d})")
    print(f"  Distancia de ajuste localidad->red: mediana {it.ajuste_m.median():.0f} m, "
          f"máx {it.ajuste_m.max():,.0f} m | hospital->red: mediana {h.ajuste_m.median():.0f} m, "
          f"máx {h.ajuste_m.max():,.0f} m")
    print(f"\n  {'':34s}{'':>12s}{'':>18s}{'Por localidad':>17s}{'Por habitante':>18s}")
    print(f"  {'Hospital':9s} {'Localidades':24s}{'n':>6s}{'Habitantes':>12s}  {'Intervalo (km)':>16s}"
          f"{'Media':>8s}{'Mediana':>9s}{'Media':>9s}{'Mediana':>9s}")
    for r in resumen:
        print(f"  {r['hospital']:9s} {r['localidades']:24s}{r['n']:>6,d}{r['habitantes']:>12,d}  "
              f"{r['min_km']:>6.2f} - {r['max_km']:>7.2f}{r['media_km']:>8.2f}{r['mediana_km']:>9.2f}"
              f"{r['media_ponderada_hab_km']:>9.2f}{r['mediana_ponderada_hab_km']:>9.2f}")
        if r["sin_ruta"]:
            print(f"            ({r['sin_ruta']} localidades sin ruta)")
    print("  (km sobre la red, como el costo de QGIS; 'por habitante' = ponderado por POBTOT)")

    SALIDAS.mkdir(exist_ok=True)
    out.to_csv(SALIDAS / "06_distancia_localidades_hospital.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(resumen).round(4).to_csv(SALIDAS / "06_resumen_distancias.csv", index=False, encoding="utf-8-sig")
    print(f"\nArchivos guardados en: {SALIDAS}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
