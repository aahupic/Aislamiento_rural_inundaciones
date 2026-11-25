# -*- coding: utf-8 -*-
"""
Script 11 - Aislamiento por Inundaciones: tramos críticos por cierre individual
==============================================================================

OBJETIVO
--------
Identificar los tramos inundables cuyo cierre INDIVIDUAL (uno a la vez, con
el resto de la red abierta) obliga a la población rural a desviarse 15
minutos o más para llegar al hospital público más cercano, o la deja sin
ruta. Para cada tramo crítico se reporta:
  - nombre de la vialidad (y tipo y código de la RNC),
  - municipio,
  - tiempo de desvío máximo (min; "aislamiento" si alguna localidad queda
    sin ruta),
  - habitantes rurales afectados (desvío >= 15 min o aislamiento).

PARÁMETROS DE ENTRADA
---------------------
  --intervalo   espaciado de los puntos de muestreo sobre los tramos (m);
                por defecto el del script 03 (15 m).
  --umbral      % de la longitud en zona inundable para que un tramo sea
                CANDIDATO a cierre (candidato si % > umbral); defecto 50.
  --desvio      desvío mínimo para considerar a una localidad afectada y a un
                tramo crítico (min); defecto 15.
  Hospitales    públicos ('DE HOSPITALIZACIÓN', 'EN OPERACION', institución
                distinta de 'SERVICIOS MEDICOS PRIVADOS'), dentro del
                polígono 20km_buffer (incluye CDMX y estados vecinos).
  Localidades   rurales (< 2,500 habitantes) del ITER 2020.

FUENTES
-------
  rnc2025Edomex.gpkg                 red vial (20km_red_vial_buffer): topología,
                                     velocidad, NOMBRE, TIPO_VIAL y CODIGO.
  CENAPRED.zip -> CENAPRED_IPI_2015_Inund_Munici.zip
                                     municipios: se asigna a cada tramo el
                                     municipio donde cae su punto medio.
  pub-aislamiento\\data\\raw\\tr100_prcent.tif
                                     zonas inundables TR100 (0 = inundable).
  inegi\\iter_15csv20.zip             localidades rurales y población.
  ESTABLECIMIENTO_SALUD_202609.xlsx  CLUES: hospitales públicos.
  03_puntos_muestreo_15m.py          intervalo de muestreo y exclusiones
                                     (a través del script 07).

MÉTODO
------
 1. Red, % inundado y tiempos de tramo: idénticos al script 07 (tramos
    válidos de la componente principal; % inundado con puntos cada 15 m;
    tiempo = longitud / VELOCIDAD RNC; entre tramos paralelos manda el más
    rápido; localidades y hospitales al nodo más cercano, conexión a 5 km/h).
 2. Situación base: Dijkstra multi-origen desde todos los hospitales
    públicos sobre la red completa (nodo ficticio unido al nodo de cada
    hospital con su tiempo de conexión). Se guarda, para cada nodo, su
    tiempo d(n) y su PREDECESOR en la ruta más corta. Los predecesores
    forman el ÁRBOL DE RUTAS MÁS CORTAS.
 3. Observación clave que hace el cálculo factible: cerrar un tramo solo
    cambia el tiempo de los nodos cuya ruta óptima pasa por ese tramo, es
    decir, los nodos del SUBÁRBOL que cuelga de él. Por lo tanto:
      - un tramo que no pertenece al árbol no altera a nadie;
      - un tramo del árbol cuyo subárbol no contiene ninguna localidad
        rural no afecta a la población estudiada.
    Solo se evalúan los tramos candidatos (> umbral inundable) que están en
    el árbol y tienen localidades rurales colgando. El resultado es exacto:
    equivale a correr Dijkstra completo con cada tramo cerrado, pero miles
    de veces más rápido (se comprueba con --verificar).
 4. Para cada tramo evaluado (p -> c, con c el nodo hijo):
      a. Subárbol T = c y todos sus descendientes.
      b. Si el tramo tiene un paralelo, la arista sigue existiendo con el
         tiempo del siguiente paralelo más rápido; si no, desaparece.
      c. Nuevo tiempo de los nodos de T: Dijkstra restringido a T, sembrado
         con min( d(x) + w(x, y) ) para cada arista que entra a T desde un
         nodo x fuera de T (cuyo tiempo no cambia; incluye la conexión
         directa del nodo ficticio de hospitales).
      d. Desvío de cada localidad rural de T = nuevo tiempo - tiempo base
         (infinito si no hay ruta: aislamiento).
 5. Un tramo es CRÍTICO si alguna localidad rural tiene desvío >= 15 min o
    queda aislada. Habitantes afectados = suma de POBTOT de esas localidades.
 6. Municipio del tramo: municipio (CENAPRED IPI 2015) que contiene el punto
    medio del tramo; "Fuera del Edomex" si cae en la franja de 20 km.

CRITERIOS QUE CONVIENE DEBATIR
------------------------------
  - Análisis de un solo cierre (N-1): no considera que otros tramos
    inundables se cierren a la vez (eso lo hacen los scripts 07 y 10). Un
    tramo puede no ser crítico aquí y sí serlo combinado con otros.
  - Varios tramos consecutivos de un mismo camino suelen aparecer juntos con
    el mismo efecto (cerrar cualquiera corta la misma ruta). Se reportan por
    separado porque son tramos distintos de la RNC.
  - Con empates exactos en el tiempo base, el árbol toma una de las rutas
    óptimas; el efecto es despreciable con tiempos continuos.
  - Tramos sin nombre en la RNC aparecen como 'N/D' o 'N/A' (valor original).

USO
---
  python 11_tramos_criticos_cierre_individual.py
  python 11_tramos_criticos_cierre_individual.py --umbral 50 --desvio 15 --intervalo 15
  python 11_tramos_criticos_cierre_individual.py --verificar 10   (compara 10 tramos con Dijkstra completo)

REQUISITOS
----------
  numpy, pandas, networkx, scipy, matplotlib, Pillow, openpyxl.
  En la misma carpeta: scripts 02, 03, 04, 05, 07 y 08.

SALIDAS (carpeta salidas\\)
--------------------------
  11_tramos_criticos.csv                 un renglón por tramo crítico,
                                         ordenado por habitantes afectados.
  11_tramos_criticos_por_municipio.csv   resumen por municipio.
"""

import argparse
import heapq
import importlib.util
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

AQUI = Path(__file__).resolve().parent
SALIDAS = AQUI / "salidas"


def _importar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


red02 = _importar("red02", "02_red_vial_edomex.py")
loc04 = _importar("loc04", "04_localidades_iter_edomex.py")
s05 = _importar("s05", "05_inundacion_caminos_hospitales.py")
s07 = _importar("s07", "07_aislamiento_por_umbral_inundacion.py")   # usa 02 y 03
s08 = _importar("s08", "08_mapa_tramos_inundables_hospitales.py")   # lectura de municipios

RUTAS = dict(s07.RUTAS)
S = "__HOSPITALES__"          # nodo ficticio de origen


# ============================================================================
# Utilidades
# ============================================================================
def etiquetas_municipio(R, muns):
    """Ráster (alineado al TR100) con el índice 1..n del municipio, 0 fuera."""
    img = Image.new("I", (R.w, R.h), 0)
    dr = ImageDraw.Draw(img)
    for i, (_, _, g) in enumerate(muns, start=1):
        for r in g:
            dr.polygon([((x - R.x0) / R.sx, (R.y0 - y) / R.sy) for x, y in r], fill=i)
    return np.asarray(img)


def texto_utf8(t):
    """El .dbf del IPI está en UTF-8 pero el lector común (script 05) lo
    decodifica como latin-1 ('ZumpahuacÃ¡n'); aquí se corrige."""
    try:
        return t.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return t


def subarbol(hijos, raiz):
    """Nodos del subárbol que cuelga de 'raiz' (incluida)."""
    out, pila = [], [raiz]
    while pila:
        n = pila.pop()
        out.append(n)
        pila.extend(hijos.get(n, ()))
    return out


def dijkstra_subarbol(G, nodos_T, d_base, arista_cerrada, w_nueva):
    """Nuevo tiempo de los nodos del subárbol T con la arista (p, c) cerrada.

    G incluye el nodo ficticio de hospitales. Los nodos fuera de T conservan
    su tiempo base; cada nodo de T se siembra con la mejor llegada desde fuera
    de T y luego se propaga dentro de T.
    """
    T = set(nodos_T)
    cerrada = frozenset(arista_cerrada)

    def peso(x, y, attr):
        if frozenset((x, y)) == cerrada:
            return w_nueva                 # None = sin paralelo: arista eliminada
        return attr["t"]

    dist, heap = {}, []
    for y in T:
        mejor = np.inf
        for x, attr in G.adj[y].items():
            if x in T or x not in d_base:
                continue
            w = peso(x, y, attr)
            if w is not None and d_base[x] + w < mejor:
                mejor = d_base[x] + w
        if mejor < np.inf:
            dist[y] = mejor
            heap.append((mejor, y))
    heapq.heapify(heap)
    hecho = set()
    while heap:
        dy, y = heapq.heappop(heap)
        if y in hecho:
            continue
        hecho.add(y)
        for z, attr in G.adj[y].items():
            if z not in T or z in hecho:
                continue
            w = peso(y, z, attr)
            if w is not None and dy + w < dist.get(z, np.inf):
                dist[z] = dy + w
                heapq.heappush(heap, (dy + w, z))
    return dist


# ============================================================================
# Programa principal
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUTAS.items():
        ap.add_argument(f"--{k}", type=Path, default=v)
    ap.add_argument("--intervalo", type=float, default=s07.m03.INTERVALO, help="muestreo sobre tramos (m)")
    ap.add_argument("--umbral", type=float, default=50.0, help="%% inundado para ser candidato (> umbral)")
    ap.add_argument("--desvio", type=float, default=15.0, help="desvío mínimo (min)")
    ap.add_argument("--vel-conexion", type=float, default=5.0, help="km/h del tramo de conexión")
    ap.add_argument("--verificar", type=int, default=0,
                    help="comprobar N tramos críticos con Dijkstra completo (lento, ~3 s c/u)")
    a = ap.parse_args()
    for k in RUTAS:
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()
    SALIDAS.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- 1. Red
    print("1/5 Ráster, municipios y % inundado por tramo...")
    R = s05.Raster(a.tr100)
    inund = R.a == s05.VALOR_INUNDABLE
    muns, _ = s08.municipios(a.cenapred)
    m_edomex = R.rasterizar([r for _, _, g in muns for r in g])
    etiq = etiquetas_municipio(R, muns)
    con = sqlite3.connect(f"file:{a.gpkg.as_posix()}?mode=ro", uri=True)
    buffer = s05.anillos_gpkg_poligono(con.execute(f'SELECT geom FROM "{s07.CAPA_BUFFER}"').fetchone()[0])
    tr = s07.tramos_y_porcentaje(con, R, inund, m_edomex, a.intervalo)
    tr["candidato"] = tr["pct_inund"] > a.umbral

    G = s07.grafo(tr, np.ones(len(tr), bool))
    tramos_arista = defaultdict(list)                  # (a, b) -> [(t_min, índice)] ordenados
    for i, (u, v, t) in enumerate(zip(tr.u, tr.v, tr.t_min)):
        if u != v:
            tramos_arista[(min(u, v), max(u, v))].append((t, i))
    for k in tramos_arista:
        tramos_arista[k].sort()

    # ---------------------------------------------------------------- 2. Base
    print("2/5 Localidades rurales, hospitales públicos y árbol de rutas más cortas...")
    coord = pd.concat([
        tr[["u", "lon_ini", "lat_ini"]].set_axis(["n", "lon", "lat"], axis=1),
        tr[["v", "lon_fin", "lat_fin"]].set_axis(["n", "lon", "lat"], axis=1)]).drop_duplicates("n")
    cx, cy = s05.lcc(coord.lon, coord.lat)
    arbol_kd, nodos = cKDTree(np.column_stack([cx, cy])), coord["n"].to_numpy()
    vcon = a.vel_conexion / 3.6 * 60

    def asignar(x, y):
        d, j = arbol_kd.query(np.column_stack([x, y]))
        return nodos[j], d / vcon

    it = loc04.leer_iter(a.iter)
    it = it[~it["LOC"].isin(loc04.LOC_AGREGADOS)].copy()
    it["pob"] = pd.to_numeric(it["POBTOT"], errors="coerce").fillna(0)
    it["lon"] = it["LONGITUD"].map(loc04.dms_a_decimal)
    it["lat"] = it["LATITUD"].map(loc04.dms_a_decimal)
    it = it[it["lon"].notna() & it["lat"].notna() & (it["pob"] < loc04.UMBRAL_RURAL)].reset_index(drop=True)
    it["nodo"], it["con_min"] = asignar(*s05.lcc(it["lon"], it["lat"]))

    cl = pd.read_excel(a.clues, sheet_name=0, dtype=str)
    h = cl[(cl["NOMBRE TIPO ESTABLECIMIENTO"] == "DE HOSPITALIZACIÓN")
           & (cl["ESTATUS DE OPERACION"] == "EN OPERACION")
           & (cl["NOMBRE DE LA INSTITUCION"] != "SERVICIOS MEDICOS PRIVADOS")].copy()
    h["lat"] = pd.to_numeric(h["LATITUD"], errors="coerce")
    h["lon"] = pd.to_numeric(h["LONGITUD"], errors="coerce")
    h = h[h["lat"].notna() & h["lon"].notna()].copy()
    hx, hy = s05.lcc(h["lon"], h["lat"])
    dentro = np.zeros(len(h), bool)
    for anillo in buffer:
        dentro |= MplPath(anillo).contains_points(np.column_stack([hx, hy]))
    h = h[dentro].copy()
    h["nodo"], h["con_min"] = asignar(*s05.lcc(h["lon"], h["lat"]))

    g2 = G.copy()                                       # red + nodo ficticio de hospitales
    for n, c_ in h.groupby("nodo")["con_min"].min().items():
        g2.add_edge(S, n, t=c_)
    pred, d_base = nx.dijkstra_predecessor_and_distance(g2, S, weight="t")
    padre = {n: p[0] for n, p in pred.items() if p}
    hijos = defaultdict(list)
    for n, p in padre.items():
        hijos[p].append(n)

    loc_por_nodo = defaultdict(list)
    for i, n in enumerate(it["nodo"]):
        loc_por_nodo[n].append(i)
    con_loc = it["con_min"].to_numpy()
    t_base_loc = np.array([d_base.get(n, np.inf) for n in it["nodo"]]) + con_loc

    # ---------------------------------------------------------------- 3. Tramos a evaluar
    print("3/5 Tramos candidatos en el árbol con localidades rurales colgando...")
    con_rural = set()                                   # nodos con alguna rural en su subárbol
    for n in loc_por_nodo:
        while n is not None and n != S and n not in con_rural and n in d_base:
            con_rural.add(n)
            n = padre.get(n)
    evaluar = []                                        # (índice tramo, padre, hijo, peso nuevo)
    for c in con_rural:
        p = padre.get(c)
        if p is None or p == S:
            continue
        lista = tramos_arista[(min(p, c), max(p, c))]
        i_rapido = lista[0][1]                          # solo el más rápido altera el tiempo
        if tr.at[i_rapido, "candidato"]:
            evaluar.append((i_rapido, p, c, lista[1][0] if len(lista) > 1 else None))
    print(f"    candidatos (> {a.umbral:g} % inundable): {int(tr.candidato.sum()):,d}; "
          f"a evaluar: {len(evaluar):,d}")

    # ---------------------------------------------------------------- 4. Cierres individuales
    print("4/5 Evaluando cierres individuales...")
    pob = it["pob"].to_numpy()
    nodo_loc = it["nodo"].to_numpy()
    filas, cierres = [], {}
    for k, (i_tr, p, c, w_nueva) in enumerate(evaluar, 1):
        T = subarbol(hijos, c)
        idx = np.array([j for n in T for j in loc_por_nodo.get(n, ())], dtype=int)
        if not len(idx):
            continue
        nuevo = dijkstra_subarbol(g2, T, d_base, (p, c), w_nueva)
        t_new = np.array([nuevo.get(nodo_loc[j], np.inf) for j in idx]) + con_loc[idx]
        desv = t_new - t_base_loc[idx]
        afect = desv >= a.desvio
        if afect.any():
            ais = ~np.isfinite(desv)
            fin = desv[afect & ~ais]
            filas.append({"indice": i_tr,
                          "desvio_max_num": np.inf if ais.any() else float(desv.max()),
                          "desvio_max_finito_min": round(float(fin.max()), 1) if len(fin) else np.nan,
                          "habitantes_afectados": int(pob[idx][afect].sum()),
                          "localidades_afectadas": int(afect.sum()),
                          "habitantes_aislados": int(pob[idx][ais].sum()),
                          "localidades_aisladas": int(ais.sum())})
            cierres[i_tr] = (p, c, w_nueva, idx, t_new)
        if k % 2000 == 0:
            print(f"    {k:,d} / {len(evaluar):,d}  [{time.time() - t0:.0f} s]")

    res = pd.DataFrame(filas)
    if res.empty:
        print("Ningún tramo crítico.")
        return

    # Verificación opcional contra Dijkstra completo
    if a.verificar:
        print(f"    Verificando {a.verificar} tramos con Dijkstra completo...")
        errores = 0
        for i_tr in res.sort_values("habitantes_afectados", ascending=False)["indice"].head(a.verificar):
            p, c, w_nueva, idx, t_new = cierres[i_tr]
            g3 = g2.copy()
            if w_nueva is None:
                g3.remove_edge(p, c)
            else:
                g3[p][c]["t"] = w_nueva
            d3 = nx.single_source_dijkstra_path_length(g3, S, weight="t")
            ref = np.array([d3.get(nodo_loc[j], np.inf) for j in idx]) + con_loc[idx]
            ok = np.allclose(np.nan_to_num(ref, posinf=1e9), np.nan_to_num(t_new, posinf=1e9), atol=1e-6)
            errores += not ok
        print(f"    Verificación: {a.verificar - errores} de {a.verificar} coinciden exactamente.")

    # ---------------------------------------------------------------- 5. Nombres, municipios, salida
    print("5/5 Nombres de vialidad y municipios...")
    res = pd.concat([res.reset_index(drop=True),
                     tr.loc[res["indice"], ["fid", "L", "vel", "pct_inund"]].reset_index(drop=True)], axis=1)
    attrs = pd.DataFrame(con.execute(
        f'SELECT fid, NOMBRE, TIPO_VIAL, CODIGO, geom FROM "{s07.CAPA_RED}" WHERE fid IN '
        f'({",".join(str(int(f)) for f in res.fid)})').fetchall(),
        columns=["fid", "NOMBRE", "TIPO_VIAL", "CODIGO", "geom"])
    medio = np.array([red02.puntos_a_lo_largo(red02.gpkg_a_lineas(g), 1e12)[1][0] for g in attrs.geom])
    lab = R.valores(*s05.lcc(medio[:, 0], medio[:, 1]), etiq)
    nombres_mun = ["Fuera del Edomex"] + [texto_utf8(n) for _, n, _ in muns]
    claves_mun = [""] + [k for k, _, _ in muns]
    attrs["municipio"] = [nombres_mun[i] for i in lab]
    attrs["CVE_MPIO"] = [claves_mun[i] for i in lab]
    attrs["lon_medio"], attrs["lat_medio"] = medio[:, 0].round(6), medio[:, 1].round(6)
    res = res.merge(attrs.drop(columns="geom"), on="fid")
    res["desvio_max_min"] = np.where(np.isfinite(res.desvio_max_num),
                                     res.desvio_max_num.round(1).astype(str), "aislamiento")
    res = res.sort_values(["habitantes_afectados", "desvio_max_num"], ascending=False).reset_index(drop=True)
    out = res[["fid", "NOMBRE", "TIPO_VIAL", "CODIGO", "municipio", "CVE_MPIO", "desvio_max_min",
               "habitantes_afectados", "localidades_afectadas", "habitantes_aislados", "localidades_aisladas",
               "desvio_max_finito_min", "pct_inund", "L", "vel", "lon_medio", "lat_medio"]].rename(columns={
        "fid": "fid_rnc", "NOMBRE": "nombre_vialidad", "TIPO_VIAL": "tipo_vial", "CODIGO": "codigo",
        "L": "longitud_m", "vel": "velocidad_kmh"})
    out["pct_inund"] = out["pct_inund"].round(1)
    out["longitud_m"] = out["longitud_m"].round(1)
    out.to_csv(SALIDAS / "11_tramos_criticos.csv", index=False, encoding="utf-8-sig")
    pm = out.groupby("municipio").agg(
        tramos_criticos=("fid_rnc", "size"),
        tramos_que_aislan=("localidades_aisladas", lambda s: int((s > 0).sum())),
        max_habitantes_afectados_un_tramo=("habitantes_afectados", "max")) \
        .sort_values("tramos_criticos", ascending=False)
    pm.to_csv(SALIDAS / "11_tramos_criticos_por_municipio.csv", encoding="utf-8-sig")

    print("\n" + "=" * 112)
    print(f" TRAMOS CRÍTICOS: cierre individual de tramos con > {a.umbral:g} % inundable que provoca desvíos "
          f">= {a.desvio:g} min")
    print(" (localidades rurales -> hospital público más cercano)")
    print("=" * 112)
    print(f"  Localidades rurales: {len(it):,d} ({int(pob.sum()):,d} hab.); hospitales públicos: {len(h):,d}")
    print(f"  Tramos evaluados: {len(evaluar):,d}; TRAMOS CRÍTICOS: {len(out):,d} "
          f"(aíslan alguna localidad: {int((out.localidades_aisladas > 0).sum()):,d})")
    print("\n  Los 25 con más habitantes afectados:")
    print(f"  {'Nombre de la vialidad (tipo)':38s}{'Municipio':28s}{'Desvío máx.':>13s}{'Hab. afectados':>16s}")
    for r in out.head(25).itertuples():
        nom = f"{r.nombre_vialidad} ({r.tipo_vial})"[:37]
        dm = "aislamiento" if r.desvio_max_min == "aislamiento" else f"{float(r.desvio_max_min):.1f} min"
        print(f"  {nom:38s}{r.municipio[:27]:28s}{dm:>13s}{r.habitantes_afectados:>16,d}")
    print(f"\nArchivos guardados en: {SALIDAS}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
