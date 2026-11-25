# -*- coding: utf-8 -*-
"""
Script 10 - Aislamiento por Inundaciones: aislamiento según el conjunto de hospitales
====================================================================================

OBJETIVO
--------
Con la red vial a la que se le cierran los tramos con MÁS DEL 50 % de su
longitud en zona inundable (periodo de retorno 100 años), medir el
aislamiento y el retraso de las localidades respecto a distintos CONJUNTOS de
hospitales públicos en operación (escenarios):

  1. Todos         todos los hospitales públicos.
  2. Inundables    solo los hospitales cuyo punto cae en píxel inundable.
  3. Secos         solo los hospitales fuera de zona inundable (equivale a
                   suponer que los inundables quedan fuera de servicio).
  4. Segundo nivel solo los hospitales públicos de SEGUNDO NIVEL de atención.
                   Nota: en el CLUES ningún hospital público es de primer
                   nivel (en el Edomex: 106 de segundo y 14 de tercer nivel);
                   el primer nivel solo existe en unidades de consulta
                   externa. Por decisión del usuario (2026-11-22) se usa el
                   segundo nivel, el más básico con hospitalización.

Para cada escenario se reporta:
  - número total de localidades aisladas,
  - habitantes aislados,
  - habitantes aislados que están FUERA de zona inundable,
  - habitantes con retraso >= 15 minutos.
Para todas las localidades y para las rurales (< 2,500 hab.).

PARÁMETROS DE ENTRADA
---------------------
  --intervalo  espaciado de los puntos de muestreo sobre los tramos (m);
               por defecto el del script 03 (15 m).
  --umbral     % de la longitud del tramo en zona inundable a partir del cual
               el tramo se cierra (se cierra si % > umbral); defecto 50.
  Hospitales   públicos (institución distinta de 'SERVICIOS MEDICOS
               PRIVADOS'), 'DE HOSPITALIZACIÓN', 'EN OPERACION'.

FUENTES
-------
  rnc2025Edomex.gpkg                 red vial (capa 20km_red_vial_buffer) y
                                     polígono 20km_buffer.
  CENAPRED.zip -> CENAPRED_IPI_2015_Inund_Munici.zip
                                     límite del Edomex (125 municipios).
  pub-aislamiento\\data\\raw\\tr100_prcent.tif
                                     zonas inundables TR100 (0 = inundable).
  inegi\\iter_15csv20.zip             localidades, coordenadas y población.
  ESTABLECIMIENTO_SALUD_202609.xlsx  CLUES: hospitales, institución y nivel.
  03_puntos_muestreo_15m.py          intervalo de muestreo y criterios de
                                     exclusión (a través del script 07).

MÉTODO, PASO A PASO (mismos criterios del script 07)
----------------------------------------------------
 1. Red: tramos válidos (sin Deshabilitados, En construcción - cerrado ni
    Cerrada en ambos sentidos) de la componente principal.
 2. % inundado de cada tramo = puntos de muestreo (cada <= 15 m) en píxel
    inundable / puntos del tramo.
 3. Dos grafos no dirigidos con tiempo de tramo = longitud / VELOCIDAD RNC:
      - base:      red completa,
      - escenario: sin los tramos con % inundado > umbral.
 4. Localidades y hospitales se asignan al nodo más cercano de la red base;
    conexión a 5 km/h (a pie).
 5. Para cada conjunto de hospitales: Dijkstra multi-origen en ambos grafos
    -> t_base y t_esc de cada localidad.
 6. Indicadores:
      - aislada:            t_esc = infinito.
      - aislado fuera de zona inundable: aislado y el punto de la localidad
                            cae en píxel no inundable (efecto indirecto).
      - retraso >= 15 min:  no aislada y t_esc - t_base >= 15.
    REFERENCIA DEL RETRASO (decisión del usuario, 2026-11-22): el retraso se
    mide contra la situación normal = red completa con TODOS los hospitales
    públicos (t_ref = t_base del escenario "Todos"):
          retraso = t_esc(conjunto) - t_base(Todos)
    Así el retraso suma dos efectos: el cierre de tramos y la pérdida de
    hospitales del conjunto (p. ej., en "Solo secos", ir a un hospital seco
    más lejano porque el inundable de siempre ya no está disponible). En el
    escenario "Todos" ambos criterios coinciden.
    Como referencia también se reporta cuántas localidades no tienen ruta
    al conjunto ni siquiera en la red completa (debería ser 0 o casi 0).

CRITERIOS QUE CONVIENE DEBATIR
------------------------------
  - Se usan los hospitales públicos dentro del polígono 20km_buffer
    (incluye CDMX y estados vecinos), como en los scripts 05 a 09.
  - "Inundable" se evalúa con el punto del hospital en el ráster de 30 m.
  - Inundación simultánea de toda la zona TR100: peor caso.

USO
---
  python 10_aislamiento_por_conjunto_hospitales.py
  python 10_aislamiento_por_conjunto_hospitales.py --umbral 50 --intervalo 15

REQUISITOS
----------
  numpy, pandas, networkx, scipy, matplotlib, Pillow, openpyxl.
  En la misma carpeta: scripts 02, 03, 04, 05 y 07.

SALIDAS (carpeta salidas\\)
--------------------------
  10_resumen_aislamiento_conjuntos.csv   un renglón por escenario y conjunto
                                         de localidades.
  10_hospitales_escenarios.csv           hospitales usados, con nivel y
                                         bandera inundable.
  10_localidades_tiempos_conjuntos.csv   tiempos base y escenario por
                                         localidad y conjunto (inf = aislada).
"""

import argparse
import importlib.util
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

AQUI = Path(__file__).resolve().parent
SALIDAS = AQUI / "salidas"


def _importar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


loc04 = _importar("loc04", "04_localidades_iter_edomex.py")
s05 = _importar("s05", "05_inundacion_caminos_hospitales.py")
s07 = _importar("s07", "07_aislamiento_por_umbral_inundacion.py")   # usa 02 y 03

RUTAS = dict(s07.RUTAS)
UMBRAL_DEFECTO = 50.0

# Escenarios: nombre -> función que filtra el DataFrame de hospitales públicos
ESCENARIOS = {
    "Todos": lambda h: h,
    "Solo inundables": lambda h: h[h["inundable"]],
    "Solo secos": lambda h: h[~h["inundable"]],
    "Solo segundo nivel": lambda h: h[h["NIVEL ATENCION"] == "SEGUNDO NIVEL"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUTAS.items():
        ap.add_argument(f"--{k}", type=Path, default=v)
    ap.add_argument("--intervalo", type=float, default=s07.m03.INTERVALO, help="muestreo sobre tramos (m)")
    ap.add_argument("--umbral", type=float, default=UMBRAL_DEFECTO,
                    help="se cierran los tramos con %% inundado MAYOR que este valor (defecto 50)")
    ap.add_argument("--vel-conexion", type=float, default=5.0, help="km/h del tramo de conexión")
    a = ap.parse_args()
    for k in RUTAS:
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()
    SALIDAS.mkdir(exist_ok=True)

    # ------------------------------------------------ 1-2. Red y % inundado
    print("1/4 Ráster, límite estatal y % inundado por tramo...")
    R = s05.Raster(a.tr100)
    inund = R.a == s05.VALOR_INUNDABLE
    anillos, _ = s05.anillos_edomex(a.cenapred)
    m_edomex = R.rasterizar(anillos)
    con = sqlite3.connect(f"file:{a.gpkg.as_posix()}?mode=ro", uri=True)
    buffer = s05.anillos_gpkg_poligono(con.execute(f'SELECT geom FROM "{s07.CAPA_BUFFER}"').fetchone()[0])
    tr = s07.tramos_y_porcentaje(con, R, inund, m_edomex, a.intervalo)
    cerrados = (tr["pct_inund"] > a.umbral).to_numpy()

    # ------------------------------------------------ 4. Asignación a la red
    print("2/4 Localidades y hospitales públicos...")
    coord = pd.concat([
        tr[["u", "lon_ini", "lat_ini"]].set_axis(["n", "lon", "lat"], axis=1),
        tr[["v", "lon_fin", "lat_fin"]].set_axis(["n", "lon", "lat"], axis=1)]).drop_duplicates("n")
    cx, cy = s05.lcc(coord.lon, coord.lat)
    arbol, nodos = cKDTree(np.column_stack([cx, cy])), coord["n"].to_numpy()
    vcon = a.vel_conexion / 3.6 * 60                                    # m/min

    def asignar(x, y):
        d, j = arbol.query(np.column_stack([x, y]))
        return nodos[j], d / vcon

    it = loc04.leer_iter(a.iter)
    it = it[~it["LOC"].isin(loc04.LOC_AGREGADOS)].copy()
    it["pob"] = pd.to_numeric(it["POBTOT"], errors="coerce").fillna(0)
    it["lon"] = it["LONGITUD"].map(loc04.dms_a_decimal)
    it["lat"] = it["LATITUD"].map(loc04.dms_a_decimal)
    it = it[it["lon"].notna() & it["lat"].notna()].reset_index(drop=True)
    it["CVEGEO"] = it["ENTIDAD"] + it["MUN"] + it["LOC"]
    it["rural"] = it["pob"] < loc04.UMBRAL_RURAL
    lx, ly = s05.lcc(it["lon"], it["lat"])
    it["inundable"] = R.valores(lx, ly, inund)
    it["nodo"], it["con_min"] = asignar(lx, ly)

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
    hx, hy = s05.lcc(h["lon"], h["lat"])
    h["inundable"] = R.valores(hx, hy, inund)
    h["edomex"] = h["CLAVE DE LA ENTIDAD"] == "15"
    h["nodo"], h["con_min"] = asignar(hx, hy)

    # ------------------------------------------------ 3, 5. Grafos y tiempos
    print("3/4 Grafos (base y tramos > umbral cerrados) y tiempos por escenario...")
    G_base = s07.grafo(tr, np.ones(len(tr), bool))
    G_esc = s07.grafo(tr, ~cerrados)
    nl, cl_ = it["nodo"].to_numpy(), it["con_min"].to_numpy()
    T, info = {}, []
    for nombre, filtro in ESCENARIOS.items():
        hs = filtro(h)
        T[(nombre, "base")] = s07.tiempos(G_base, hs["nodo"].to_numpy(), hs["con_min"].to_numpy(), nl, cl_)
        T[(nombre, "esc")] = s07.tiempos(G_esc, hs["nodo"].to_numpy(), hs["con_min"].to_numpy(), nl, cl_)
        info.append({"escenario": nombre, "hospitales_usados": len(hs),
                     "hospitales_edomex": int(hs["edomex"].sum()),
                     "hospitales_edomex_inundables": int((hs["edomex"] & hs["inundable"]).sum()),
                     "localidades_sin_ruta_red_completa": int((~np.isfinite(T[(nombre, "base")])).sum())})

    # ------------------------------------------------ 6. Indicadores
    print("4/4 Indicadores...")
    pob, loc_inund = it["pob"].to_numpy(), it["inundable"].to_numpy()
    filas = []
    for nombre in ESCENARIOS:
        for conj, m in (("Todas", np.ones(len(it), bool)), ("Rurales (< 2,500 hab.)", it["rural"].to_numpy())):
            # Referencia del retraso: red completa con TODOS los hospitales públicos
            filas.append({"escenario": nombre, "conjunto": conj, "umbral_pct": a.umbral,
                          **s07.indicadores(T[(nombre, "esc")], T[("Todos", "base")], pob, loc_inund, m)})
    res = pd.DataFrame(filas)

    # ------------------------------------------------ Impresión
    print("\n" + "=" * 100)
    print(f" AISLAMIENTO POR CONJUNTO DE HOSPITALES PÚBLICOS - tramos con > {a.umbral:g} % inundable cerrados (TR100)")
    print("=" * 100)
    print(f"  Tramos cerrados: {int(cerrados.sum()):,d} de {len(tr):,d} ({100 * cerrados.mean():.2f} %); "
          f"muestreo cada {a.intervalo:g} m")
    print(f"  Localidades: {len(it):,d} ({int(pob.sum()):,d} hab.); rurales {int(it.rural.sum()):,d} "
          f"({int(pob[it.rural].sum()):,d} hab.)")
    print(f"\n  {'Escenario':20s}{'Hosp. usados':>13s}{'en Edomex':>11s}{'Edomex inund.':>15s}"
          f"{'Loc. sin ruta (red completa)':>30s}")
    for r in info:
        print(f"  {r['escenario']:20s}{r['hospitales_usados']:>13,d}{r['hospitales_edomex']:>11,d}"
              f"{r['hospitales_edomex_inundables']:>15,d}{r['localidades_sin_ruta_red_completa']:>30,d}")
    for conj in ("Todas", "Rurales (< 2,500 hab.)"):
        print(f"\n  Localidades: {conj}")
        print(f"    {'Escenario':20s}{'Loc. aisladas':>14s}{'Hab. aislados':>15s}"
              f"{'Aisl. fuera zona inund.':>25s}{'Hab. retraso >=15 min':>23s}")
        for r in res[res.conjunto == conj].itertuples():
            print(f"    {r.escenario:20s}{r.localidades_aisladas:>14,d}{r.habitantes_aislados:>15,d}"
                  f"{r.habitantes_aislados_fuera_zona_inundable:>25,d}{r.habitantes_retraso_15min_o_mas:>23,d}")
    print("  (retraso = tiempo en el escenario vs. red completa con TODOS los hospitales públicos)")

    # ------------------------------------------------ Salidas
    res.merge(pd.DataFrame(info), on="escenario").to_csv(
        SALIDAS / "10_resumen_aislamiento_conjuntos.csv", index=False, encoding="utf-8-sig")
    h[["CLUES", "CLAVE DE LA ENTIDAD", "NOMBRE DE LA INSTITUCION", "NOMBRE DE LA UNIDAD", "NIVEL ATENCION",
       "lon", "lat", "inundable", "edomex"]].to_csv(SALIDAS / "10_hospitales_escenarios.csv", index=False,
                                                    encoding="utf-8-sig")
    out = it[["CVEGEO", "NOM_MUN", "NOM_LOC", "lon", "lat", "POBTOT", "rural", "inundable"]].copy()
    claves = {"Todos": "todos", "Solo inundables": "inund", "Solo secos": "secos", "Solo segundo nivel": "nivel2"}
    for (nombre, red), t in T.items():
        out[f"min_{claves[nombre]}_{red}"] = np.round(t, 2)
    out.to_csv(SALIDAS / "10_localidades_tiempos_conjuntos.csv", index=False, encoding="utf-8-sig")
    print(f"\nArchivos guardados en: {SALIDAS}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
