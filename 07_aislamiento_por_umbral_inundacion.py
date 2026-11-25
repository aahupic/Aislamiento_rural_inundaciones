# -*- coding: utf-8 -*-
"""
Script 07 - Aislamiento por Inundaciones: aislamiento de hospitales por umbral
=============================================================================

OBJETIVO
--------
Simular el cierre de los tramos de la red vial que tienen MÁS de un cierto
porcentaje de su longitud sobre zonas inundables (periodo de retorno 100 años)
y medir, para cada umbral, cuántas localidades y habitantes quedan AISLADOS de
los hospitales (tiempo de viaje infinito: no existe ninguna ruta) y cuántos,
sin quedar aislados, sufren un RETRASO de 15 minutos o más.

Umbrales por defecto: 50 %, 25 % y 0 % (parámetro de entrada --umbrales).
  - "más del 50 %": se cierra un tramo si más de la mitad de su longitud es
    inundable (criterio más tolerante, menos cierres).
  - "más del 0 %": se cierra un tramo con cualquier parte inundable (criterio
    más estricto, máximo de cierres; equivale al escenario extremo del
    script 05).

FUENTES
-------
  rnc2025Edomex.gpkg                 Red Nacional de Caminos 2025 (INEGI),
                                     capa 20km_red_vial_buffer y polígono
                                     20km_buffer.
  CENAPRED.zip -> CENAPRED_IPI_2015_Inund_Munici.zip
                                     municipios; el límite del Edomex (unión
                                     de los 125 municipios 15xxx) se usa para
                                     contar tramos cerrados dentro del estado.
  pub-aislamiento\\data\\raw\\tr100_prcent.tif
                                     zonas inundables TR100 (CENAPRED), píxel
                                     de 30 m; valor 0 = inundable, 255 = NoData.
  inegi\\iter_15csv20.zip             ITER 2020: localidades, coordenadas y
                                     población.
  ESTABLECIMIENTO_SALUD_202609.xlsx  CLUES: hospitales en operación.
  03_puntos_muestreo_15m.py          puntos de muestreo cada <= 15 m sobre los
                                     tramos (intervalo, exclusiones y
                                     componente principal se toman de ahí).

MÉTODO, PASO A PASO
-------------------
 1. Red vial (criterios de los scripts 02 y 03)
    - Se excluyen tramos Deshabilitados, En construcción - cerrado y con
      circulación Cerrada en ambos sentidos.
    - Se usa la componente conexa principal del grafo no dirigido de uniones
      (UNION_INI / UNION_FIN).
 2. Porcentaje inundado de cada tramo
    - Sobre cada tramo se colocan n = max(1, ceil(L / 15 m)) puntos en los
      centros de n partes iguales (función del script 03). Cada punto
      representa L/n metros, así que:
          % inundado = 100 * (puntos en píxel TR100 = 0) / n
    - Los puntos (lon/lat EPSG:6365) se proyectan a la Cónica Conforme de
      Lambert de INEGI, la del ráster (función lcc del script 05).
 3. Cierre por umbral
    - Para el umbral U se eliminan los tramos con % inundado > U.
    - Entre dos nodos con tramos paralelos se usa el más rápido que siga
      abierto.
 4. Tiempo de viaje (criterios del script 05)
    - Tiempo de tramo = longitud elipsoidal / VELOCIDAD de la RNC
      ('N/A' -> 30 km/h). Flujo libre, sin congestión.
    - Localidades y hospitales se asignan al nodo más cercano de la red base;
      el tramo de conexión se recorre a 5 km/h (a pie), parámetro
      --vel-conexion.
    - Dijkstra multi-origen desde todos los hospitales del tipo (nodo
      ficticio unido al nodo de cada hospital con su tiempo de conexión).
    - Se calcula para hospital público, privado y "cualquiera" (el más
      cercano de los dos tipos).
 5. Indicadores por umbral y tipo de hospital
    - Localidades aisladas: tiempo = infinito (su nodo quedó en una parte de
      la red sin ningún hospital del tipo).
    - Habitantes aislados: suma de POBTOT de esas localidades.
    - Habitantes aislados fuera de zona inundable: aislados cuya coordenada
      de localidad cae en píxel no inundable. Son personas que no se
      inundan pero quedan incomunicadas: el efecto indirecto de la
      inundación sobre la red.
    - Habitantes con retraso >= 15 min: localidades NO aisladas cuyo tiempo
      en el escenario supera al tiempo base (red completa) en 15 min o más.
      Los aislados no se suman aquí (su retraso es infinito y ya se cuentan
      arriba).

CRITERIOS QUE CONVIENE DEBATIR
------------------------------
  - Hospitales: se mantienen todos disponibles; solo se cierran tramos. Un
    hospital en zona inundable queda inaccesible de hecho si se cierran sus
    tramos de acceso. Con --excluir-hospitales-inundables se retiran además
    los hospitales cuyo punto cae en zona inundable.
  - La localidad se representa por un punto (su coordenada del ITER); toda su
    población se considera aislada o no según ese punto.
  - Se usa toda la zona TR100 a la vez (inundación simultánea); es una
    simulación de peor caso para ese periodo de retorno, no un evento real.
  - Hospitales usados: todos los que están dentro del polígono 20km_buffer
    (incluye CDMX y estados vecinos). Hospital = CLUES 'DE HOSPITALIZACIÓN'
    en operación; privado = 'SERVICIOS MEDICOS PRIVADOS'.

USO
---
  python 07_aislamiento_por_umbral_inundacion.py
  python 07_aislamiento_por_umbral_inundacion.py --umbrales 50 25 10 0
  python 07_aislamiento_por_umbral_inundacion.py --excluir-hospitales-inundables

REQUISITOS
----------
  numpy, pandas, networkx, scipy, Pillow, matplotlib, openpyxl.
  En la misma carpeta: 02_red_vial_edomex.py, 03_puntos_muestreo_15m.py,
  04_localidades_iter_edomex.py, 05_inundacion_caminos_hospitales.py.

SALIDAS (carpeta salidas\\)
--------------------------
  07_resumen_aislamiento_umbrales.csv   un renglón por umbral, tipo de hospital
                                        y conjunto de localidades.
  07_tramos_porcentaje_inundado.csv     fid y % inundado de cada tramo.
  07_localidades_tiempos_umbrales.csv   tiempo base y por umbral de cada
                                        localidad (inf = aislada).
"""

import argparse
import importlib.util
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


red02 = _importar("red02", "02_red_vial_edomex.py")          # GPKG, longitudes, puntos
m03 = _importar("m03", "03_puntos_muestreo_15m.py")          # intervalo 15 m, exclusión
loc04 = _importar("loc04", "04_localidades_iter_edomex.py")  # ITER
s05 = _importar("s05", "05_inundacion_caminos_hospitales.py")  # ráster, LCC, límites

RUTAS = {
    "gpkg": BASE / "rnc2025Edomex.gpkg",
    "cenapred": BASE / "CENAPRED.zip",
    "tr100": BASE / "pub-aislamiento" / "data" / "raw" / "tr100_prcent.tif",
    "iter": BASE / "inegi" / "iter_15csv20.zip",
    "clues": BASE / "ESTABLECIMIENTO_SALUD_202609.xlsx",
}
CAPA_RED = "20km_red_vial_buffer"
CAPA_BUFFER = "20km_buffer"
UMBRALES_DEFECTO = [50.0, 25.0, 0.0]
RETRASO_MIN = 15.0
VEL_NA = 30.0


# ============================================================================
# 1-2. Tramos válidos de la componente principal y su % inundado
# ============================================================================
def tramos_y_porcentaje(con, R, inund, m_edomex, intervalo):
    """Devuelve un DataFrame por tramo de la componente principal con:
    fid, u, v, L (m), vel (km/h), pct_inund, en_edomex (algún punto en el
    estado), lon/lat de sus extremos."""
    filas = con.execute(
        f'SELECT fid, ESTATUS, CONDICION, CIRCULA, VELOCIDAD, UNION_INI, UNION_FIN, geom '
        f'FROM "{CAPA_RED}"').fetchall()
    validos = [f for f in filas if not m03.excluido(f[1], f[2], f[3])]

    # Componente principal (igual que el script 03)
    G = nx.MultiGraph()
    for f in validos:
        G.add_edge(f[5], f[6], key=f[0])
    comp = max(nx.connected_components(G), key=len)
    validos = [f for f in validos if f[5] in comp and f[5] != f[6]]

    reg, lon_l, lat_l, idx_l = [], [], [], []
    for i, (fid, _, _, _, vel, u, v, g) in enumerate(validos):
        partes = red02.gpkg_a_lineas(g)
        L, pts = red02.puntos_a_lo_largo(partes, intervalo)
        lon_l.append(pts[:, 0]); lat_l.append(pts[:, 1]); idx_l.append(np.full(len(pts), i))
        velk = float(vel) if str(vel).replace(".", "").isdigit() else VEL_NA
        reg.append((fid, u, v, L, velk, len(pts), *partes[0][0], *partes[-1][-1]))
    t = pd.DataFrame(reg, columns=["fid", "u", "v", "L", "vel", "n_puntos",
                                   "lon_ini", "lat_ini", "lon_fin", "lat_fin"])
    x, y = s05.lcc(np.concatenate(lon_l), np.concatenate(lat_l))
    idx = np.concatenate(idx_l)
    t["pct_inund"] = 100 * np.bincount(idx, weights=R.valores(x, y, inund), minlength=len(t)) / t["n_puntos"]
    t["en_edomex"] = np.bincount(idx, weights=R.valores(x, y, m_edomex), minlength=len(t)) > 0
    t["t_min"] = t["L"] / 1000 / t["vel"] * 60
    return t


# ============================================================================
# 3. Grafo con los tramos abiertos
# ============================================================================
def grafo(tr, abiertos):
    """Grafo no dirigido; entre paralelos queda el tramo abierto más rápido."""
    s = tr[abiertos]
    s = s.assign(a=np.minimum(s.u, s.v), b=np.maximum(s.u, s.v))
    s = s.loc[s.groupby(["a", "b"])["t_min"].idxmin()]
    G = nx.Graph()
    G.add_weighted_edges_from(zip(s.a, s.b, s.t_min), weight="t")
    return G


# ============================================================================
# 4. Tiempos de viaje
# ============================================================================
def tiempos(G, nodo_hosp, con_hosp, nodo_loc, con_loc):
    """Tiempo (min) de cada localidad al hospital más cercano del conjunto.
    inf = no hay ruta (aislada)."""
    S = "__S__"
    g2 = G.copy()
    for n, c in pd.Series(con_hosp).groupby(nodo_hosp).min().items():
        if n in g2:
            g2.add_edge(S, n, t=c)
    if S not in g2:
        return np.full(len(nodo_loc), np.inf)
    d = nx.single_source_dijkstra_path_length(g2, S, weight="t")
    return np.array([d.get(n, np.inf) for n in nodo_loc]) + con_loc


def indicadores(t_esc, t_base, pob, loc_inund, mascara):
    t_esc, t_base, pob, loc_inund = t_esc[mascara], t_base[mascara], pob[mascara], loc_inund[mascara]
    ais = ~np.isfinite(t_esc)
    ret = (~ais) & ((t_esc - t_base) >= RETRASO_MIN)
    return {"localidades": int(mascara.sum()),
            "localidades_aisladas": int(ais.sum()),
            "habitantes_aislados": int(pob[ais].sum()),
            "habitantes_aislados_fuera_zona_inundable": int(pob[ais & ~loc_inund].sum()),
            "habitantes_retraso_15min_o_mas": int(pob[ret].sum()),
            "localidades_retraso_15min_o_mas": int(ret.sum())}


# ============================================================================
# 5. Programa principal
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUTAS.items():
        ap.add_argument(f"--{k}", type=Path, default=v)
    ap.add_argument("--umbrales", type=float, nargs="+", default=UMBRALES_DEFECTO,
                    help="se cierran los tramos con %% inundado MAYOR que cada umbral (defecto 50 25 0)")
    ap.add_argument("--intervalo", type=float, default=m03.INTERVALO, help="muestreo (m), defecto 15")
    ap.add_argument("--vel-conexion", type=float, default=5.0, help="km/h del tramo de conexión")
    ap.add_argument("--excluir-hospitales-inundables", action="store_true")
    a = ap.parse_args()
    for k in RUTAS:
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()
    SALIDAS.mkdir(exist_ok=True)

    print("1/5 Ráster TR100 y límite estatal...")
    R = s05.Raster(a.tr100)
    inund = R.a == s05.VALOR_INUNDABLE
    anillos, _ = s05.anillos_edomex(a.cenapred)
    m_edomex = R.rasterizar(anillos)
    con = sqlite3.connect(f"file:{a.gpkg.as_posix()}?mode=ro", uri=True)
    buffer = s05.anillos_gpkg_poligono(con.execute(f'SELECT geom FROM "{CAPA_BUFFER}"').fetchone()[0])

    print(f"2/5 Puntos cada {a.intervalo:g} m y % inundado por tramo...")
    tr = tramos_y_porcentaje(con, R, inund, m_edomex, a.intervalo)

    print("3/5 Localidades y hospitales...")
    # Nodos de la red base y su coordenada (LCC)
    coord = pd.concat([
        tr[["u", "lon_ini", "lat_ini"]].set_axis(["n", "lon", "lat"], axis=1),
        tr[["v", "lon_fin", "lat_fin"]].set_axis(["n", "lon", "lat"], axis=1)]).drop_duplicates("n")
    nx_, ny_ = s05.lcc(coord.lon, coord.lat)
    arbol = cKDTree(np.column_stack([nx_, ny_]))
    nodos = coord["n"].to_numpy()
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
           & (cl["ESTATUS DE OPERACION"] == "EN OPERACION")].copy()
    h["lat"] = pd.to_numeric(h["LATITUD"], errors="coerce")
    h["lon"] = pd.to_numeric(h["LONGITUD"], errors="coerce")
    h = h[h["lat"].notna() & h["lon"].notna()].copy()
    h["tipo"] = np.where(h["NOMBRE DE LA INSTITUCION"] == "SERVICIOS MEDICOS PRIVADOS", "Privado", "Público")
    hx, hy = s05.lcc(h["lon"], h["lat"])
    dentro = np.zeros(len(h), bool)
    for anillo in buffer:
        dentro |= MplPath(anillo).contains_points(np.column_stack([hx, hy]))
    h = h[dentro].copy()
    hx, hy = s05.lcc(h["lon"], h["lat"])
    h["inundable"] = R.valores(hx, hy, inund)
    h["nodo"], h["con_min"] = asignar(hx, hy)

    print("4/5 Tiempos base y por umbral...")
    escenarios = [("Base", None)] + [(f"> {u:g} %", u) for u in a.umbrales]
    T = {}
    cierres = []
    for nombre, u in escenarios:
        abiertos = np.ones(len(tr), bool) if u is None else (tr["pct_inund"] <= u).to_numpy()
        G = grafo(tr, abiertos)
        hs_all = h if (u is None or not a.excluir_hospitales_inundables) else h[~h["inundable"]]
        for tipo in ("Público", "Privado"):
            hs = hs_all[hs_all["tipo"] == tipo]
            T[(nombre, tipo)] = tiempos(G, hs["nodo"].to_numpy(), hs["con_min"].to_numpy(),
                                        it["nodo"].to_numpy(), it["con_min"].to_numpy())
        T[(nombre, "Cualquiera")] = np.minimum(T[(nombre, "Público")], T[(nombre, "Privado")])
        cerr = ~abiertos
        cierres.append({"escenario": nombre, "tramos_cerrados": int(cerr.sum()),
                        "km_cerrados": tr.loc[cerr, "L"].sum() / 1000,
                        "tramos_cerrados_edomex": int((cerr & tr["en_edomex"]).sum()),
                        "pct_tramos_cerrados": 100 * cerr.mean()})

    print("5/5 Indicadores...")
    pob = it["pob"].to_numpy()
    loc_inund = it["inundable"].to_numpy()
    resumen = []
    for nombre, u in escenarios[1:]:
        for tipo in ("Público", "Privado", "Cualquiera"):
            for conj, m in (("Todas", np.ones(len(it), bool)), ("Rurales (< 2,500 hab.)", it["rural"].to_numpy())):
                resumen.append({"umbral_pct": u, "escenario": nombre, "hospital": tipo, "conjunto": conj,
                                **indicadores(T[(nombre, tipo)], T[("Base", tipo)], pob, loc_inund, m)})
    res = pd.DataFrame(resumen)

    # ------------------------------------------------------------------ salida
    print("\n" + "=" * 96)
    print(" AISLAMIENTO DE HOSPITALES POR CIERRE DE TRAMOS INUNDABLES (TR100) - ESTADO DE MÉXICO")
    print("=" * 96)
    print(f"  Tramos de la componente principal: {len(tr):,d}; con alguna parte inundable: "
          f"{int((tr.pct_inund > 0).sum()):,d}")
    print(f"  Localidades: {len(it):,d} ({int(pob.sum()):,d} hab.); en zona inundable: "
          f"{int(loc_inund.sum()):,d} ({int(pob[loc_inund].sum()):,d} hab.)")
    print(f"  Hospitales (20km_buffer): {(h.tipo=='Público').sum()} públicos, {(h.tipo=='Privado').sum()} privados"
          + ("; se retiran los inundables en los escenarios" if a.excluir_hospitales_inundables else
             "; todos disponibles (solo se cierran tramos)"))
    print("\n  Tramos cerrados por umbral")
    for c in cierres[1:]:
        print(f"    {c['escenario']:8s}: {c['tramos_cerrados']:>9,d} tramos ({c['pct_tramos_cerrados']:.2f} %), "
              f"{c['km_cerrados']:>10,.1f} km; {c['tramos_cerrados_edomex']:,d} tocan el Edomex")
    aisl_base = {t: int((~np.isfinite(T[('Base', t)])).sum()) for t in ("Público", "Privado")}
    print(f"  (localidades sin ruta en la red completa: público {aisl_base['Público']}, privado {aisl_base['Privado']})")

    for conj in ("Todas", "Rurales (< 2,500 hab.)"):
        print(f"\n  Localidades: {conj}")
        print(f"    {'Umbral':8s}{'Hospital':12s}{'Loc. aisladas':>14s}{'Hab. aislados':>15s}"
              f"{'Aisl. fuera zona inund.':>25s}{'Hab. retraso >=15 min':>23s}")
        for _, r in res[res.conjunto == conj].iterrows():
            print(f"    {r.escenario:8s}{r.hospital:12s}{r.localidades_aisladas:>14,d}{r.habitantes_aislados:>15,d}"
                  f"{r.habitantes_aislados_fuera_zona_inundable:>25,d}{r.habitantes_retraso_15min_o_mas:>23,d}")

    res.to_csv(SALIDAS / "07_resumen_aislamiento_umbrales.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cierres).round(3).to_csv(SALIDAS / "07_tramos_cerrados_umbrales.csv", index=False,
                                          encoding="utf-8-sig")
    tr[["fid", "u", "v", "L", "vel", "n_puntos", "pct_inund", "en_edomex"]].round(3) \
        .to_csv(SALIDAS / "07_tramos_porcentaje_inundado.csv", index=False, encoding="utf-8-sig")
    out = it[["CVEGEO", "NOM_MUN", "NOM_LOC", "lon", "lat", "POBTOT", "rural", "inundable"]].copy()
    for (nombre, tipo), t in T.items():
        out[f"min_{tipo.lower().replace('ú', 'u')}_{nombre.replace('> ', 'gt').replace(' %', '')}"] = np.round(t, 2)
    out.to_csv(SALIDAS / "07_localidades_tiempos_umbrales.csv", index=False, encoding="utf-8-sig")
    print(f"\nArchivos guardados en: {SALIDAS}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
