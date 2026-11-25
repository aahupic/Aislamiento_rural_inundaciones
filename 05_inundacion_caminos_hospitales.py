# -*- coding: utf-8 -*-
"""
Script 05 - Aislamiento por Inundaciones: caminos, hospitales y accesibilidad
============================================================================
Fuentes:
  rnc2025Edomex.gpkg                     red vial (capa 20km_red_vial_buffer) y
                                         polígono 20km_buffer
  CENAPRED.zip -> CENAPRED_IPI_2015_Inund_Munici.zip
                                         municipios (límite del Edomex = unión
                                         de los 125 municipios, CVE 15xxx)
  pub-aislamiento\\data\\raw\\tr100_prcent.tif
                                         zonas inundables, periodo de retorno
                                         100 años (CENAPRED), píxel 30 m
  inegi\\iter_15csv20.zip                 localidades y población (Censo 2020)
  ESTABLECIMIENTO_SALUD_202609.xlsx      CLUES (Secretaría de Salud), sep-2026

Indicadores:
  1. Km de caminos en el Estado de México.
  2. Km de caminos en zonas inundables (TR100) y su porcentaje.
  3. Hospitales públicos, privados y total (Edomex).
  4. Hospitales públicos, privados y total en zonas inundables (TR100) y %.
  5. Población de localidades rurales (< 2,500 hab.) por tiempo de viaje al
     hospital público y al privado más cercano: 0-15, 15-30 y > 30 min.
     Habitantes por grupo, porcentaje y percentiles del tiempo de viaje.

Criterios (explícitos para que puedan debatirse):
  RÁSTER TR100
  - tr100_prcent.tif solo tiene dos valores: 0 y 255 (255 = NoData). Se toma
    "zona inundable" = píxel con valor 0. Control: dentro del Edomex suma
    ~3,278 km2, del mismo orden que el área inundable municipal del IPI 2015
    (2,114 km2), y los municipios con todas sus localidades en zona
    inundable son los del antiguo lecho lacustre (Nezahualcóyotl, Valle de
    Chalco, Atenco, Chiconcuac...). Conviene confirmarlo con la descripción
    original del ráster.
  - Proyección: Cónica Conforme de Lambert ITRF92 de INEGI (lat0 12, lon0
    -102, paralelos 17.5 y 29.5, FE 2,500,000, GRS80). Los puntos en
    grados (EPSG:6365) se proyectan con las fórmulas de Snyder (sin GDAL);
    la diferencia ITRF92/ITRF2008 es de centímetros, despreciable a 30 m.
  CAMINOS
  - Tramos válidos = mismos criterios del script 02 (se excluyen
    Deshabilitado, En construcción - cerrado, Cerrada en ambos sentidos).
  - Longitud dentro del Edomex e inundable: puntos cada <= 15 m sobre cada
    tramo (script 03); cada punto representa L/n metros del tramo. Un punto
    cuenta en el Edomex si cae en un píxel del límite estatal rasterizado y
    como inundable si además el píxel TR100 vale 0.
  HOSPITALES
  - Hospital = CLUES con 'NOMBRE TIPO ESTABLECIMIENTO' = 'DE HOSPITALIZACIÓN'
    y 'ESTATUS DE OPERACION' = 'EN OPERACION'.
  - Privado = institución 'SERVICIOS MEDICOS PRIVADOS'; público = el resto
    (SSA, IMSS-Bienestar, IMSS, ISSSTE, SEDENA, estatales, etc.).
  - Conteos (3 y 4): CLAVE DE LA ENTIDAD = 15.
  - Accesibilidad (5): se usan TODOS los hospitales en operación dentro del
    polígono 20km_buffer (incluye CDMX y estados vecinos), porque la población
    rural cercana a un límite estatal puede atenderse al otro lado.
  ACCESIBILIDAD
  - Grafo no dirigido de tramos válidos, componente principal (script 02).
    Tiempo de tramo = longitud / VELOCIDAD de la RNC (velocidad de
    circulación de referencia; 'N/A' -> 30 km/h, la moda). Es un tiempo a
    flujo libre: optimista en zonas urbanas congestionadas.
  - Localidades y hospitales se asignan al nodo más cercano (en metros,
    LCC). El tramo de conexión se recorre a VEL_CONEXION (defecto 5 km/h, a
    pie) para no suponer un camino que la RNC no registra.
  - Grupos: [0, 15), [15, 30) y >= 30 min; 'sin acceso' si no hay ruta.
  - Percentiles: del tiempo de viaje, ponderados por habitantes.
  - Escenario opcional (no solicitado; se activa con --escenario-inundacion):
    se eliminan los tramos con algún punto en zona inundable TR100 y los
    hospitales inundables, y se recalculan los tiempos; las localidades sin
    ruta quedan como 'sin acceso'. Es una cota extrema: supone que toda la
    zona TR100 se inunda a la vez y que un solo punto inundado cierra el tramo.

Uso:
  python 05_inundacion_caminos_hospitales.py
  python 05_inundacion_caminos_hospitales.py --vel-conexion 5 --intervalo 15

Requisitos: numpy, pandas, networkx, scipy, Pillow, openpyxl.
Requiere en la misma carpeta: 02_red_vial_edomex.py y
04_localidades_iter_edomex.py (se reutilizan sus funciones).
"""

import argparse
import importlib.util
import io
import math
import sqlite3
import struct
import time
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None
AQUI = Path(__file__).resolve().parent
BASE = AQUI.parent                                         # D:\ARTUAEM
SALIDAS = AQUI / "salidas"


def _importar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


red02 = _importar("red02", "02_red_vial_edomex.py")
loc04 = _importar("loc04", "04_localidades_iter_edomex.py")

RUTAS = {
    "gpkg": BASE / "rnc2025Edomex.gpkg",
    "cenapred": BASE / "CENAPRED.zip",
    "tr100": BASE / "pub-aislamiento" / "data" / "raw" / "tr100_prcent.tif",
    "iter": BASE / "inegi" / "iter_15csv20.zip",
    "clues": BASE / "ESTABLECIMIENTO_SALUD_202609.xlsx",
}
CAPA_RED = "20km_red_vial_buffer"
CAPA_BUFFER = "20km_buffer"
IPI_ZIP = "CENAPRED_IPI_2015_Inund_Munici.zip"
VALOR_INUNDABLE = 0
VEL_NA = 30.0
CORTES = [15.0, 30.0]
PERCENTILES = [10, 25, 50, 75, 90]

# ----------------------------------------------------------------------------
# Proyección Lambert Conformal Conic 2SP (Snyder 1987), elipsoide GRS80
# ----------------------------------------------------------------------------
A, F = 6378137.0, 1 / 298.257222101
E = math.sqrt(F * (2 - F))
LAT0, LON0, LAT1, LAT2, FE, FN = 12.0, -102.0, 17.5, 29.5, 2500000.0, 0.0


def _m(phi):
    return np.cos(phi) / np.sqrt(1 - (E * np.sin(phi)) ** 2)


def _t(phi):
    s = E * np.sin(phi)
    return np.tan(np.pi / 4 - phi / 2) / ((1 - s) / (1 + s)) ** (E / 2)


_p0, _p1, _p2 = np.radians([LAT0, LAT1, LAT2])
_N = (np.log(_m(_p1)) - np.log(_m(_p2))) / (np.log(_t(_p1)) - np.log(_t(_p2)))
_F = _m(_p1) / (_N * _t(_p1) ** _N)
_RHO0 = A * _F * _t(_p0) ** _N


def lcc(lon, lat):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    rho = A * _F * _t(np.radians(lat)) ** _N
    th = _N * np.radians(lon - LON0)
    return FE + rho * np.sin(th), FN + _RHO0 - rho * np.cos(th)


# ----------------------------------------------------------------------------
# Ráster y máscaras
# ----------------------------------------------------------------------------
class Raster:
    def __init__(self, ruta):
        im = Image.open(ruta)
        self.a = np.asarray(im)
        tags = im.tag_v2
        sx, sy = tags[33550][:2]                           # ModelPixelScale
        tp = tags[33922]                                   # ModelTiepoint
        self.x0, self.y0, self.sx, self.sy = tp[3], tp[4], sx, sy
        self.h, self.w = self.a.shape

    def filcol(self, x, y):
        c = np.floor((np.asarray(x) - self.x0) / self.sx).astype(np.int64)
        f = np.floor((self.y0 - np.asarray(y)) / self.sy).astype(np.int64)
        dentro = (c >= 0) & (c < self.w) & (f >= 0) & (f < self.h)
        return f, c, dentro

    def valores(self, x, y, mascara=None, fuera=False):
        f, c, ok = self.filcol(x, y)
        arr = self.a if mascara is None else mascara
        out = np.full(len(f), fuera, dtype=arr.dtype)
        out[ok] = arr[f[ok], c[ok]]
        return out

    def rasterizar(self, anillos):
        img = Image.new("L", (self.w, self.h), 0)
        dr = ImageDraw.Draw(img)
        for r in anillos:
            dr.polygon([((x - self.x0) / self.sx, (self.y0 - y) / self.sy) for x, y in r], fill=1)
        return np.asarray(img).astype(bool)


def anillos_edomex(ruta_cenapred):
    """Anillos (LCC) de los municipios 15xxx del shapefile IPI 2015."""
    with zipfile.ZipFile(ruta_cenapred) as z:
        nombre = next(n for n in z.namelist() if n.endswith(IPI_ZIP))
        interior = zipfile.ZipFile(io.BytesIO(z.read(nombre)))
    shp = next(n for n in interior.namelist() if n.lower().endswith(".shp"))
    dbf = shp[:-4] + ".dbf"
    claves = _dbf_campo(interior.read(dbf), "CVE_MPIO")
    geoms = _shp_poligonos(interior.read(shp))
    anillos = [r for k, g in zip(claves, geoms) if k.startswith("15") and len(k) == 5 for r in g]
    n_mun = sum(1 for k in claves if k.startswith("15") and len(k) == 5)
    return anillos, n_mun


def _dbf_campo(f, campo):
    n = struct.unpack("<I", f[4:8])[0]
    hl, rl = struct.unpack("<HH", f[8:12])
    flds, o = [], 32
    while f[o] != 0x0D:
        flds.append((f[o:o + 11].split(b"\0")[0].decode("latin-1"), f[o + 16]))
        o += 32
    ini = sum(L for _, L in flds[:[n for n, _ in flds].index(campo)])
    L = dict(flds)[campo]
    return [f[hl + i * rl + 1 + ini: hl + i * rl + 1 + ini + L].decode("latin-1").strip() for i in range(n)]


def _shp_poligonos(f):
    o, out = 100, []
    while o < len(f):
        clen = struct.unpack(">i", f[o + 4:o + 8])[0] * 2
        c = o + 8
        if struct.unpack("<i", f[c:c + 4])[0] == 0:
            out.append([])
        else:
            npart, npts = struct.unpack("<2i", f[c + 36:c + 44])
            partes = list(struct.unpack(f"<{npart}i", f[c + 44:c + 44 + 4 * npart])) + [npts]
            pts = np.frombuffer(f, "<f8", 2 * npts, c + 44 + 4 * npart).reshape(npts, 2)
            out.append([pts[partes[i]:partes[i + 1]] for i in range(npart)])
        o = c + clen
    return out


def anillos_gpkg_poligono(blob):
    """Anillos exteriores de un (Multi)Polygon GeoPackage (WKB XY)."""
    b = memoryview(blob)
    off = 8 + red02._ENV_BYTES[(blob[3] >> 1) & 7]
    anillos = []

    def poligono(off):
        orden = "<" if b[off] == 1 else ">"
        off += 5
        nr = struct.unpack_from(orden + "I", b, off)[0]
        off += 4
        for k in range(nr):
            n = struct.unpack_from(orden + "I", b, off)[0]
            off += 4
            pts = np.frombuffer(b, orden + "f8", 2 * n, off).reshape(n, 2)
            if k == 0:
                anillos.append(pts)
            off += 16 * n
        return off

    orden = "<" if b[off] == 1 else ">"
    tipo = struct.unpack_from(orden + "I", b, off + 1)[0] % 1000
    if tipo == 3:
        poligono(off)
    else:
        n = struct.unpack_from(orden + "I", b, off + 5)[0]
        off += 9
        for _ in range(n):
            off = poligono(off)
    return anillos


# ----------------------------------------------------------------------------
# Utilidades de resultados
# ----------------------------------------------------------------------------
def percentiles_ponderados(valores, pesos, ps):
    v, w = np.asarray(valores, float), np.asarray(pesos, float)
    ok = np.isfinite(v)
    v, w = v[ok], w[ok]
    orden = np.argsort(v)
    v, w = v[orden], w[orden]
    acum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return {f"P{p}": float(np.interp(p / 100, acum, v)) for p in ps}


def agrupar(tiempos, pob):
    etiquetas = [f"0-{CORTES[0]:g} min", f"{CORTES[0]:g}-{CORTES[1]:g} min",
                 f"> {CORTES[1]:g} min", "sin acceso"]
    t = np.asarray(tiempos)
    grupo = np.select([t < CORTES[0], t < CORTES[1], np.isfinite(t)], etiquetas[:3], etiquetas[3])
    df = pd.DataFrame({"grupo": grupo, "pob": pob})
    tab = df.groupby("grupo").agg(localidades=("pob", "size"), habitantes=("pob", "sum")) \
        .reindex(etiquetas, fill_value=0)
    tab["porcentaje"] = 100 * tab["habitantes"] / tab["habitantes"].sum()
    return tab


# ----------------------------------------------------------------------------
# Proceso
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUTAS.items():
        ap.add_argument(f"--{k}", type=Path, default=v)
    ap.add_argument("--intervalo", type=float, default=15.0, help="muestreo sobre tramos (m)")
    ap.add_argument("--vel-conexion", type=float, default=5.0, help="km/h del tramo de conexión")
    ap.add_argument("--escenario-inundacion", action="store_true",
                    help="agrega el escenario extremo con tramos inundables cerrados")
    a = ap.parse_args()
    for k in RUTAS:
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()
    SALIDAS.mkdir(exist_ok=True)

    # --- Ráster y máscaras --------------------------------------------------
    print("Leyendo ráster TR100 y límites...")
    R = Raster(a.tr100)
    inund = R.a == VALOR_INUNDABLE
    anillos, n_mun = anillos_edomex(a.cenapred)
    m_edomex = R.rasterizar(anillos)
    con = sqlite3.connect(f"file:{a.gpkg.as_posix()}?mode=ro", uri=True)
    m_buffer = R.rasterizar(anillos_gpkg_poligono(
        con.execute(f'SELECT geom FROM "{CAPA_BUFFER}"').fetchone()[0]))
    area_edomex = m_edomex.sum() * R.sx * R.sy / 1e6
    area_inund = (m_edomex & inund).sum() * R.sx * R.sy / 1e6

    # --- Caminos: puntos de muestreo -----------------------------------------
    print("Muestreando tramos...")
    filas = con.execute(
        f'SELECT fid, ESTATUS, CONDICION, CIRCULA, VELOCIDAD, UNION_INI, UNION_FIN, geom '
        f'FROM "{CAPA_RED}"').fetchall()
    E = red02.EXCLUSION
    validos = [f for f in filas if not (f[1] in E["ESTATUS"] or f[2] in E["CONDICION"]
                                        or f[3] in E["CIRCULA"])]
    lon_l, lat_l, peso_l, tramo_l = [], [], [], []
    tramos = []                      # (fid, u, v, L, vel, xy_ini, xy_fin)
    for i, (fid, _, _, _, vel, u, v, g) in enumerate(validos):
        partes = red02.gpkg_a_lineas(g)
        L, pts = red02.puntos_a_lo_largo(partes, a.intervalo)
        lon_l.append(pts[:, 0]); lat_l.append(pts[:, 1])
        peso_l.append(np.full(len(pts), L / len(pts)))
        tramo_l.append(np.full(len(pts), i))
        velk = float(vel) if str(vel).replace(".", "").isdigit() else VEL_NA
        tramos.append((fid, u, v, L, velk, partes[0][0], partes[-1][-1]))
    lon = np.concatenate(lon_l); lat = np.concatenate(lat_l)
    peso = np.concatenate(peso_l); tramo = np.concatenate(tramo_l)
    x, y = lcc(lon, lat)
    en_edomex = R.valores(x, y, m_edomex)
    en_inund = R.valores(x, y, inund)
    km_edomex = peso[en_edomex].sum() / 1000
    km_inund = peso[en_edomex & en_inund].sum() / 1000
    tramo_inundado = np.zeros(len(tramos), bool)
    tramo_inundado[np.unique(tramo[en_inund])] = True
    del lon_l, lat_l, peso_l, tramo_l

    # --- Hospitales ----------------------------------------------------------
    print("Leyendo CLUES...")
    cl = pd.read_excel(a.clues, sheet_name=0, dtype=str)
    h = cl[(cl["NOMBRE TIPO ESTABLECIMIENTO"] == "DE HOSPITALIZACIÓN")
           & (cl["ESTATUS DE OPERACION"] == "EN OPERACION")].copy()
    h["lat"] = pd.to_numeric(h["LATITUD"], errors="coerce")
    h["lon"] = pd.to_numeric(h["LONGITUD"], errors="coerce")
    h = h[h["lat"].notna() & h["lon"].notna()].copy()
    h["tipo"] = np.where(h["NOMBRE DE LA INSTITUCION"] == "SERVICIOS MEDICOS PRIVADOS", "Privado", "Público")
    h["x"], h["y"] = lcc(h["lon"], h["lat"])
    h["inundable"] = R.valores(h["x"], h["y"], inund)
    h["en_buffer"] = R.valores(h["x"], h["y"], m_buffer)
    h["en_poligono_edomex"] = R.valores(h["x"], h["y"], m_edomex)
    he = h[h["CLAVE DE LA ENTIDAD"] == "15"]

    # --- Grafo y tiempos -----------------------------------------------------
    print("Construyendo grafo y calculando tiempos...")
    # Grafo base: entre dos nodos con tramos paralelos se conserva el más rápido.
    # 'seco' = tiempo del tramo paralelo más rápido SIN puntos inundables.
    G = nx.Graph()
    coord, seco = {}, {}
    for i, (fid, u, v, L, vel, p_ini, p_fin) in enumerate(tramos):
        coord.setdefault(u, p_ini)
        coord.setdefault(v, p_fin)
        if u == v:
            continue
        tmin = L / 1000 / vel * 60
        if not G.has_edge(u, v) or G[u][v]["t"] > tmin:
            G.add_edge(u, v, t=tmin)
        if not tramo_inundado[i]:
            k = (min(u, v), max(u, v))
            seco[k] = min(seco.get(k, np.inf), tmin)
    principal = max(nx.connected_components(G), key=len)
    G = G.subgraph(principal).copy()
    nodos = np.array(list(G.nodes))
    nxy = np.array([coord[n] for n in nodos])
    nX, nY = lcc(nxy[:, 0], nxy[:, 1])
    arbol = cKDTree(np.column_stack([nX, nY]))

    def asignar(xs, ys):
        d, j = arbol.query(np.column_stack([xs, ys]))
        return d, nodos[j]

    hb = h[h["en_buffer"]].copy()
    hb["dist_red_m"], hb["nodo"] = asignar(hb["x"], hb["y"])

    # Localidades rurales
    it = loc04.leer_iter(a.iter)
    it = it[~it["LOC"].isin(loc04.LOC_AGREGADOS)].copy()
    it["POBTOT_n"] = pd.to_numeric(it["POBTOT"], errors="coerce")
    it["lon"] = it["LONGITUD"].map(loc04.dms_a_decimal)
    it["lat"] = it["LATITUD"].map(loc04.dms_a_decimal)
    rur = it[it["lon"].notna() & (it["POBTOT_n"] < loc04.UMBRAL_RURAL)].copy()
    rur["x"], rur["y"] = lcc(rur["lon"], rur["lat"])
    rur["dist_red_m"], rur["nodo"] = asignar(rur["x"], rur["y"])
    vcon = a.vel_conexion / 3.6 * 60                      # m/min

    def tiempos(grafo, hosp):
        S = "__fuente__"
        g2 = grafo.copy()
        for n, d in hosp.groupby("nodo")["dist_red_m"].min().items():
            if n in g2:
                g2.add_edge(S, n, t=d / vcon)
        dist = nx.single_source_dijkstra_path_length(g2, S, weight="t")
        red = np.array([dist.get(n, np.inf) for n in rur["nodo"]])
        return red + rur["dist_red_m"].to_numpy() / vcon

    escenarios = {"Base": G}
    if a.escenario_inundacion:
        Gi = G.copy()
        quitar = []
        for u, v, d in Gi.edges(data=True):
            k = (min(u, v), max(u, v))
            if k in seco:
                d["t"] = seco[k]                           # queda un paralelo seco
            else:
                quitar.append((u, v))
        Gi.remove_edges_from(quitar)
        escenarios["Inundación TR100"] = Gi

    res_tiempos = {}
    for esc, g in escenarios.items():
        for tipo in ("Público", "Privado"):
            hs = hb[hb["tipo"] == tipo]
            if esc != "Base":
                hs = hs[~hs["inundable"]]                  # hospital inundable: no disponible
            res_tiempos[(esc, tipo)] = tiempos(g, hs)

    # ======================= IMPRESIÓN ======================================
    pob_rur = rur["POBTOT_n"].to_numpy()
    print("\n" + "=" * 72)
    print(" CAMINOS, HOSPITALES Y ACCESIBILIDAD - ESTADO DE MÉXICO (TR100)")
    print("=" * 72)
    print(f"  Superficie del Edomex (ráster 30 m, {n_mun} municipios): {area_edomex:,.1f} km2; "
          f"inundable TR100: {area_inund:,.1f} km2 ({100*area_inund/area_edomex:.2f} %)")
    print(f"\n  1. Km de caminos en el Estado de México            : {km_edomex:>12,.1f}")
    print(f"  2. Km de caminos en zonas inundables (TR100)       : {km_inund:>12,.1f}  "
          f"({100*km_inund/km_edomex:.2f} %)")

    print("\n  3-4. Hospitales en operación (CLUES, entidad 15)")
    print(f"     {'':10s}{'Total':>8s}{'Inundables':>12s}{'%':>8s}")
    for tipo in ("Público", "Privado", "Total"):
        s = he if tipo == "Total" else he[he["tipo"] == tipo]
        print(f"     {tipo:10s}{len(s):>8,d}{int(s['inundable'].sum()):>12,d}"
              f"{100*s['inundable'].mean():>7.2f}%")
    fuera = int((~he["en_poligono_edomex"]).sum())
    if fuera:
        print(f"     Nota: {fuera} hospital(es) con entidad 15 tienen coordenadas fuera del límite estatal.")

    print(f"\n  5. Población rural (< {loc04.UMBRAL_RURAL:,d} hab.): {len(rur):,d} localidades, "
          f"{int(pob_rur.sum()):,d} habitantes")
    print(f"     Hospitales usados para el cálculo (dentro de 20km_buffer): "
          f"{(hb['tipo']=='Público').sum():,d} públicos, {(hb['tipo']=='Privado').sum():,d} privados")
    dl = rur["dist_red_m"]
    print(f"     Distancia localidad -> red: mediana {dl.median():.0f} m, P90 {dl.quantile(.9):.0f} m, "
          f"máx {dl.max():,.0f} m (a {a.vel_conexion:g} km/h)")

    filas_res, filas_pct = [], []
    for (esc, tipo), t in res_tiempos.items():
        tab = agrupar(t, pob_rur)
        pc = percentiles_ponderados(t, pob_rur, PERCENTILES)
        print(f"\n     [{esc}] Hospital {tipo.lower()} más cercano")
        print(f"     {'Grupo':14s}{'Localidades':>12s}{'Habitantes':>14s}{'%':>9s}")
        for g, r in tab.iterrows():
            print(f"     {g:14s}{int(r.localidades):>12,d}{int(r.habitantes):>14,d}{r.porcentaje:>8.2f}%")
            filas_res.append({"escenario": esc, "hospital": tipo, "grupo": g,
                              "localidades": int(r.localidades), "habitantes": int(r.habitantes),
                              "porcentaje": round(r.porcentaje, 4)})
        print("     Percentiles del tiempo (min, ponderados por habitantes): "
              + ", ".join(f"{k} {v:.1f}" for k, v in pc.items()))
        filas_pct.append({"escenario": esc, "hospital": tipo, **{k: round(v, 2) for k, v in pc.items()}})

    # ======================= SALIDAS ========================================
    pd.DataFrame([{
        "km_caminos_edomex": round(km_edomex, 3), "km_caminos_inundables": round(km_inund, 3),
        "pct_km_inundables": round(100 * km_inund / km_edomex, 4),
        "hosp_publicos": int((he.tipo == "Público").sum()), "hosp_privados": int((he.tipo == "Privado").sum()),
        "hosp_total": len(he),
        "hosp_publicos_inundables": int(he[he.tipo == "Público"].inundable.sum()),
        "hosp_privados_inundables": int(he[he.tipo == "Privado"].inundable.sum()),
        "hosp_total_inundables": int(he.inundable.sum()),
        "pob_rural": int(pob_rur.sum()), "localidades_rurales": len(rur),
        "intervalo_muestreo_m": a.intervalo, "vel_conexion_kmh": a.vel_conexion,
    }]).to_csv(SALIDAS / "05_resumen_caminos_hospitales.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_res).to_csv(SALIDAS / "05_grupos_tiempo_hospital.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_pct).to_csv(SALIDAS / "05_percentiles_tiempo_hospital.csv", index=False, encoding="utf-8-sig")
    out = rur[["ENTIDAD", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "lon", "lat", "POBTOT", "dist_red_m"]].copy()
    out.insert(0, "CVEGEO", out.ENTIDAD + out.MUN + out.LOC)
    for (esc, tipo), t in res_tiempos.items():
        out[f"min_{tipo.lower()}_{'base' if esc == 'Base' else 'inund'}"] = np.round(t, 2)
    out.to_csv(SALIDAS / "05_localidades_rurales_tiempos.csv", index=False, encoding="utf-8-sig")
    he[["CLUES", "NOMBRE DE LA INSTITUCION", "tipo", "NOMBRE DE LA UNIDAD", "MUNICIPIO", "NIVEL ATENCION",
        "lon", "lat", "inundable"]].to_csv(SALIDAS / "05_hospitales_edomex.csv", index=False, encoding="utf-8-sig")
    print(f"\nArchivos guardados en: {SALIDAS}   [{time.time()-t0:.0f} s]")


if __name__ == "__main__":
    main()
