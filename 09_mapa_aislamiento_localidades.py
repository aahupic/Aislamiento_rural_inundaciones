# -*- coding: utf-8 -*-
"""
Script 09 - Aislamiento por Inundaciones: mapa de localidades aisladas y con retraso
==================================================================================

OBJETIVO
--------
  A. Agregar al GeoPackage del script 08 (08_mapa_tramos_inundables_hospitales.gpkg)
     tres capas de puntos de LOCALIDADES RURALES (< 2,500 habitantes, umbral
     INEGI), con su simbología QGIS (círculos de área proporcional a la
     población). Con --todas se incluyen todas las localidades.
       - localidades_aisladas        ROJO     sin ruta a ningún hospital público
       - localidades_retraso_15min   AMARILLO retraso >= 15 min
       - localidades_sin_cambio      GRIS     retraso < 15 min (incluye 0)
  B. Dibujar el mapa del Estado de México con esas capas, las vialidades
     (gris; azul si > 50 % inundable) y los límites municipales. Los
     hospitales NO se dibujan por defecto (se agregan con --con-hospitales:
     rombos negros = secos, naranjas = en píxel inundable); siguen en el
     GeoPackage para usarlos en QGIS. Leyenda
     arriba a la izquierda con fondo 60 % opaco. Imagen PNG a 300 ppp.

ESCENARIO QUE SE REPRESENTA
---------------------------
  El mismo del mapa 08: se cierran los tramos con MÁS DEL 50 % de su longitud
  en zona inundable TR100 y se mide el acceso a HOSPITALES PÚBLICOS. Es el
  renglón "> 50 % / Público / Rurales" del script 07 (1,170 localidades
  rurales aisladas, 595,991 habitantes). Umbral y tipo de hospital son parámetros
  (--umbral, --hospital).

  Clasificación de cada localidad (t_base = red completa; t_esc = escenario):
    - aislada        : t_esc = infinito (no existe ruta).
    - retraso >= 15  : t_esc - t_base >= 15 min.
    - sin cambio     : t_esc - t_base < 15 min. Se llama "sin cambio" porque el
                       retraso no alcanza el umbral de 15 min; el número de
                       localidades con retraso exactamente 0 se informa aparte.

FUENTES
-------
  08_mapa_tramos_inundables_hospitales.gpkg   municipios, vialidades (con
                                              pct_inund) y hospitales públicos
                                              para el mapa; aquí se agregan las
                                              3 capas de localidades.
  CENAPRED.zip -> CENAPRED_IPI_2015_Inund_Munici.zip
                                              límite estatal para recortar las
                                              vialidades del mapa.
  tr100_prcent.tif                            zonas inundables TR100: bandera
                                              'inundable' de cada localidad.
  iter_15csv20.zip                            localidades y población.
  ESTABLECIMIENTO_SALUD_202609.xlsx           hospitales usados en los tiempos.
  rnc2025Edomex.gpkg  (parte A)               el cálculo de tiempos necesita la
                                              topología de la RNC (UNION_INI /
                                              UNION_FIN) y la red completa de
                                              20 km, que no están en el gpkg 08.
                                              Si las 3 capas ya existen en el
                                              gpkg 08 con el mismo conjunto de
                                              localidades (rurales o todas), la
                                              parte A se omite y la RNC no se
                                              lee (--recalcular las rehace).

MÉTODO
------
  Parte A (cálculo; mismos criterios y funciones del script 07)
    1. % inundado por tramo con puntos cada <= 15 m (scripts 03 y 07).
    2. Grafo de la red completa (base) y grafo sin los tramos > umbral.
    3. Tiempo al hospital público más cercano en cada grafo: tiempo de tramo
       = longitud / VELOCIDAD RNC; conexión localidad/hospital -> nodo más
       cercano a 5 km/h; Dijkstra multi-origen.
    4. Clasificación y escritura de las 3 capas (EPSG:6365) con atributos
       CVEGEO, NOM_MUN, NOM_LOC, POBTOT, rural, inundable, min_base,
       min_escenario, retraso_min y diam_mm.
  Círculos proporcionales
    - ÁREA proporcional a la población (convención cartográfica: el ojo
      compara áreas): diámetro = DIAM_MAX * sqrt(POBTOT / POB_MAX), con
      POB_MAX = la localidad más poblada del conjunto representado.
      DIAM_MAX = 2.0 mm para rurales (su población va de 1 a 2,499 hab.) y
      7 mm con --todas (hasta 1.6 millones). Se impone un diámetro mínimo
      (DIAM_MIN_MM = 0.5 mm) para que las localidades muy pequeñas sigan
      siendo visibles; con rurales afecta a las de menos de ~160 hab.
      Con 4,215 localidades rurales en el estado, un diámetro máximo mayor
      hace que los círculos se encimen y oculten el patrón espacial.
    - En QGIS el tamaño se define por expresión sobre el campo diam_mm
      (unidades: mm), así que basta con abrir la capa.
    - En la imagen se dibujan primero los círculos grandes y encima los
      pequeños, con relleno semitransparente y borde fino, para que no se
      oculten unos a otros.
  Parte B (mapa)
    - Todas las capas se leen del gpkg 08, se proyectan a la Cónica Conforme
      de Lambert de INEGI (metros) y se dibujan con matplotlib.
    - Orden de dibujo: vialidades grises, vialidades azules, límites,
      círculos (gris, amarillo, rojo) y rombos de hospitales encima.
    - Leyenda arriba a la izquierda (framealpha = 0.6) con las categorías y
      tres círculos de referencia de tamaño (1,000 / 50,000 / 500,000 hab.).

USO
---
  python 09_mapa_aislamiento_localidades.py
  python 09_mapa_aislamiento_localidades.py --recalcular --umbral 50 --hospital Público
  python 09_mapa_aislamiento_localidades.py --ppp 300

REQUISITOS
----------
  numpy, pandas, networkx, scipy, matplotlib, Pillow, openpyxl.
  En la misma carpeta: scripts 02, 03, 04, 05, 07 y 08.

SALIDAS (carpeta salidas\\)
--------------------------
  08_mapa_tramos_inundables_hospitales.gpkg  (se le agregan las 3 capas)
  09_mapa_aislamiento_localidades.png        (300 ppp)
  09_resumen_localidades_mapa.csv            conteos por categoría
"""

import argparse
import importlib.util
import sqlite3
import struct
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection   # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402
from matplotlib.patches import PathPatch                            # noqa: E402
from matplotlib.path import Path as MplPath                         # noqa: E402
from scipy.spatial import cKDTree                                   # noqa: E402

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parent
SALIDAS = AQUI / "salidas"


def _importar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


red02 = _importar("red02", "02_red_vial_edomex.py")
loc04 = _importar("loc04", "04_localidades_iter_edomex.py")
s05 = _importar("s05", "05_inundacion_caminos_hospitales.py")
s07 = _importar("s07", "07_aislamiento_por_umbral_inundacion.py")
s08 = _importar("s08", "08_mapa_tramos_inundables_hospitales.py")

RUTAS = {
    "gpkg08": SALIDAS / "08_mapa_tramos_inundables_hospitales.gpkg",
    "cenapred": BASE / "CENAPRED.zip",
    "tr100": BASE / "pub-aislamiento" / "data" / "raw" / "tr100_prcent.tif",
    "iter": BASE / "inegi" / "iter_15csv20.zip",
    "clues": BASE / "ESTABLECIMIENTO_SALUD_202609.xlsx",
    "rnc": BASE / "rnc2025Edomex.gpkg",
}
NOMBRE = "09_mapa_aislamiento_localidades"

# Categorías: tabla, etiqueta, color de relleno, color de borde
CATEGORIAS = [
    ("localidades_sin_cambio", "Sin cambio (retraso < 15 min)", "#8c8c8c", "#5a5a5a"),
    ("localidades_retraso_15min", "Retraso ≥ 15 min", "#ffd21f", "#9c7c00"),
    ("localidades_aisladas", "Aislada (sin ruta al hospital)", "#e31a1c", "#7a0000"),
]
DIAM_MAX_MM = {"rurales": 2.0, "todas": 7.0}   # diámetro de la localidad más poblada
DIAM_MIN_MM = 0.5          # diámetro mínimo visible
ALFA_CIRCULOS = 0.75
REFERENCIAS_POB = {"rurales": [100, 1_000, 2_499], "todas": [1_000, 50_000, 500_000]}
MM_A_PT = 72 / 25.4


# ============================================================================
# Parte A: cálculo de categorías y escritura en el GeoPackage
# ============================================================================
def calcular_categorias(a):
    """Reproduce el cálculo del script 07 para un umbral y tipo de hospital."""
    import networkx  # noqa: F401  (lo usan las funciones del script 07)
    from matplotlib.path import Path as _P

    R = s05.Raster(a.tr100)
    inund = R.a == s05.VALOR_INUNDABLE
    anillos, _ = s05.anillos_edomex(a.cenapred)
    m_edomex = R.rasterizar(anillos)
    con = sqlite3.connect(f"file:{a.rnc.as_posix()}?mode=ro", uri=True)
    buffer = s05.anillos_gpkg_poligono(con.execute(f'SELECT geom FROM "{s07.CAPA_BUFFER}"').fetchone()[0])
    print("   % inundado por tramo (puntos cada 15 m)...")
    tr = s07.tramos_y_porcentaje(con, R, inund, m_edomex, s07.m03.INTERVALO)

    coord = pd.concat([
        tr[["u", "lon_ini", "lat_ini"]].set_axis(["n", "lon", "lat"], axis=1),
        tr[["v", "lon_fin", "lat_fin"]].set_axis(["n", "lon", "lat"], axis=1)]).drop_duplicates("n")
    cx, cy = s05.lcc(coord.lon, coord.lat)
    arbol, nodos = cKDTree(np.column_stack([cx, cy])), coord["n"].to_numpy()
    vcon = 5.0 / 3.6 * 60

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
    it["rural"] = (it["pob"] < loc04.UMBRAL_RURAL).astype(int)
    lx, ly = s05.lcc(it["lon"], it["lat"])
    it["inundable"] = R.valores(lx, ly, inund).astype(int)
    it["nodo"], it["con_min"] = asignar(lx, ly)

    cl = pd.read_excel(a.clues, sheet_name=0, dtype=str)
    h = cl[(cl["NOMBRE TIPO ESTABLECIMIENTO"] == "DE HOSPITALIZACIÓN")
           & (cl["ESTATUS DE OPERACION"] == "EN OPERACION")].copy()
    h["lat"] = pd.to_numeric(h["LATITUD"], errors="coerce")
    h["lon"] = pd.to_numeric(h["LONGITUD"], errors="coerce")
    h = h[h["lat"].notna() & h["lon"].notna()].copy()
    h["tipo"] = np.where(h["NOMBRE DE LA INSTITUCION"] == "SERVICIOS MEDICOS PRIVADOS", "Privado", "Público")
    h = h[h["tipo"] == a.hospital].copy()
    hx, hy = s05.lcc(h["lon"], h["lat"])
    dentro = np.zeros(len(h), bool)
    for anillo in buffer:
        dentro |= _P(anillo).contains_points(np.column_stack([hx, hy]))
    h = h[dentro].copy()
    hn, hc = asignar(*s05.lcc(h["lon"], h["lat"]))

    print("   Tiempos: red completa y escenario...")
    t = {}
    for nombre, abiertos in (("base", np.ones(len(tr), bool)), ("esc", (tr["pct_inund"] <= a.umbral).to_numpy())):
        G = s07.grafo(tr, abiertos)
        t[nombre] = s07.tiempos(G, hn, hc, it["nodo"].to_numpy(), it["con_min"].to_numpy())
    it["min_base"], it["min_escenario"] = t["base"], t["esc"]
    it["retraso_min"] = it["min_escenario"] - it["min_base"]
    it["categoria"] = np.select(
        [~np.isfinite(it["min_escenario"]), it["retraso_min"] >= s07.RETRASO_MIN],
        ["localidades_aisladas", "localidades_retraso_15min"], "localidades_sin_cambio")
    return it


def diametro_mm(pob, pob_max, modo):
    return np.maximum(DIAM_MIN_MM, DIAM_MAX_MM[modo] * np.sqrt(np.asarray(pob, float) / pob_max))


def _qml_circulo(color, borde):
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    rb, gb, bb = (int(borde[i:i + 2], 16) for i in (1, 3, 5))
    alfa = int(round(255 * ALFA_CIRCULOS))
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'><qgis version="3.28" '
            'styleCategories="Symbology"><renderer-v2 type="singleSymbol" symbollevels="0" '
            'enableorderby="1" forceraster="0"><symbols><symbol type="marker" name="0" alpha="1" '
            'force_rhr="0" clip_to_extent="1"><layer class="SimpleMarker" enabled="1" pass="0" locked="0">'
            '<Option type="Map"><Option type="QString" name="name" value="circle"/>'
            f'<Option type="QString" name="color" value="{r},{g},{b},{alfa}"/>'
            f'<Option type="QString" name="outline_color" value="{rb},{gb},{bb},255"/>'
            '<Option type="QString" name="outline_width" value="0.1"/>'
            '<Option type="QString" name="size" value="1"/>'
            '<Option type="QString" name="size_unit" value="MM"/></Option>'
            '<data_defined_properties><Option type="Map"><Option type="QString" name="name" value=""/>'
            '<Option type="Map" name="properties"><Option type="Map" name="size">'
            '<Option type="bool" name="active" value="true"/>'
            '<Option type="QString" name="expression" value="&quot;diam_mm&quot;"/>'
            '<Option type="int" name="type" value="3"/></Option></Option>'
            '<Option type="QString" name="type" value="collection"/></Option></data_defined_properties>'
            '</layer></symbol></symbols><orderby><orderByClause asc="0" nullsFirst="0">"POBTOT"'
            '</orderByClause></orderby></renderer-v2></qgis>')


def escribir_capas(ruta_gpkg, it, umbral, hospital, modo):
    c = sqlite3.connect(ruta_gpkg)
    pob_max = it["pob"].max()
    it = it.assign(diam_mm=diametro_mm(it["pob"], pob_max, modo))
    for tabla, etiqueta, color, borde in CATEGORIAS:
        c.execute(f'DROP TABLE IF EXISTS "{tabla}"')
        for t in ("gpkg_contents", "gpkg_geometry_columns"):
            c.execute(f"DELETE FROM {t} WHERE table_name = ?", (tabla,))
        c.execute("DELETE FROM layer_styles WHERE f_table_name = ?", (tabla,))
        c.execute(f'CREATE TABLE "{tabla}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom POINT, CVEGEO TEXT,'
                  ' NOM_MUN TEXT, NOM_LOC TEXT, POBTOT INTEGER, rural INTEGER, inundable INTEGER,'
                  ' min_base REAL, min_escenario REAL, retraso_min REAL, diam_mm REAL)')
        s = it[it["categoria"] == tabla].sort_values("pob", ascending=False)
        def _r(v):
            return None if not np.isfinite(v) else round(float(v), 2)
        c.executemany(f'INSERT INTO "{tabla}" (geom, CVEGEO, NOM_MUN, NOM_LOC, POBTOT, rural, inundable,'
                      ' min_base, min_escenario, retraso_min, diam_mm) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                      ((s08.gp_blob(6365, s08.wkb_punto(r.lon, r.lat), (r.lon, r.lon, r.lat, r.lat)),
                        r.CVEGEO, r.NOM_MUN, r.NOM_LOC, int(r.pob), int(r.rural), int(r.inundable),
                        _r(r.min_base), _r(r.min_escenario), _r(r.retraso_min), round(float(r.diam_mm), 3))
                       for r in s.itertuples()))
        env = (s.lon.min(), s.lon.max(), s.lat.min(), s.lat.max()) if len(s) else (0, 0, 0, 0)
        s08.registrar_capa(c, tabla, "POINT", 6365, env,
                           f"{etiqueta}. Localidades {'rurales (< 2,500 hab.)' if modo == 'rurales' else 'todas'}. "
                           f"Escenario: tramos > {umbral:g} % inundables cerrados; hospital {hospital.lower()}")
        s08.guardar_estilo(c, tabla, _qml_circulo(color, borde), f"Círculos proporcionales ({etiqueta})")
    c.commit()
    c.close()


# ============================================================================
# Parte B: lectura del GeoPackage y mapa
# ============================================================================
def _gp_wkb(blob):
    return memoryview(blob)[8 + red02._ENV_BYTES[(blob[3] >> 1) & 7]:]


def leer_puntos(c, tabla, campos):
    filas = c.execute(f'SELECT geom, {", ".join(campos)} FROM "{tabla}"').fetchall()
    xy = [struct.unpack_from("<dd", _gp_wkb(f[0]), 5) for f in filas]
    df = pd.DataFrame([f[1:] for f in filas], columns=campos)
    df["lon"], df["lat"] = (np.array([p[0] for p in xy]), np.array([p[1] for p in xy])) if xy else ([], [])
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUTAS.items():
        ap.add_argument(f"--{k}", type=Path, default=v)
    ap.add_argument("--umbral", type=float, default=50.0)
    ap.add_argument("--hospital", choices=["Público", "Privado"], default="Público")
    ap.add_argument("--recalcular", action="store_true", help="rehacer las 3 capas aunque existan")
    ap.add_argument("--todas", action="store_true",
                    help="incluir todas las localidades (por defecto solo rurales < 2,500 hab.)")
    ap.add_argument("--ppp", type=int, default=300)
    ap.add_argument("--con-hospitales", action="store_true", help="dibujar los hospitales en la imagen")
    a = ap.parse_args()
    t0 = time.time()

    c = sqlite3.connect(a.gpkg08)
    existentes = {r[0] for r in c.execute("SELECT table_name FROM gpkg_contents")}
    modo = "todas" if a.todas else "rurales"
    faltan = [t for t, *_ in CATEGORIAS if t not in existentes]
    if not faltan:
        # ¿Las capas existentes son del mismo conjunto? (rurales = ningún POBTOT >= 2,500)
        pmax = max(c.execute(f'SELECT COALESCE(MAX(POBTOT), 0) FROM "{t}"').fetchone()[0] for t, *_ in CATEGORIAS)
        if (pmax >= loc04.UMBRAL_RURAL) != (modo == "todas"):
            faltan = ["conjunto distinto"]
    c.close()
    if faltan or a.recalcular:
        print("Parte A: calculando categorías de localidades (requiere la RNC)...")
        for k in ("cenapred", "tr100", "iter", "clues", "rnc"):
            if not getattr(a, k).exists():
                raise SystemExit(f"No existe: {getattr(a, k)}")
        it = calcular_categorias(a)
        if modo == "rurales":
            it = it[it["rural"] == 1]
        escribir_capas(a.gpkg08, it, a.umbral, a.hospital, modo)
        print(f"   3 capas agregadas a {a.gpkg08}")
    else:
        print("Parte A omitida: las 3 capas ya están en el GeoPackage (usar --recalcular para rehacerlas).")

    print("Parte B: dibujando mapa...")
    c = sqlite3.connect(a.gpkg08)
    muns, _ = s08.municipios(a.cenapred)
    anillos = [r for _, _, g in muns for r in g]
    vial = c.execute("SELECT geom, inund_gt50 FROM vialidades_edomex").fetchall()
    hosp = leer_puntos(c, "hospitales_publicos_edomex", ["inundable"])
    locs = {t: leer_puntos(c, t, ["POBTOT", "diam_mm", "retraso_min"]) for t, *_ in CATEGORIAS}
    c.close()

    fig, ax = plt.subplots(figsize=(8.0, 9.0))
    limite = MplPath.make_compound_path(*[MplPath(r) for r in anillos])
    clip = PathPatch(limite, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    segs = {0: [], 1: []}
    for blob, b in vial:
        for p in red02.gpkg_a_lineas(blob):
            X, Y = s05.lcc(p[:, 0], p[:, 1])
            segs[b].append(np.column_stack([X, Y]))
    for b, color, ancho, z in ((0, s08.GRIS, 0.12, 2), (1, s08.AZUL, 0.35, 3)):
        lc = LineCollection(segs[b], colors=color, linewidths=ancho, zorder=z, alpha=0.8 if b else 1)
        ax.add_collection(lc)
        lc.set_clip_path(clip)
    ax.add_collection(PolyCollection(anillos, facecolors="none", edgecolors=s08.LIMITE, linewidths=0.35, zorder=4))

    # Círculos: todos juntos ordenados de mayor a menor población (los pequeños encima)
    todos = pd.concat([d.assign(cat=i) for i, (t, *_ ) in enumerate(CATEGORIAS) for d in [locs[t]]],
                      ignore_index=True).sort_values("POBTOT", ascending=False)
    x, y = s05.lcc(todos.lon, todos.lat)
    relleno = np.array([CATEGORIAS[i][2] for i in todos.cat])
    borde = np.array([CATEGORIAS[i][3] for i in todos.cat])
    ax.scatter(x, y, s=(todos.diam_mm.to_numpy() * MM_A_PT) ** 2, c=relleno, edgecolors=borde,
               linewidths=0.25, alpha=ALFA_CIRCULOS, zorder=5)
    if a.con_hospitales:
        for v, color in ((0, s08.NEGRO), (1, s08.NARANJA)):
            s = hosp[hosp.inundable == v]
            hx, hy = s05.lcc(s.lon, s.lat)
            ax.scatter(hx, hy, marker="D", s=20, c=color, edgecolors="white", linewidths=0.4, zorder=7)

    xy = np.vstack(anillos)
    mx = 0.03 * (xy[:, 0].max() - xy[:, 0].min())
    ax.set_xlim(xy[:, 0].min() - mx, xy[:, 0].max() + mx)
    ax.set_ylim(xy[:, 1].min() - mx, xy[:, 1].max() + mx)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    # Leyenda
    pob_max = max(d.POBTOT.max() for d in locs.values() if len(d))
    h_cat = [Line2D([], [], marker="o", ls="", ms=6, markerfacecolor=col, markeredgecolor=bor, alpha=ALFA_CIRCULOS,
                    label=f"{et} ({len(locs[t]):,d} loc.; {int(locs[t].POBTOT.sum()):,d} hab.)")
             for t, et, col, bor in reversed(CATEGORIAS)]
    h_tam = [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#333333",
                    ms=float(diametro_mm(p, pob_max, modo)) * MM_A_PT, label=f"{p:,d} hab.")
             for p in REFERENCIAS_POB[modo]]
    n_seco, n_inun = int((hosp.inundable == 0).sum()), int((hosp.inundable == 1).sum())
    h_base = [
        Line2D([], [], color=s08.GRIS, lw=1.0, label="Vialidades"),
        Line2D([], [], color=s08.AZUL, lw=1.6, label=f"Tramo con > {a.umbral:g} % inundable (TR100), cerrado"),
    ]
    if a.con_hospitales:
        h_base += [
            Line2D([], [], marker="D", ls="", color=s08.NEGRO, markeredgecolor="white", ms=5,
                   label=f"Hospital {a.hospital.lower()} seco ({n_seco})"),
            Line2D([], [], marker="D", ls="", color=s08.NARANJA, markeredgecolor="white", ms=5,
                   label=f"Hospital {a.hospital.lower()} en píxel inundable ({n_inun})"),
        ]
    h_base.append(Line2D([], [], color=s08.LIMITE, lw=0.8, label="Límite municipal"))
    ax.legend(handles=h_cat + h_tam + h_base, loc="upper left", fontsize=5.6, frameon=True, framealpha=0.6,
              edgecolor="#bbbbbb", title=(f"Localidades rurales (< 2,500 hab.): acceso a hospital {a.hospital.lower()}"
                     if modo == "rurales" else f"Localidades: acceso a hospital {a.hospital.lower()}"),
              title_fontsize=6.5, labelspacing=0.7, borderpad=0.7)

    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    bx, by, L = x1 - 0.06 * (x1 - x0) - 50000, y0 + 0.04 * (y1 - y0), 50000
    for k in range(5):
        ax.add_patch(plt.Rectangle((bx + k * L / 5, by), L / 5, 0.006 * (y1 - y0),
                                   facecolor="black" if k % 2 == 0 else "white", edgecolor="black", lw=0.5, zorder=8))
    for d, t in ((0, "0"), (L / 2, "25"), (L, "50 km")):
        ax.text(bx + d, by + 0.012 * (y1 - y0), t, ha="center", va="bottom", fontsize=6, zorder=8)
    nx_, ny_ = x1 - 0.06 * (x1 - x0), y1 - 0.12 * (y1 - y0)
    ax.annotate("N", xy=(nx_, ny_ + 0.06 * (y1 - y0)), xytext=(nx_, ny_), ha="center", fontsize=9,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2), zorder=8)

    quien = "Localidades rurales: aislamiento" if modo == "rurales" else "Aislamiento"
    ax.set_title(f"{quien} y retraso en el acceso a hospitales {a.hospital.lower()}s\n"
                 f"al cerrar los tramos con más del {a.umbral:g} % de su longitud en zona inundable (TR100)",
                 fontsize=9.5)
    ax.text(0.5, -0.012,
            "Fuentes: INEGI, Red Nacional de Caminos 2025, Censo 2020 (ITER) y Marco municipal (vía CENAPRED IPI 2015); "
            "CENAPRED, zonas inundables TR100;\nSecretaría de Salud, CLUES septiembre 2026. Círculos de área "
            "proporcional a la población. Proyección: Cónica Conforme de Lambert (ITRF92).",
            ha="center", va="top", fontsize=5.5, color="#444444", transform=ax.transAxes)
    SALIDAS.mkdir(exist_ok=True)
    ruta_png = SALIDAS / f"{NOMBRE}.png"
    fig.savefig(ruta_png, dpi=a.ppp, bbox_inches="tight")
    plt.close(fig)

    # Resumen
    filas = []
    for t, et, *_ in CATEGORIAS:
        d = locs[t]
        filas.append({"capa": t, "categoria": et, "localidades": len(d), "habitantes": int(d.POBTOT.sum()),
                      "localidades_retraso_cero": int((d.retraso_min.fillna(-1).abs() < 1e-9).sum())})
    res = pd.DataFrame(filas)
    res.to_csv(SALIDAS / "09_resumen_localidades_mapa.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 72)
    print(f" LOCALIDADES {modo.upper()} - tramos > {a.umbral:g} % inundables cerrados; hospital {a.hospital.lower()}")
    print("=" * 72)
    for r in res.itertuples():
        print(f"  {r.categoria:34s}{r.localidades:>8,d} loc.{r.habitantes:>14,d} hab.")
    sc = res[res.capa == "localidades_sin_cambio"].iloc[0]
    print(f"  (de 'sin cambio', {sc.localidades_retraso_cero:,d} localidades tienen retraso 0)")
    print(f"\n  Imagen ({a.ppp} ppp): {ruta_png}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
