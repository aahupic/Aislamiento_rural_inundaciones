# -*- coding: utf-8 -*-
"""
Script 08 - Aislamiento por Inundaciones: mapa de tramos inundables y hospitales
===============================================================================

OBJETIVO
--------
Generar dos productos del Estado de México:
  1. Un GeoPackage (.gpkg) con las capas del mapa, listo para abrir en QGIS
     con su simbología ya cargada (estilos guardados en la tabla
     layer_styles, que QGIS aplica automáticamente):
       - municipios_edomex          límites municipales (125 polígonos)
       - vialidades_edomex          tramos de la red vial con su % inundado y
                                    la bandera inund_gt50 (1 = más del 50 %)
       - hospitales_publicos_edomex hospitales públicos en operación con la
                                    bandera inundable (1 = píxel inundable)
  2. Una imagen PNG del mapa a 300 ppp con:
       - límites municipales,
       - vialidades en gris y, en azul, los tramos con más del 50 % de su
         longitud sobre zona inundable (periodo de retorno 100 años),
       - hospitales públicos en operación: rombo negro = seco, rombo naranja
         = en píxel inundable,
       - leyenda en la esquina superior izquierda, barra de escala y norte.

FUENTES
-------
  rnc2025Edomex.gpkg                 red vial (capa 20km_red_vial_buffer).
  CENAPRED.zip -> CENAPRED_IPI_2015_Inund_Munici.zip
                                     límites municipales del Edomex (CVE 15xxx).
  pub-aislamiento\\data\\raw\\tr100_prcent.tif
                                     zonas inundables TR100 (valor 0 = inundable).
  ESTABLECIMIENTO_SALUD_202609.xlsx  CLUES: hospitales.
  inegi\\iter_15csv20.zip             no interviene en el mapa; se acepta como
                                     parámetro solo para mantener la misma
                                     interfaz que los scripts 05 y 07.
  03_puntos_muestreo_15m.py          puntos cada <= 15 m sobre los tramos, a
                                     través de la función de % inundado del
                                     script 07 (que a su vez usa el 03).

MÉTODO
------
  1. % inundado por tramo (idéntico al script 07): puntos cada <= 15 m,
     % = puntos en píxel inundable / puntos del tramo. Tramo azul si
     % > 50 (mismo criterio "más del 50 %" del script 07).
  2. Tramos dibujados: los de la componente principal con al menos un punto
     dentro del Edomex. En la imagen se recortan al límite estatal (lo que
     sale del estado no se dibuja); en el GeoPackage se guardan completos.
  3. Hospitales: CLUES 'DE HOSPITALIZACIÓN', 'EN OPERACION', institución
     distinta de 'SERVICIOS MEDICOS PRIVADOS', entidad 15. Inundable = su
     coordenada cae en píxel TR100 = 0.
  4. Sistemas de referencia en el GeoPackage:
       - vialidades y hospitales: EPSG:6365 (grados, como la RNC y CLUES);
         las geometrías de los tramos se copian tal cual de la RNC.
       - municipios: Cónica Conforme de Lambert de INEGI (la del shapefile
         IPI y del ráster), registrada como SRS 100000.
     QGIS reproyecta al vuelo; la imagen se dibuja en la Cónica de Lambert
     (metros), que conserva formas y permite una barra de escala correcta.
  5. El GeoPackage se escribe con sqlite3 siguiendo el estándar OGC GPKG 1.3
     (tablas gpkg_spatial_ref_sys, gpkg_contents, gpkg_geometry_columns y
     geometrías en formato GeoPackageBinary). No requiere GDAL.

USO
---
  python 08_mapa_tramos_inundables_hospitales.py
  python 08_mapa_tramos_inundables_hospitales.py --umbral 50 --ppp 300

REQUISITOS
----------
  numpy, pandas, networkx, scipy, matplotlib, Pillow, openpyxl.
  En la misma carpeta: scripts 02, 03, 04, 05 y 07.

SALIDAS (carpeta salidas\\)
--------------------------
  08_mapa_tramos_inundables_hospitales.gpkg
  08_mapa_tramos_inundables_hospitales.png   (300 ppp)
"""

import argparse
import importlib.util
import io
import sqlite3
import struct
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import PathPatch                         # noqa: E402
from matplotlib.path import Path as MplPath                      # noqa: E402

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parent
SALIDAS = AQUI / "salidas"


def _importar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


red02 = _importar("red02", "02_red_vial_edomex.py")
s05 = _importar("s05", "05_inundacion_caminos_hospitales.py")
s07 = _importar("s07", "07_aislamiento_por_umbral_inundacion.py")   # usa el script 03

RUTAS = dict(s07.RUTAS)
NOMBRE = "08_mapa_tramos_inundables_hospitales"

# Colores y estilos (mismos en la imagen y en los estilos QGIS)
GRIS, AZUL, NEGRO, NARANJA = "#9a9a9a", "#1f5fbf", "#000000", "#ff8c00"
LIMITE = "#4d4d4d"


# ============================================================================
# 1. Datos
# ============================================================================
def municipios(ruta_cenapred):
    """Polígonos (lista de anillos LCC), clave y nombre de los municipios 15xxx."""
    with zipfile.ZipFile(ruta_cenapred) as z:
        nombre = next(n for n in z.namelist() if n.endswith(s05.IPI_ZIP))
        interior = zipfile.ZipFile(io.BytesIO(z.read(nombre)))
    shp = next(n for n in interior.namelist() if n.lower().endswith(".shp"))
    dbf = interior.read(shp[:-4] + ".dbf")
    prj = interior.read(shp[:-4] + ".prj").decode("latin-1")
    claves = s05._dbf_campo(dbf, "CVE_MPIO")
    nombres = s05._dbf_campo(dbf, "NOM_MUN")
    geoms = s05._shp_poligonos(interior.read(shp))
    sel = [(k, n, g) for k, n, g in zip(claves, nombres, geoms) if k.startswith("15") and len(k) == 5]
    return sel, prj


def hospitales_publicos(ruta_clues, R, inund):
    cl = pd.read_excel(ruta_clues, sheet_name=0, dtype=str)
    h = cl[(cl["NOMBRE TIPO ESTABLECIMIENTO"] == "DE HOSPITALIZACIÓN")
           & (cl["ESTATUS DE OPERACION"] == "EN OPERACION")
           & (cl["NOMBRE DE LA INSTITUCION"] != "SERVICIOS MEDICOS PRIVADOS")
           & (cl["CLAVE DE LA ENTIDAD"] == "15")].copy()
    h["lat"] = pd.to_numeric(h["LATITUD"], errors="coerce")
    h["lon"] = pd.to_numeric(h["LONGITUD"], errors="coerce")
    h = h[h["lat"].notna() & h["lon"].notna()].reset_index(drop=True)
    h["x"], h["y"] = s05.lcc(h["lon"], h["lat"])
    h["inundable"] = R.valores(h["x"], h["y"], inund).astype(int)
    return h


# ============================================================================
# 2. Escritura del GeoPackage (OGC GPKG 1.3, sin GDAL)
# ============================================================================
def gp_blob(srs_id, wkb, env):
    """Encabezado GeoPackageBinary (little endian, envolvente XY) + WKB."""
    minx, maxx, miny, maxy = env
    return b"GP" + bytes([0, 0b00000011]) + struct.pack("<i4d", srs_id, minx, maxx, miny, maxy) + wkb


def wkb_punto(x, y):
    return struct.pack("<BIdd", 1, 1, x, y)


def wkb_multipoligono(anillos):
    """Cada anillo exterior se escribe como un polígono (los municipios no
    tienen huecos; verificado en el script 05)."""
    out = struct.pack("<BII", 1, 6, len(anillos))
    for r in anillos:
        out += struct.pack("<BIII", 1, 3, 1, len(r)) + np.ascontiguousarray(r, "<f8").tobytes()
    return out


def crear_gpkg(ruta, prj_lcc):
    if ruta.exists():
        ruta.unlink()
    c = sqlite3.connect(ruta)
    c.execute("PRAGMA application_id = 1196444487")       # 'GPKG'
    c.execute("PRAGMA user_version = 10300")
    c.executescript("""
    CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
      organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,
      definition TEXT NOT NULL, description TEXT);
    CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
      identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME NOT NULL,
      min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER);
    CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL,
      geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL,
      m TINYINT NOT NULL, CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name));
    CREATE TABLE layer_styles (id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT,
      f_table_schema TEXT, f_table_name TEXT, f_geometry_column TEXT, styleName TEXT,
      styleQML TEXT, styleSLD TEXT, useAsDefault BOOLEAN, description TEXT, owner TEXT,
      ui TEXT, update_time DATETIME);
    """)
    srs = [
        ("Undefined cartesian SRS", -1, "NONE", -1, "undefined", None),
        ("Undefined geographic SRS", 0, "NONE", 0, "undefined", None),
        ("WGS 84 geodetic", 4326, "EPSG", 4326,
         'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
         'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]', None),
        ("Mexico ITRF2008", 6365, "EPSG", 6365,
         'GEOGCS["Mexico ITRF2008",DATUM["Mexico_ITRF2008",SPHEROID["GRS 1980",6378137,'
         '298.257222101]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
         'AUTHORITY["EPSG","6365"]]', None),
        ("ITRF92 Conica Conforme de Lambert (INEGI)", 100000, "NONE", 100000, prj_lcc,
         "LCC lat0 12, lon0 -102, paralelos 17.5 y 29.5, FE 2500000"),
    ]
    c.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", srs)
    return c


def registrar_capa(c, tabla, tipo_geom, srs, env, descripcion):
    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    c.execute("INSERT INTO gpkg_contents VALUES (?,?,?,?,?,?,?,?,?,?)",
              (tabla, "features", tabla, descripcion, ahora, env[0], env[2], env[1], env[3], srs))
    c.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,0,0)", (tabla, "geom", tipo_geom, srs))


# ---------------------------------------------------------------- estilos QGIS
def _simbolo_linea(nombre, color, ancho_mm):
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return (f'<symbol type="line" name="{nombre}" alpha="1" force_rhr="0" clip_to_extent="1">'
            f'<layer class="SimpleLine" enabled="1" pass="0" locked="0">'
            f'<Option type="Map"><Option type="QString" name="line_color" value="{r},{g},{b},255"/>'
            f'<Option type="QString" name="line_width" value="{ancho_mm}"/>'
            f'<Option type="QString" name="line_width_unit" value="MM"/>'
            f'<Option type="QString" name="capstyle" value="round"/></Option></layer></symbol>')


def _simbolo_rombo(nombre, color, tam_mm):
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return (f'<symbol type="marker" name="{nombre}" alpha="1" force_rhr="0" clip_to_extent="1">'
            f'<layer class="SimpleMarker" enabled="1" pass="0" locked="0">'
            f'<Option type="Map"><Option type="QString" name="name" value="diamond"/>'
            f'<Option type="QString" name="color" value="{r},{g},{b},255"/>'
            f'<Option type="QString" name="outline_color" value="255,255,255,255"/>'
            f'<Option type="QString" name="outline_width" value="0.2"/>'
            f'<Option type="QString" name="size" value="{tam_mm}"/>'
            f'<Option type="QString" name="size_unit" value="MM"/></Option></layer></symbol>')


def _qml_categorizado(campo, cats, simbolos):
    cs = "".join(f'<category value="{v}" label="{et}" symbol="{i}" render="true"/>'
                 for i, (v, et) in enumerate(cats))
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'><qgis version="3.28" '
            'styleCategories="Symbology"><renderer-v2 type="categorizedSymbol" attr="' + campo +
            '" symbollevels="0" enableorderby="0" forceraster="0"><categories>' + cs +
            '</categories><symbols>' + "".join(simbolos) + '</symbols></renderer-v2></qgis>')


def _qml_contorno():
    r, g, b = (int(LIMITE[i:i + 2], 16) for i in (1, 3, 5))
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'><qgis version="3.28" '
            'styleCategories="Symbology"><renderer-v2 type="singleSymbol" symbollevels="0" '
            'enableorderby="0" forceraster="0"><symbols><symbol type="fill" name="0" alpha="1" '
            'force_rhr="0" clip_to_extent="1"><layer class="SimpleFill" enabled="1" pass="0" '
            'locked="0"><Option type="Map"><Option type="QString" name="style" value="no"/>'
            f'<Option type="QString" name="outline_color" value="{r},{g},{b},255"/>'
            '<Option type="QString" name="outline_width" value="0.2"/>'
            '<Option type="QString" name="outline_width_unit" value="MM"/></Option></layer>'
            '</symbol></symbols></renderer-v2></qgis>')


def guardar_estilo(c, tabla, qml, desc):
    c.execute("INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, f_geometry_column,"
              " styleName, styleQML, styleSLD, useAsDefault, description, owner, ui, update_time)"
              " VALUES ('', '', ?, 'geom', ?, ?, '', 1, ?, '', NULL, ?)",
              (tabla, tabla, qml, desc, datetime.now(timezone.utc).isoformat()))


# ============================================================================
# 3. Programa principal
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUTAS.items():
        ap.add_argument(f"--{k}", type=Path, default=v)
    ap.add_argument("--umbral", type=float, default=50.0, help="%% inundado para marcar en azul (defecto 50)")
    ap.add_argument("--ppp", type=int, default=300, help="resolución de la imagen (defecto 300)")
    a = ap.parse_args()
    for k in ("gpkg", "cenapred", "tr100", "clues"):
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()
    SALIDAS.mkdir(exist_ok=True)

    print("1/4 Ráster, límites y % inundado por tramo (puntos cada 15 m)...")
    R = s05.Raster(a.tr100)
    inund = R.a == s05.VALOR_INUNDABLE
    muns, prj = municipios(a.cenapred)
    anillos = [r for _, _, g in muns for r in g]
    m_edomex = R.rasterizar(anillos)
    src = sqlite3.connect(f"file:{a.gpkg.as_posix()}?mode=ro", uri=True)
    tr = s07.tramos_y_porcentaje(src, R, inund, m_edomex, s07.m03.INTERVALO)
    tr = tr[tr["en_edomex"]].reset_index(drop=True)
    tr["inund_gt50"] = (tr["pct_inund"] > a.umbral).astype(int)

    print("2/4 Hospitales públicos...")
    h = hospitales_publicos(a.clues, R, inund)

    print("3/4 Escribiendo GeoPackage...")
    ruta_gpkg = SALIDAS / f"{NOMBRE}.gpkg"
    c = crear_gpkg(ruta_gpkg, prj)

    # Municipios (LCC, SRS 100000)
    c.execute('CREATE TABLE municipios_edomex (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom MULTIPOLYGON,'
              ' CVE_MPIO TEXT, NOM_MUN TEXT)')
    env_tot = [np.inf, -np.inf, np.inf, -np.inf]
    for k, n, g in muns:
        xy = np.vstack(g)
        env = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
        env_tot = [min(env_tot[0], env[0]), max(env_tot[1], env[1]), min(env_tot[2], env[2]), max(env_tot[3], env[3])]
        c.execute("INSERT INTO municipios_edomex (geom, CVE_MPIO, NOM_MUN) VALUES (?,?,?)",
                  (gp_blob(100000, wkb_multipoligono(g), env), k, n))
    registrar_capa(c, "municipios_edomex", "MULTIPOLYGON", 100000, env_tot,
                   "Límites municipales del Edomex (CENAPRED IPI 2015)")

    # Vialidades (EPSG:6365; geometría copiada de la RNC)
    c.execute('CREATE TABLE vialidades_edomex (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom MULTILINESTRING,'
              ' fid_rnc INTEGER, longitud_m REAL, velocidad REAL, pct_inund REAL, inund_gt50 INTEGER)')
    quiero = set(tr["fid"].tolist())
    blobs = {f: g for f, g in src.execute(f'SELECT fid, geom FROM "{s07.CAPA_RED}"') if f in quiero}
    c.executemany("INSERT INTO vialidades_edomex (geom, fid_rnc, longitud_m, velocidad, pct_inund, inund_gt50)"
                  " VALUES (?,?,?,?,?,?)",
                  ((blobs[f], int(f), round(L, 2), v, round(p, 2), int(b)) for f, L, v, p, b in
                   zip(tr.fid, tr.L, tr.vel, tr.pct_inund, tr.inund_gt50)))
    lon_all = np.r_[tr.lon_ini, tr.lon_fin]
    lat_all = np.r_[tr.lat_ini, tr.lat_fin]
    registrar_capa(c, "vialidades_edomex", "MULTILINESTRING", 6365,
                   (lon_all.min(), lon_all.max(), lat_all.min(), lat_all.max()),
                   f"Tramos RNC 2025 que tocan el Edomex; inund_gt50 = % inundado TR100 > {a.umbral:g}")

    # Hospitales públicos (EPSG:6365)
    c.execute('CREATE TABLE hospitales_publicos_edomex (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom POINT,'
              ' CLUES TEXT, institucion TEXT, unidad TEXT, municipio TEXT, nivel TEXT, inundable INTEGER)')
    c.executemany("INSERT INTO hospitales_publicos_edomex (geom, CLUES, institucion, unidad, municipio, nivel,"
                  " inundable) VALUES (?,?,?,?,?,?,?)",
                  ((gp_blob(6365, wkb_punto(x, y), (x, x, y, y)), r.CLUES, r["NOMBRE DE LA INSTITUCION"],
                    r["NOMBRE DE LA UNIDAD"], r.MUNICIPIO, r["NIVEL ATENCION"], int(r.inundable))
                   for (x, y), (_, r) in zip(zip(h.lon, h.lat), h.iterrows())))
    registrar_capa(c, "hospitales_publicos_edomex", "POINT", 6365,
                   (h.lon.min(), h.lon.max(), h.lat.min(), h.lat.max()),
                   "Hospitales públicos en operación (CLUES 09-2026); inundable = píxel TR100")

    # Estilos QGIS por defecto
    n_seco, n_inun = int((h.inundable == 0).sum()), int((h.inundable == 1).sum())
    guardar_estilo(c, "municipios_edomex", _qml_contorno(), "Contorno gris oscuro")
    guardar_estilo(c, "vialidades_edomex", _qml_categorizado(
        "inund_gt50", [(0, "Vialidades"), (1, f"Tramo con > {a.umbral:g} % inundable (TR100)")],
        [_simbolo_linea("0", GRIS, 0.1), _simbolo_linea("1", AZUL, 0.35)]), "Gris / azul")
    guardar_estilo(c, "hospitales_publicos_edomex", _qml_categorizado(
        "inundable", [(0, f"Hospital público seco ({n_seco})"),
                      (1, f"Hospital público en píxel inundable ({n_inun})")],
        [_simbolo_rombo("0", NEGRO, 2.6), _simbolo_rombo("1", NARANJA, 2.6)]), "Rombos negro / naranja")
    c.commit()
    c.close()

    print("4/4 Dibujando mapa...")
    fig, ax = plt.subplots(figsize=(8.0, 9.0))
    # Trayectoria compuesta del límite estatal, para recortar las vialidades
    limite = MplPath.make_compound_path(*[MplPath(r) for r in anillos])
    clip = PathPatch(limite, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)

    segs = {0: [], 1: []}
    for f, b in zip(tr.fid, tr.inund_gt50):
        for p in red02.gpkg_a_lineas(blobs[f]):
            X, Y = s05.lcc(p[:, 0], p[:, 1])
            segs[b].append(np.column_stack([X, Y]))
    for b, color, ancho, z in ((0, GRIS, 0.12, 2), (1, AZUL, 0.35, 3)):
        lc = LineCollection(segs[b], colors=color, linewidths=ancho, zorder=z, capstyle="round")
        ax.add_collection(lc)
        lc.set_clip_path(clip)
    ax.add_collection(PolyCollection([r for r in anillos], facecolors="none", edgecolors=LIMITE,
                                     linewidths=0.35, zorder=4))
    for v, color in ((0, NEGRO), (1, NARANJA)):
        s = h[h.inundable == v]
        ax.scatter(s.x, s.y, marker="D", s=22, c=color, edgecolors="white", linewidths=0.4, zorder=6)

    xy = np.vstack(anillos)
    mx = 0.03 * (xy[:, 0].max() - xy[:, 0].min())
    ax.set_xlim(xy[:, 0].min() - mx, xy[:, 0].max() + mx)
    ax.set_ylim(xy[:, 1].min() - mx, xy[:, 1].max() + mx)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    km_azul = tr.loc[tr.inund_gt50 == 1, "L"].sum() / 1000
    leyenda = [
        Line2D([], [], color=LIMITE, lw=0.8, label="Límite municipal"),
        Line2D([], [], color=GRIS, lw=1.0, label="Vialidades"),
        Line2D([], [], color=AZUL, lw=1.6, label=f"Tramo con > {a.umbral:g} % de su longitud\nen zona inundable (TR100)"),
        Line2D([], [], marker="D", ls="", color=NEGRO, markeredgecolor="white", ms=6,
               label=f"Hospital público seco ({n_seco})"),
        Line2D([], [], marker="D", ls="", color=NARANJA, markeredgecolor="white", ms=6,
               label=f"Hospital público en píxel inundable ({n_inun})"),
    ]
    ax.legend(handles=leyenda, loc="lower right", fontsize=7, frameon=True, framealpha=0.6,
              edgecolor="#bbbbbb", title="Estado de México", title_fontsize=8)

    # Barra de escala (50 km) y norte (la LCC está en metros)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    bx, by, L = x1 - 0.5 * (x1 - x0) - 50000, y0 + 0.04 * (y1 - y0), 50000
    for k in range(5):
        ax.add_patch(plt.Rectangle((bx + k * L / 5, by), L / 5, 0.006 * (y1 - y0),
                                   facecolor="black" if k % 2 == 0 else "white", edgecolor="black", lw=0.5, zorder=7))
    for d, t in ((0, "0"), (L / 2, "25"), (L, "50 km")):
        ax.text(bx + d, by + 0.012 * (y1 - y0), t, ha="center", va="bottom", fontsize=6, zorder=7)
    nx_, ny_ = x1 - 0.06 * (x1 - x0), y1 - 0.12 * (y1 - y0)
    ax.annotate("N", xy=(nx_, ny_ + 0.06 * (y1 - y0)), xytext=(nx_, ny_), ha="center", fontsize=9,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2), zorder=7)

    ax.set_title(f"Tramos viales con más del {a.umbral:g} % de su longitud en zona inundable (TR100)\n"
                 "y hospitales públicos en operación", fontsize=9.5)
    ax.text(0.5, -0.012,
             "Fuentes: INEGI, Red Nacional de Caminos 2025 y Marco municipal (vía CENAPRED IPI 2015); "
             "CENAPRED, zonas inundables TR100;\nSecretaría de Salud, CLUES septiembre 2026. "
             "Proyección: Cónica Conforme de Lambert (ITRF92). Puntos de muestreo cada 15 m.",
             ha="center", va="top", fontsize=5.5, color="#444444", transform=ax.transAxes)
    ruta_png = SALIDAS / f"{NOMBRE}.png"
    fig.savefig(ruta_png, dpi=a.ppp, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 70)
    print(" MAPA DE TRAMOS INUNDABLES Y HOSPITALES PÚBLICOS - ESTADO DE MÉXICO")
    print("=" * 70)
    print(f"  Municipios: {len(muns)}")
    print(f"  Tramos dibujados (tocan el Edomex): {len(tr):,d}; con > {a.umbral:g} % inundable: "
          f"{int(tr.inund_gt50.sum()):,d} ({km_azul:,.1f} km de los tramos completos)")
    print(f"  Hospitales públicos en operación: {len(h)} (secos {n_seco}, en píxel inundable {n_inun})")
    print(f"  GeoPackage: {ruta_gpkg}")
    print(f"  Imagen ({a.ppp} ppp): {ruta_png}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
