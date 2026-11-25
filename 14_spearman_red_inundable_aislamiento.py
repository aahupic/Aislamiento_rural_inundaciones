# -*- coding: utf-8 -*-
"""
Script 14 - Aislamiento por Inundaciones: correlación de Spearman municipal
=========================================================================

OBJETIVO
--------
Medir la asociación monótona, a nivel MUNICIPAL, entre:

  X = % de la red vial municipal en zona inundable (TR100)
  Y = % de la población rural del municipio que queda aislada de los
      hospitales públicos operativos en el escenario de inundación

  H0: rho_s = 0 (no hay asociación monótona).
  H1: rho_s != 0 (bilateral).

Se usa Spearman (correlación de rangos) porque ambas variables son
porcentajes acotados, asimétricos y con valores extremos (muchos municipios
con 0 % de población aislada), y la relación esperada es monótona pero no
necesariamente lineal.

DEFINICIÓN DE LAS VARIABLES
---------------------------
  Unidad de análisis: municipio del Estado de México (125; límites del
  shapefile CENAPRED IPI 2015).

  X (red inundable):
    - Red: tramos válidos de la componente principal de la RNC 2025 (mismos
      criterios de los scripts 02, 03 y 07).
    - Puntos de muestreo cada <= 15 m sobre cada tramo (script 03); cada
      punto representa L/n metros de su tramo.
    - Cada punto se asigna al municipio donde cae (ráster de municipios
      alineado al TR100, 30 m). Así, un tramo que cruza un límite municipal
      reparte su longitud entre ambos municipios.
    - X = km de red del municipio en píxel inundable / km de red del
      municipio x 100.

  Y (población rural aislada):
    - Localidades rurales (< 2,500 hab., ITER 2020) y escenario idénticos a
      los scripts 09, 10, 12 y 13: se cierran los tramos con > 50 % de su
      longitud inundable; destino = hospitales públicos operativos de la
      franja de 20 km; aislada = sin ruta.
    - El municipio de la localidad es el de su clave del ITER (MUN).
    - Y = habitantes rurales aislados / habitantes rurales del municipio
      x 100.
    - Municipios sin población rural (Y indefinida) se excluyen de la prueba
      y se informan.

CÁLCULO
-------
  1. Rangos de X y de Y por separado (1 = menor); empates = rango promedio.
  2. rho_s = correlación de Pearson entre los rangos (definición general,
     válida con empates; la fórmula 1 − 6Σd²/(n(n²−1)) solo es exacta sin
     empates y aquí hay muchos en Y = 0 %; se reporta también como
     referencia).
  3. Significancia: t = rho·√((n−2)/(1−rho²)), gl = n − 2,
     p bilateral = P(|T| > |t|). Es lo que hace scipy.stats.spearmanr.
  4. IC 95 % por transformación z de Fisher con el error estándar de
     Fieller (1.06/√(n−3), recomendado para Spearman):
     tanh( atanh(rho) ± 1.96 · 1.06/√(n−3) ).
  5. Fuerza (guía usual): |rho| < 0.1 nula, 0.1–0.3 débil, 0.3–0.5
     moderada, 0.5–0.7 fuerte, >= 0.7 muy fuerte.
  6. Sensibilidad (solo informativa): la misma prueba excluyendo los
     municipios con Y = 0, para ver si la asociación depende de ellos.

ADVERTENCIA METODOLÓGICA
------------------------
  - Es una asociación ecológica (entre municipios): no implica que dentro de
    cada municipio la población más expuesta sea la que se aísla.
  - Los municipios vecinos no son independientes (autocorrelación
    espacial); el valor p puede exagerarse.
  - Municipios con pocas localidades rurales tienen Y muy variable (0 % o
    100 % con una sola localidad).

FUENTES
-------
  rnc2025Edomex.gpkg; CENAPRED.zip (IPI 2015, municipios); tr100_prcent.tif;
  inegi\\iter_15csv20.zip; ESTABLECIMIENTO_SALUD_202609.xlsx; scripts 02, 03,
  04, 05, 07, 08, 11 y 12 (se reutilizan sus funciones).

USO
---
  python 14_spearman_red_inundable_aislamiento.py
  python 14_spearman_red_inundable_aislamiento.py --umbral 50 --alfa 0.05

SALIDAS (carpeta salidas\\)
--------------------------
  14_spearman_red_inundable_aislamiento.xlsx   hojas:
      Resumen      resultados, interpretación y gráfico de dispersión
      Spearman     tabla municipal con X, Y, rangos y cálculo con fórmulas
      Municipios   km de red y población rural por municipio (insumos)
      Localidades  una fila por localidad rural (fuente de Y)
      Python       resultados de scipy para comparar
      Metodo       definiciones y fuentes
  14_municipios_spearman.csv
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
from scipy import stats

AQUI = Path(__file__).resolve().parent
SALIDAS = AQUI / "salidas"


def _importar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


red02 = _importar("red02", "02_red_vial_edomex.py")
s05 = _importar("s05", "05_inundacion_caminos_hospitales.py")
s07 = _importar("s07", "07_aislamiento_por_umbral_inundacion.py")
s08 = _importar("s08", "08_mapa_tramos_inundables_hospitales.py")
s11 = _importar("s11", "11_tramos_criticos_cierre_individual.py")
s12 = _importar("s12", "12_chi_cuadrada_aislamiento_inundable.py")
RUTAS = dict(s12.RUTAS)
NOMBRE = "14_spearman_red_inundable_aislamiento"


# ============================================================================
# 1. X: km de red e inundables por municipio (puntos cada 15 m)
# ============================================================================
def red_por_municipio(a, muns):
    R = s05.Raster(a.tr100)
    inund = R.a == s05.VALOR_INUNDABLE
    etiq = s11.etiquetas_municipio(R, muns)
    con = sqlite3.connect(f"file:{a.gpkg.as_posix()}?mode=ro", uri=True)
    filas = con.execute(f'SELECT ESTATUS, CONDICION, CIRCULA, UNION_INI, UNION_FIN, geom '
                        f'FROM "{s07.CAPA_RED}"').fetchall()
    validos = [f for f in filas if not s07.m03.excluido(f[0], f[1], f[2])]
    G = nx.Graph()
    G.add_edges_from((f[3], f[4]) for f in validos)
    comp = max(nx.connected_components(G), key=len)
    km = np.zeros(len(muns) + 1)
    km_inund = np.zeros(len(muns) + 1)
    lon_l, lat_l, w_l = [], [], []
    for f in validos:
        if f[3] not in comp or f[3] == f[4]:
            continue
        L, pts = red02.puntos_a_lo_largo(red02.gpkg_a_lineas(f[5]), a.intervalo)
        lon_l.append(pts[:, 0]); lat_l.append(pts[:, 1]); w_l.append(np.full(len(pts), L / len(pts)))
    x, y = s05.lcc(np.concatenate(lon_l), np.concatenate(lat_l))
    w = np.concatenate(w_l)
    lab = R.valores(x, y, etiq)
    ii = R.valores(x, y, inund)
    km += np.bincount(lab, weights=w, minlength=len(km)) / 1000
    km_inund += np.bincount(lab, weights=w * ii, minlength=len(km)) / 1000
    return km[1:], km_inund[1:]


# ============================================================================
# 2. Estadísticos
# ============================================================================
def spearman(x, y):
    n = len(x)
    rho, p = stats.spearmanr(x, y)
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    se = 1.06 / math.sqrt(n - 3)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    return {"n": n, "rho_spearman": float(rho), "t": t, "gl": n - 2, "p_valor": float(p),
            "ic95_inf": math.tanh(math.atanh(rho) - 1.96 * se),
            "ic95_sup": math.tanh(math.atanh(rho) + 1.96 * se),
            "rho_formula_d2": 1 - 6 * float(((rx - ry) ** 2).sum()) / (n * (n ** 2 - 1)),
            "empates_x": int(n - len(np.unique(x))), "empates_y": int(n - len(np.unique(y)))}


def fuerza(r):
    r = abs(r)
    return ("nula" if r < 0.1 else "débil" if r < 0.3 else "moderada" if r < 0.5
            else "fuerte" if r < 0.7 else "muy fuerte")


# ============================================================================
# 3. Excel
# ============================================================================
def escribir_excel(ruta, mun, it, res, res_s, info, alfa):
    from openpyxl import Workbook
    from openpyxl.chart import Reference, ScatterChart, Series
    from openpyxl.chart.marker import Marker
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    F = "Arial"
    f_n, f_b, f_t = Font(name=F, size=10), Font(name=F, size=10, bold=True), Font(name=F, size=13, bold=True)
    f_i = Font(name=F, size=9, italic=True, color="555555")
    f_in, f_link = Font(name=F, size=10, color="0000FF"), Font(name=F, size=10, color="008000")
    hdr, clave = PatternFill("solid", start_color="D9E1F2"), PatternFill("solid", start_color="FFF2CC")
    fino = Side(style="thin", color="999999")
    caja = Border(left=fino, right=fino, top=fino, bottom=fino)
    wb = Workbook()

    def encabezado(ws, cols, anchos):
        ws.append(cols)
        for c in ws[1]:
            c.font, c.fill, c.alignment = f_b, hdr, Alignment(wrap_text=True, vertical="center")
        for i, w in enumerate(anchos):
            ws.column_dimensions[chr(65 + i)].width = w
        ws.freeze_panes = "A2"

    # ------------------------------------------------------------- Localidades
    wl = wb.active
    wl.title = "Localidades"
    encabezado(wl, ["CVEGEO", "CVE_MPIO", "Municipio", "Localidad", "Población 2020", "Inundable (1=sí)",
                    "Aislada (1=sí)"], [14, 10, 26, 34, 12, 11, 11])
    for r in it.itertuples():
        wl.append([r.CVEGEO, r.CVE_MPIO, r.NOM_MUN, r.NOM_LOC, int(r.pob), int(r.inundable), int(r.aislada)])
    nl = len(it) + 1
    for row in wl.iter_rows(min_row=2):
        for c in row:
            c.font = f_n
        row[4].number_format = "#,##0"
    wl.auto_filter.ref = f"A1:G{nl}"
    LM, LP, LA = f"Localidades!$B$2:$B${nl}", f"Localidades!$E$2:$E${nl}", f"Localidades!$G$2:$G${nl}"

    # ------------------------------------------------------------- Municipios
    wm = wb.create_sheet("Municipios")
    encabezado(wm, ["CVE_MPIO", "Municipio", "Km de red (componente principal)", "Km en zona inundable TR100",
                    "Localidades rurales", "Habitantes rurales", "Localidades rurales aisladas",
                    "Habitantes rurales aislados"], [10, 28, 16, 16, 12, 14, 14, 14])
    for i, r in enumerate(mun.itertuples(), 2):
        wm.append([r.CVE_MPIO, r.municipio, round(r.km, 3), round(r.km_inund, 3),
                   f'=COUNTIF({LM},A{i})', f'=SUMIF({LM},A{i},{LP})',
                   f'=COUNTIFS({LM},A{i},{LA},1)', f'=SUMIFS({LP},{LM},A{i},{LA},1)'])
    nm = len(mun) + 1
    for row in wm.iter_rows(min_row=2):
        for c in row:
            c.font = f_n
        row[2].font = row[3].font = f_in
        row[2].number_format = row[3].number_format = "#,##0.0"
        for c in row[4:]:
            c.number_format = "#,##0"
    wm.cell(nm + 2, 1, "Texto azul (km): calculado por el script 14 con puntos de muestreo cada 15 m sobre la RNC "
            "2025 y el ráster TR100; no es reproducible con fórmulas de Excel. Las columnas E-H son fórmulas "
            "sobre la hoja Localidades.").font = f_i

    # ------------------------------------------------------------- Spearman
    ws = wb.create_sheet("Spearman")
    sel = mun[mun.incluido].reset_index(drop=True)
    encabezado(ws, ["CVE_MPIO", "Municipio", "X: % red inundable", "Y: % población rural aislada",
                    "Rango X", "Rango Y", "d = Rango X − Rango Y", "d²"], [10, 28, 13, 15, 10, 10, 13, 10])
    n = len(sel)
    f1, fn = 2, n + 1
    X, Y = f"$C${f1}:$C${fn}", f"$D${f1}:$D${fn}"
    for k, r in enumerate(sel.itertuples(), f1):
        fila_m = int(mun.index[mun.CVE_MPIO == r.CVE_MPIO][0]) + 2
        ws.cell(k, 1, r.CVE_MPIO)
        ws.cell(k, 2, r.municipio)
        ws.cell(k, 3, f"=Municipios!D{fila_m}/Municipios!C{fila_m}*100").number_format = "0.00"
        ws.cell(k, 4, f"=Municipios!H{fila_m}/Municipios!F{fila_m}*100").number_format = "0.00"
        ws.cell(k, 5, f"=RANK(C{k},{X},1)+(COUNTIF({X},C{k})-1)/2").number_format = "0.0"
        ws.cell(k, 6, f"=RANK(D{k},{Y},1)+(COUNTIF({Y},D{k})-1)/2").number_format = "0.0"
        ws.cell(k, 7, f"=E{k}-F{k}").number_format = "0.0"
        ws.cell(k, 8, f"=G{k}^2").number_format = "0.00"
        for j in range(1, 9):
            ws.cell(k, j).font = f_n
    ws["J1"] = "Correlación de Spearman: % red inundable × % población rural aislada (municipios)"
    ws["J1"].font = f_t
    ws["J2"] = "Todas las cifras son fórmulas. Rango 1 = menor valor; empates = rango promedio."
    ws["J2"].font = f_i
    filas = [
        ("n (municipios)", f"=COUNT({X})", "0", ""),
        ("rho de Spearman", f"=CORREL($E${f1}:$E${fn},$F${f1}:$F${fn})", "0.0000",
         "Pearson entre rangos (válida con empates)"),
        ("rho con 1 − 6Σd²/(n(n²−1))", f"=1-6*SUM($H${f1}:$H${fn})/(L{{n}}*(L{{n}}^2-1))", "0.0000",
         "Solo exacta sin empates; referencia"),
        ("Grados de libertad", "=L{n}-2", "0", "n − 2"),
        ("t", "=L{rho}*SQRT(L{gl}/(1-L{rho}^2))", "0.000", "rho·√((n−2)/(1−rho²))"),
        ("Valor p (bilateral)", "=TDIST(ABS(L{t}),L{gl},2)", "0.0000E+00", "P(|T| > |t|), t de Student"),
        ("α", alfa, "0.00", "Parámetro (--alfa). Dato de entrada."),
        ("Decisión", '=IF(L{p}<L{alfa},"Se rechaza H0: hay asociación monótona","No se rechaza H0")', "@", ""),
        ("Error estándar z (Fieller)", "=1.06/SQRT(L{n}-3)", "0.0000", "1.06/√(n−3)"),
        ("IC 95 % inferior", "=TANH(ATANH(L{rho})-1.96*L{se})", "0.000", "tanh(atanh(rho) − 1.96·EE)"),
        ("IC 95 % superior", "=TANH(ATANH(L{rho})+1.96*L{se})", "0.000", ""),
        ("Fuerza", '=IF(ABS(L{rho})<0.1,"nula",IF(ABS(L{rho})<0.3,"débil",IF(ABS(L{rho})<0.5,"moderada",'
                   'IF(ABS(L{rho})<0.7,"fuerte","muy fuerte"))))', "@", "Guía usual de |rho|"),
        ("Municipios con Y = 0 %", f"=COUNTIF({Y},0)", "0", "Empates en el rango más bajo de Y"),
        ("Mediana X (%)", f"=MEDIAN({X})", "0.00", ""),
        ("Mediana Y (%)", f"=MEDIAN({Y})", "0.00", ""),
    ]
    claves = ["n", "rho", "rhod", "gl", "t", "p", "alfa", "dec", "se", "icl", "icu", "fz", "y0", "mx", "my"]
    pos = {k: 4 + i for i, k in enumerate(claves)}
    for k, (et, val, fm, nota) in zip(claves, filas):
        r = pos[k]
        ws.cell(r, 10, et).font = f_n
        v = val.format(**pos) if isinstance(val, str) else val
        c = ws.cell(r, 12, v)
        c.number_format, c.border = fm, caja
        c.font = f_in if k == "alfa" else f_n
        ws.cell(r, 13, nota).font = f_i
    for k in ("rho", "p", "dec", "fz"):
        ws.cell(pos[k], 12).fill = clave
        ws.cell(pos[k], 10).font = f_b
    ws.column_dimensions["I"].width = 2
    ws.column_dimensions["J"].width = 30
    ws.column_dimensions["K"].width = 2
    ws.column_dimensions["L"].width = 22
    ws.column_dimensions["M"].width = 38

    # ------------------------------------------------------------- Resumen
    wr = wb.create_sheet("Resumen", 0)
    wr["A1"] = "Correlación de Spearman: % de red municipal inundable × % de población rural aislada"
    wr["A1"].font = f_t
    for r, (k, v) in enumerate((("Escenario", info["escenario"]),
                                ("H0", "No hay asociación monótona entre X e Y (rho = 0)."),
                                ("H1", "Hay asociación monótona (rho ≠ 0), bilateral.")), 3):
        wr.cell(r, 1, k).font = f_b
        c = wr.cell(r, 2, v)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        wr.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
    wr.row_dimensions[3].height = 56
    wr["A7"], wr["B7"] = "Resultado", "Valor"
    for c in (wr["A7"], wr["B7"]):
        c.font, c.fill = f_b, hdr
    filas_r = [("Municipios analizados (n)", "n", "0"), ("rho de Spearman", "rho", "0.0000"),
               ("t", "t", "0.000"), ("Grados de libertad", "gl", "0"), ("Valor p (bilateral)", "p", "0.0000E+00"),
               ("Decisión (α = 0.05)", "dec", "@"), ("IC 95 % inferior", "icl", "0.000"),
               ("IC 95 % superior", "icu", "0.000"), ("Fuerza de la asociación", "fz", "@"),
               ("Municipios con Y = 0 %", "y0", "0"), ("Mediana X (%)", "mx", "0.00"), ("Mediana Y (%)", "my", "0.00")]
    for i, (et, k, fm) in enumerate(filas_r, 8):
        wr.cell(i, 1, et).font = f_n
        c = wr.cell(i, 2, f"=Spearman!L{pos[k]}")
        c.font, c.number_format = f_link, fm
    fi = 8 + len(filas_r) + 1
    wr.cell(fi, 1, "Sensibilidad (Python): excluyendo municipios con Y = 0 %").font = f_b
    for j, (et, v, fm) in enumerate((("n", res_s["n"], "0"), ("rho", res_s["rho_spearman"], "0.0000"),
                                     ("Valor p", res_s["p_valor"], "0.0000E+00")), fi + 1):
        wr.cell(j, 1, et).font = f_n
        c = wr.cell(j, 2, v)
        c.font, c.number_format = f_in, fm
    fi2 = fi + 5
    wr.cell(fi2, 1, "Interpretación").font = f_b
    for k, t in enumerate(info["interpretacion"], fi2 + 1):
        c = wr.cell(k, 1, t)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        wr.merge_cells(start_row=k, start_column=1, end_row=k, end_column=4)
        wr.row_dimensions[k].height = 44
    wr.cell(fi2 + len(info["interpretacion"]) + 2, 1,
            "Verde = vínculo a la hoja Spearman (fórmulas). Azul = calculado en Python o dato de entrada.").font = f_i
    wr.column_dimensions["A"].width = 36
    wr.column_dimensions["B"].width = 30
    wr.column_dimensions["C"].width = 18
    wr.column_dimensions["D"].width = 18

    ch = ScatterChart()
    ch.title = "Municipios: % red inundable vs. % población rural aislada"
    ch.style = 13
    ch.x_axis.title = "% de la red vial municipal en zona inundable"
    ch.y_axis.title = "% de población rural aislada"
    s = Series(Reference(ws, min_col=4, min_row=f1, max_row=fn), Reference(ws, min_col=3, min_row=f1, max_row=fn),
               title="Municipios")
    s.marker = Marker(symbol="circle", size=6)
    s.graphicalProperties.line.noFill = True
    ch.series.append(s)
    ch.x_axis.scaling.min = ch.y_axis.scaling.min = 0
    ch.x_axis.scaling.max = ch.y_axis.scaling.max = 100
    ch.x_axis.delete = ch.y_axis.delete = False
    ch.roundedCorners = False
    ch.legend = None
    for t in (ch.title, ch.x_axis.title, ch.y_axis.title):
        t.overlay = False
    ch.width, ch.height = 16, 11
    wr.add_chart(ch, "F3")

    # ------------------------------------------------------------- Python
    wp = wb.create_sheet("Python")
    wp.append(["Estadístico (scipy.stats.spearmanr)", "Todos los municipios", "Excluyendo Y = 0 %"])
    for c in wp[1]:
        c.font, c.fill = f_b, hdr
    for k in res:
        wp.append([k, res[k], res_s[k]])
    for row in wp.iter_rows(min_row=2):
        for c in row:
            c.font = f_n
        for c in row[1:]:
            if isinstance(c.value, float):
                c.number_format = "0.0000E+00" if 0 < abs(c.value) < 1e-3 else "0.0000"
    wp.column_dimensions["A"].width = 34
    wp.column_dimensions["B"].width = 20
    wp.column_dimensions["C"].width = 20
    excl = mun[~mun.incluido]
    if len(excl):
        wp.cell(wp.max_row + 2, 1, "Municipios excluidos (sin población rural): "
                + ", ".join(excl.municipio)).font = f_i

    # ------------------------------------------------------------- Método
    wt = wb.create_sheet("Metodo")
    txt = [
        ("Unidad", "Municipio del Estado de México (CENAPRED IPI 2015)."),
        ("X", "km de red vial (RNC 2025, tramos válidos de la componente principal) en píxel inundable TR100 / km de "
              "red del municipio × 100. Puntos de muestreo cada 15 m; cada punto vale L/n m y se asigna al municipio "
              "donde cae."),
        ("Y", "Habitantes de localidades rurales (< 2,500 hab.) aisladas / habitantes rurales del municipio × 100. "
              "Aislada = sin ruta a hospital público operativo tras cerrar los tramos con > 50 % de su longitud "
              "inundable (mismo escenario de los scripts 09 a 13)."),
        ("Spearman", "rho = CORREL(rangos X, rangos Y), rangos promedio en empates; t = rho·√((n−2)/(1−rho²)); "
                     "p = TDIST(|t|, n−2, 2); IC 95 % por z de Fisher con EE de Fieller 1.06/√(n−3)."),
        ("Advertencia", "Asociación ecológica entre municipios; los municipios vecinos no son independientes "
                        "(autocorrelación espacial) y los de pocas localidades rurales tienen Y muy variable."),
        ("Fuentes", "rnc2025Edomex.gpkg; CENAPRED_IPI_2015_Inund_Munici.zip; tr100_prcent.tif; iter_15csv20.zip; "
                    "ESTABLECIMIENTO_SALUD_202609.xlsx; scripts 02, 03, 04, 05, 07, 08, 11, 12 y 14."),
    ]
    for i, (k, v) in enumerate(txt, 1):
        wt.cell(i, 1, k).font = f_b
        c = wt.cell(i, 2, v)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        wt.cell(i, 1).alignment = Alignment(vertical="top")
        wt.row_dimensions[i].height = 30 if len(v) < 120 else 45
    wt.column_dimensions["A"].width = 16
    wt.column_dimensions["B"].width = 105
    wb.save(ruta)


# ============================================================================
# 4. Programa principal
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUTAS.items():
        ap.add_argument(f"--{k}", type=Path, default=v)
    ap.add_argument("--intervalo", type=float, default=s07.m03.INTERVALO)
    ap.add_argument("--umbral", type=float, default=50.0)
    ap.add_argument("--alfa", type=float, default=0.05)
    a = ap.parse_args()
    for k in RUTAS:
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()
    SALIDAS.mkdir(exist_ok=True)

    print("1/4 Km de red e inundables por municipio (puntos cada 15 m)...")
    muns, _ = s08.municipios(a.cenapred)
    km, km_inund = red_por_municipio(a, muns)
    mun = pd.DataFrame({"CVE_MPIO": [k for k, _, _ in muns],
                        "municipio": [s11.texto_utf8(n) for _, n, _ in muns],
                        "km": km, "km_inund": km_inund})

    print("2/4 Aislamiento de localidades rurales (escenario > 50 %)...")
    it, n_hosp, n_cerr, n_tr = s12.variables_localidades(a)
    it["CVE_MPIO"] = it["ENTIDAD"] + it["MUN"]
    ag = it.groupby("CVE_MPIO").agg(loc_rurales=("pob", "size"), hab_rurales=("pob", "sum"),
                                    loc_aisladas=("aislada", "sum"),
                                    hab_aislados=("pob", lambda s: int(s[it.loc[s.index, "aislada"] == 1].sum())))
    mun = mun.merge(ag, on="CVE_MPIO", how="left").fillna({"loc_rurales": 0, "hab_rurales": 0,
                                                            "loc_aisladas": 0, "hab_aislados": 0})
    mun["pct_red_inundable"] = 100 * mun.km_inund / mun.km
    mun["pct_pob_rural_aislada"] = np.where(mun.hab_rurales > 0, 100 * mun.hab_aislados / mun.hab_rurales, np.nan)
    mun["incluido"] = mun.hab_rurales > 0

    print("3/4 Spearman...")
    sel = mun[mun.incluido]
    res = spearman(sel.pct_red_inundable.to_numpy(), sel.pct_pob_rural_aislada.to_numpy())
    s2 = sel[sel.pct_pob_rural_aislada > 0]
    res_s = spearman(s2.pct_red_inundable.to_numpy(), s2.pct_pob_rural_aislada.to_numpy())

    dec = "se rechaza H0" if res["p_valor"] < a.alfa else "no se rechaza H0"
    signo = "positiva" if res["rho_spearman"] > 0 else "negativa"
    interp = [
        (f"Con {res['n']} municipios, rho = {res['rho_spearman']:.3f} (IC 95 % {res['ic95_inf']:.3f} a "
         f"{res['ic95_sup']:.3f}), t = {res['t']:.2f}, gl = {res['gl']}, p = {res['p_valor']:.2e}: {dec} al nivel "
         f"α = {a.alfa:g}. La asociación es {signo} y {fuerza(res['rho_spearman'])}."),
        (f"Lectura: los municipios con mayor proporción de su red vial en zona inundable tienden a tener mayor "
         f"proporción de su población rural aislada." if res["rho_spearman"] > 0 else
         "Lectura: la proporción de red inundable no se acompaña de mayor aislamiento rural."),
        (f"{int((sel.pct_pob_rural_aislada == 0).sum())} municipios tienen 0 % de población rural aislada. "
         f"Excluyéndolos (sensibilidad), rho = {res_s['rho_spearman']:.3f} con n = {res_s['n']} "
         f"(p = {res_s['p_valor']:.2e})."),
        "Cautela: asociación ecológica entre municipios, con dependencia espacial entre vecinos; no implica que "
        "dentro de cada municipio la población expuesta sea la que se aísla.",
    ]
    excl = mun[~mun.incluido].municipio.tolist()
    info = {"escenario": (f"Unidad: municipio ({len(sel)} con población rural"
                          + (f"; excluidos sin población rural: {', '.join(excl)}" if excl else "") +
                          f"). X con puntos de muestreo cada {a.intervalo:g} m. Y: tramos con > {a.umbral:g} % "
                          f"inundable cerrados ({n_cerr:,d}); {n_hosp} hospitales públicos operativos."),
            "interpretacion": interp}

    print("4/4 Escribiendo Excel...")
    ruta_x = SALIDAS / f"{NOMBRE}.xlsx"
    escribir_excel(ruta_x, mun, it, res, res_s, info, a.alfa)
    mun.round(4).to_csv(SALIDAS / "14_municipios_spearman.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 84)
    print(" SPEARMAN: % de red municipal inundable × % de población rural aislada")
    print("=" * 84)
    print(f"  {info['escenario']}")
    print(f"\n  X (% red inundable):       mediana {sel.pct_red_inundable.median():.1f}, "
          f"rango {sel.pct_red_inundable.min():.1f} – {sel.pct_red_inundable.max():.1f}")
    print(f"  Y (% pob. rural aislada):  mediana {sel.pct_pob_rural_aislada.median():.1f}, "
          f"rango {sel.pct_pob_rural_aislada.min():.1f} – {sel.pct_pob_rural_aislada.max():.1f}; "
          f"municipios con Y = 0: {int((sel.pct_pob_rural_aislada == 0).sum())}")
    print(f"\n  rho de Spearman = {res['rho_spearman']:.4f}   IC 95 % [{res['ic95_inf']:.3f}, {res['ic95_sup']:.3f}]")
    print(f"  t = {res['t']:.3f}   gl = {res['gl']}   p = {res['p_valor']:.3e}  ->  "
          + ("SE RECHAZA H0" if res["p_valor"] < a.alfa else "no se rechaza H0")
          + f"   (asociación {fuerza(res['rho_spearman'])})")
    print(f"  Sensibilidad sin Y = 0: n = {res_s['n']}, rho = {res_s['rho_spearman']:.4f}, p = {res_s['p_valor']:.3e}")
    print("\n  10 municipios con mayor % de población rural aislada:")
    print(f"  {'Municipio':30s}{'% red inund.':>13s}{'% pob. rural ais.':>19s}{'Hab. rurales':>14s}")
    for r in sel.sort_values("pct_pob_rural_aislada", ascending=False).head(10).itertuples():
        print(f"  {r.municipio[:29]:30s}{r.pct_red_inundable:>13.1f}{r.pct_pob_rural_aislada:>19.1f}"
              f"{int(r.hab_rurales):>14,d}")
    print("\n  Interpretación:")
    for t in interp:
        print("   - " + t)
    print(f"\n  Excel: {ruta_x}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
