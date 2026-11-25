# -*- coding: utf-8 -*-
"""
Script 12 - Aislamiento por Inundaciones: prueba Chi-cuadrada
=============================================================

OBJETIVO
--------
Probar si existe asociación entre dos variables dicotómicas de las
localidades rurales (< 2,500 hab.) del Estado de México:

  X = localidad rural en zona inundable TR100        (sí / no)
  Y = localidad rural aislada de hospitales públicos (sí / no)

  H0: X e Y son independientes (estar en zona inundable no cambia la
      probabilidad de quedar aislada).
  H1: X e Y están asociadas.

Además de la prueba se entrega un libro de Excel con la integración de todos
los datos: el registro de cada localidad, la tabla de contingencia y todos
los cálculos con FÓRMULAS de Excel (se recalculan si se edita la hoja de
datos), comparados con el resultado de Python (scipy).

DEFINICIÓN DE LAS VARIABLES (mismo escenario de los scripts 07, 09 y 10)
------------------------------------------------------------------------
  - Red vial: tramos válidos de la componente principal de la RNC 2025
    (capa 20km_red_vial_buffer), puntos de muestreo cada 15 m (script 03).
  - Escenario de inundación: se cierran los tramos con MÁS DEL 50 % de su
    longitud en zona inundable TR100.
  - Hospitales: públicos en operación ('DE HOSPITALIZACIÓN', 'EN OPERACION',
    institución distinta de 'SERVICIOS MEDICOS PRIVADOS') dentro del
    polígono 20km_buffer.
  - Aislada (Y = sí): la localidad no tiene ninguna ruta a un hospital
    público en el escenario (tiempo de viaje infinito).
  - En zona inundable (X = sí): el punto de la localidad (coordenada del
    ITER 2020) cae en un píxel inundable del ráster tr100_prcent.tif
    (valor 0).
  - Unidad de análisis: la LOCALIDAD (4,215 localidades rurales). La
    prueba Chi-cuadrada cuenta casos; no puede ponderarse por población
    (los habitantes no son observaciones independientes). La población se
    incluye en el Excel solo como información descriptiva.

ESTADÍSTICOS CALCULADOS
-----------------------
  Tabla 2x2 de frecuencias observadas O y esperadas E = (fila * col) / N.
  1. Chi-cuadrada de Pearson:     X2 = suma (O - E)^2 / E, gl = 1.
  2. Chi-cuadrada con corrección de continuidad de Yates:
                                   X2 = suma (|O - E| - 0.5)^2 / E.
     (scipy la aplica por defecto en tablas 2x2; se reportan ambas.)
  3. Valor p = P(chi2_1 > X2).
  4. Condición de validez: todas las E >= 5.
  5. Tamaño del efecto: V de Cramér = sqrt(X2 / N) (= |phi| en 2x2).
     Guía de Cohen para gl = 1: 0.1 pequeño, 0.3 mediano, 0.5 grande.
  6. Razón de momios (odds ratio) OR = (a*d) / (b*c) con IC 95 % de Woolf:
     exp( ln OR +/- 1.96 * sqrt(1/a + 1/b + 1/c + 1/d) ).
  7. Riesgo de aislamiento en cada grupo y riesgo relativo.
  8. Prueba exacta de Fisher (bilateral), como verificación sin
     aproximación asintótica.
  9. Residuos estandarizados ajustados por celda:
     (O - E) / sqrt( E * (1 - fila/N) * (1 - col/N) ); |r| > 1.96 indica
     una celda que se aparta de lo esperado al 5 %.

  Nivel de significancia alfa = 0.05 (parámetro --alfa).

ADVERTENCIA METODOLÓGICA
------------------------
  La prueba supone observaciones independientes. Las localidades no lo son
  del todo: las vecinas comparten caminos y suelen quedar aisladas juntas
  (autocorrelación espacial). Con N grande esto infla la significancia; el
  valor p debe leerse con cautela y conviene apoyarse en el tamaño del
  efecto (V de Cramér, OR). Una prueba que corrija la dependencia espacial
  (p. ej., permutaciones por municipio o modelo logístico con efectos por
  municipio) sería un siguiente paso.

FUENTES
-------
  rnc2025Edomex.gpkg, CENAPRED.zip (IPI 2015, límite estatal),
  tr100_prcent.tif, inegi\\iter_15csv20.zip, ESTABLECIMIENTO_SALUD_202609.xlsx
  y los scripts 02, 03, 04, 05 y 07 (se reutilizan sus funciones).

USO
---
  python 12_chi_cuadrada_aislamiento_inundable.py
  python 12_chi_cuadrada_aislamiento_inundable.py --umbral 50 --alfa 0.05

REQUISITOS
----------
  numpy, pandas, networkx, scipy, matplotlib, Pillow, openpyxl.

SALIDAS (carpeta salidas\\)
--------------------------
  12_chi_cuadrada_aislamiento_inundable.xlsx   hojas:
      Resumen        resultados e interpretación
      Contingencia   tabla 2x2 y todos los cálculos con fórmulas de Excel
      Datos          una fila por localidad rural (fuente de la tabla)
      Python         resultados de scipy para comparar
      Metodo         definiciones, criterios y fuentes
  12_resultados_chi_cuadrada.csv
"""

import argparse
import importlib.util
import math
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy import stats
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
NOMBRE = "12_chi_cuadrada_aislamiento_inundable"


# ============================================================================
# 1. Variables por localidad rural
# ============================================================================
def variables_localidades(a):
    R = s05.Raster(a.tr100)
    inund = R.a == s05.VALOR_INUNDABLE
    anillos, _ = s05.anillos_edomex(a.cenapred)
    m_edomex = R.rasterizar(anillos)
    con = sqlite3.connect(f"file:{a.gpkg.as_posix()}?mode=ro", uri=True)
    buffer = s05.anillos_gpkg_poligono(con.execute(f'SELECT geom FROM "{s07.CAPA_BUFFER}"').fetchone()[0])
    tr = s07.tramos_y_porcentaje(con, R, inund, m_edomex, a.intervalo)
    cerrados = (tr["pct_inund"] > a.umbral).to_numpy()

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
    it["pob"] = pd.to_numeric(it["POBTOT"], errors="coerce").fillna(0).astype(int)
    it["lon"] = it["LONGITUD"].map(loc04.dms_a_decimal)
    it["lat"] = it["LATITUD"].map(loc04.dms_a_decimal)
    it = it[it["lon"].notna() & it["lat"].notna() & (it["pob"] < loc04.UMBRAL_RURAL)].reset_index(drop=True)
    it["CVEGEO"] = it["ENTIDAD"] + it["MUN"] + it["LOC"]
    lx, ly = s05.lcc(it["lon"], it["lat"])
    it["inundable"] = R.valores(lx, ly, inund).astype(int)
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
    hn, hc = asignar(*s05.lcc(h["lon"], h["lat"]))

    nl, cl_ = it["nodo"].to_numpy(), it["con_min"].to_numpy()
    t_base = s07.tiempos(s07.grafo(tr, np.ones(len(tr), bool)), hn, hc, nl, cl_)
    t_esc = s07.tiempos(s07.grafo(tr, ~cerrados), hn, hc, nl, cl_)
    it["min_base"] = np.round(t_base, 2)
    it["min_escenario"] = np.where(np.isfinite(t_esc), np.round(t_esc, 2), np.nan)
    it["aislada"] = (~np.isfinite(t_esc)).astype(int)
    return it, len(h), int(cerrados.sum()), len(tr)


# ============================================================================
# 2. Estadísticos en Python
# ============================================================================
def estadisticos(it, alfa):
    # Tabla: filas = inundable (Sí, No); columnas = aislada (Sí, No)
    O = np.array([[((it.inundable == 1) & (it.aislada == 1)).sum(), ((it.inundable == 1) & (it.aislada == 0)).sum()],
                  [((it.inundable == 0) & (it.aislada == 1)).sum(), ((it.inundable == 0) & (it.aislada == 0)).sum()]])
    N = O.sum()
    x2, p, gl, E = stats.chi2_contingency(O, correction=False)
    x2y, py, _, _ = stats.chi2_contingency(O, correction=True)
    _, p_fisher = stats.fisher_exact(O, alternative="two-sided")
    a_, b_, c_, d_ = O.ravel().astype(float)
    OR = (a_ * d_) / (b_ * c_)
    se = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    r1, r0 = a_ / (a_ + b_), c_ / (c_ + d_)
    fil, col = O.sum(1, keepdims=True), O.sum(0, keepdims=True)
    radj = (O - E) / np.sqrt(E * (1 - fil / N) * (1 - col / N))
    res = {
        "N": int(N), "a_inund_aislada": int(O[0, 0]), "b_inund_no_aislada": int(O[0, 1]),
        "c_seca_aislada": int(O[1, 0]), "d_seca_no_aislada": int(O[1, 1]),
        "chi2_pearson": x2, "gl": int(gl), "p_pearson": p,
        "chi2_yates": x2y, "p_yates": py, "p_fisher_exacta": p_fisher,
        "esperada_minima": float(E.min()), "v_cramer": math.sqrt(x2 / N),
        "phi": (a_ * d_ - b_ * c_) / math.sqrt((a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_)),
        "odds_ratio": OR, "or_ic95_inf": math.exp(math.log(OR) - 1.96 * se),
        "or_ic95_sup": math.exp(math.log(OR) + 1.96 * se),
        "riesgo_aislamiento_inundable": r1, "riesgo_aislamiento_no_inundable": r0,
        "riesgo_relativo": r1 / r0, "alfa": alfa,
        "chi2_critico": stats.chi2.ppf(1 - alfa, 1),
        "rechaza_H0": bool(p < alfa),
    }
    return res, O, E, radj


def efecto(v):
    return "despreciable" if v < 0.1 else "pequeño" if v < 0.3 else "mediano" if v < 0.5 else "grande"


# ============================================================================
# 3. Libro de Excel
# ============================================================================
def escribir_excel(ruta, it, res, info):
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    F = "Arial"
    f_n, f_b = Font(name=F, size=10), Font(name=F, size=10, bold=True)
    f_t = Font(name=F, size=13, bold=True)
    f_link = Font(name=F, size=10, color="008000")
    f_in = Font(name=F, size=10, color="0000FF")
    hdr = PatternFill("solid", start_color="D9E1F2")
    clave = PatternFill("solid", start_color="FFF2CC")
    fino = Side(style="thin", color="999999")
    caja = Border(left=fino, right=fino, top=fino, bottom=fino)
    wb = Workbook()

    # ------------------------------------------------------------- Datos
    wd = wb.active
    wd.title = "Datos"
    cols = ["CVEGEO", "Municipio", "Localidad", "Longitud", "Latitud", "Población 2020",
            "Inundable (1=sí)", "Aislada (1=sí)", "Min. red completa", "Min. escenario"]
    wd.append(cols)
    for r in it.itertuples():
        wd.append([r.CVEGEO, r.NOM_MUN, r.NOM_LOC, round(r.lon, 6), round(r.lat, 6), int(r.pob),
                   int(r.inundable), int(r.aislada), float(r.min_base),
                   None if pd.isna(r.min_escenario) else float(r.min_escenario)])
    n = len(it) + 1
    for c in wd[1]:
        c.font, c.fill, c.alignment = f_b, hdr, Alignment(wrap_text=True, vertical="center")
    for row in wd.iter_rows(min_row=2, max_row=n):
        for c in row:
            c.font = f_n
        row[3].number_format = row[4].number_format = "0.000000"
        row[5].number_format = "#,##0"
        row[8].number_format = row[9].number_format = "0.00"
    for i, w in enumerate([14, 26, 34, 12, 11, 12, 11, 11, 13, 13], 1):
        wd.column_dimensions[get_column_letter(i)].width = w
    wd.freeze_panes = "A2"
    wd.auto_filter.ref = f"A1:J{n}"
    wd["J1"].comment = Comment("Vacío = sin ruta a hospital público en el escenario (aislada).", "Script 12")
    rI, rA, rP = f"Datos!$G$2:$G${n}", f"Datos!$H$2:$H${n}", f"Datos!$F$2:$F${n}"

    # ------------------------------------------------------------- Contingencia
    wc = wb.create_sheet("Contingencia")
    wc["A1"] = "Tabla de contingencia: localidades rurales en zona inundable × aisladas de hospitales públicos"
    wc["A1"].font = f_t
    wc["A2"] = "Todas las cifras de esta hoja son fórmulas sobre la hoja Datos (se recalculan si se editan los datos)."
    wc["A2"].font = Font(name=F, size=9, italic=True)

    def tabla(fila0, titulo):
        wc.cell(fila0, 1, titulo).font = f_b
        for j, t in enumerate(["", "Aislada: Sí", "Aislada: No", "Total"]):
            c = wc.cell(fila0 + 1, 1 + j, t)
            c.font, c.fill, c.border = f_b, hdr, caja
        for i, t in enumerate(["Inundable: Sí", "Inundable: No", "Total"]):
            c = wc.cell(fila0 + 2 + i, 1, t)
            c.font, c.fill, c.border = f_b, hdr, caja
        return fila0 + 2

    # Observadas (filas 5-7, columnas B-D)
    f = tabla(4, "1. Frecuencias observadas (O)")
    obs = {(0, 0): (1, 1), (0, 1): (1, 0), (1, 0): (0, 1), (1, 1): (0, 0)}
    for (i, j), (vi, va) in obs.items():
        wc.cell(f + i, 2 + j, f"=COUNTIFS({rI},{vi},{rA},{va})")
    for i in range(2):
        wc.cell(f + i, 4, f"=SUM(B{f + i}:C{f + i})")
    for j, L in enumerate("BCD"):
        wc.cell(f + 2, 2 + j, f"=SUM({L}{f}:{L}{f + 1})")
    O = {(i, j): f"{'BC'[j]}{f + i}" for i in range(2) for j in range(2)}
    FT = {0: f"$D${f}", 1: f"$D${f + 1}"}
    CT = {0: f"B${f + 2}", 1: f"C${f + 2}"}
    NN = f"$D${f + 2}"

    # Esperadas (filas 11-13)
    fe = tabla(f + 4, "2. Frecuencias esperadas bajo H0: E = (total fila × total columna) / N")
    for i in range(2):
        for j in range(2):
            wc.cell(fe + i, 2 + j, f"={FT[i]}*{CT[j]}/{NN}").number_format = "#,##0.00"
        wc.cell(fe + i, 4, f"=SUM(B{fe + i}:C{fe + i})").number_format = "#,##0.00"
    for j, L in enumerate("BCD"):
        wc.cell(fe + 2, 2 + j, f"=SUM({L}{fe}:{L}{fe + 1})").number_format = "#,##0.00"
    E = {(i, j): f"{'BC'[j]}{fe + i}" for i in range(2) for j in range(2)}

    # Contribuciones (O-E)^2/E
    fc = tabla(fe + 4, "3. Contribución de cada celda a la Chi-cuadrada de Pearson: (O − E)² / E")
    for i in range(2):
        for j in range(2):
            wc.cell(fc + i, 2 + j, f"=({O[i, j]}-{E[i, j]})^2/{E[i, j]}").number_format = "0.000"
        wc.cell(fc + i, 4, f"=SUM(B{fc + i}:C{fc + i})").number_format = "0.000"
    for j, L in enumerate("BCD"):
        wc.cell(fc + 2, 2 + j, f"=SUM({L}{fc}:{L}{fc + 1})").number_format = "0.000"

    # Contribuciones Yates
    fy = tabla(fc + 4, "4. Contribución con corrección de Yates: (|O − E| − 0.5)² / E")
    for i in range(2):
        for j in range(2):
            wc.cell(fy + i, 2 + j, f"=(ABS({O[i, j]}-{E[i, j]})-0.5)^2/{E[i, j]}").number_format = "0.000"
        wc.cell(fy + i, 4, f"=SUM(B{fy + i}:C{fy + i})").number_format = "0.000"
    for j, L in enumerate("BCD"):
        wc.cell(fy + 2, 2 + j, f"=SUM({L}{fy}:{L}{fy + 1})").number_format = "0.000"

    # Residuos ajustados
    fr = tabla(fy + 4, "5. Residuos estandarizados ajustados: (O − E) / √(E·(1 − fila/N)·(1 − col/N))")
    for i in range(2):
        for j in range(2):
            wc.cell(fr + i, 2 + j,
                    f"=({O[i, j]}-{E[i, j]})/SQRT({E[i, j]}*(1-{FT[i]}/{NN})*(1-{CT[j]}/{NN}))").number_format = "0.00"
    wc.cell(fr + 2, 1, "|r| > 1.96: la celda se aparta de lo esperado (α = 0.05)").font = Font(name=F, size=9, italic=True)
    wc.cell(fr + 2, 1).fill = PatternFill(fill_type=None)
    for j in range(1, 4):
        wc.cell(fr + 2, 1 + j).border = Border()

    # Estadísticos
    fs = fr + 5
    wc.cell(fs, 1, "6. Estadísticos de la prueba").font = f_b
    filas = [
        ("Nivel de significancia α", 0.05, "0.00", "Parámetro (--alfa). Dato de entrada."),
        ("N (localidades rurales)", f"={NN}", "#,##0", ""),
        ("Grados de libertad", "=(2-1)*(2-1)", "0", "(filas − 1)·(columnas − 1)"),
        ("Chi-cuadrada de Pearson", f"=D{fc + 2}", "0.000", "Σ (O − E)² / E"),
        ("Valor p (Pearson)", "=CHIDIST(B{x2},B{gl})", "0.0000E+00", "P(χ²₁ > X²)"),
        ("Chi-cuadrada con corrección de Yates", f"=D{fy + 2}", "0.000", "Σ (|O − E| − 0.5)² / E"),
        ("Valor p (Yates)", "=CHIDIST(B{x2y},B{gl})", "0.0000E+00", ""),
        ("Valor crítico χ²(1, α)", "=CHIINV(B{alfa},B{gl})", "0.000", "Se rechaza H0 si X² > valor crítico"),
        ("Decisión (Pearson)", '=IF(B{p}<B{alfa},"Se rechaza H0: hay asociación","No se rechaza H0")', "@", ""),
        ("Frecuencia esperada mínima", f"=MIN(B{fe}:C{fe + 1})", "0.00", "Validez: todas las E ≥ 5"),
        ("¿Se cumple E ≥ 5?", '=IF(B{emin}>=5,"Sí","No")', "@", ""),
        ("V de Cramér", "=SQRT(B{x2}/B{N})", "0.000", "Guía de Cohen (gl = 1): 0.1 pequeño, 0.3 mediano, 0.5 grande"),
        ("Phi (con signo)", f"=({O[0, 0]}*{O[1, 1]}-{O[0, 1]}*{O[1, 0]})/SQRT({FT[0]}*{FT[1]}*{CT[0]}*{CT[1]})",
         "0.000", "Positivo: inundable se asocia con aislada"),
        ("Razón de momios (OR)", f"=({O[0, 0]}*{O[1, 1]})/({O[0, 1]}*{O[1, 0]})", "0.000",
         "Momios de aislamiento inundable / no inundable"),
        ("Error estándar de ln(OR)", f"=SQRT(1/{O[0, 0]}+1/{O[0, 1]}+1/{O[1, 0]}+1/{O[1, 1]})", "0.0000", "Woolf"),
        ("OR: límite inferior IC 95 %", "=EXP(LN(B{OR})-1.96*B{se})", "0.000", ""),
        ("OR: límite superior IC 95 %", "=EXP(LN(B{OR})+1.96*B{se})", "0.000", ""),
        ("Riesgo de aislamiento: inundables", f"={O[0, 0]}/{FT[0]}", "0.0%", ""),
        ("Riesgo de aislamiento: no inundables", f"={O[1, 0]}/{FT[1]}", "0.0%", ""),
        ("Riesgo relativo", "=B{r1}/B{r0}", "0.000", ""),
        ("Valor p exacto de Fisher (bilateral)", res["p_fisher_exacta"], "0.0000E+00",
         "Calculado en Python (scipy.stats.fisher_exact); Excel no tiene la prueba de Fisher."),
    ]
    pos = {}
    claves = ["alfa", "N", "gl", "x2", "p", "x2y", "py", "crit", "dec", "emin", "eok", "V", "phi", "OR", "se",
              "orl", "oru", "r1", "r0", "rr", "fisher"]
    for k, (et, *_r) in zip(claves, filas):
        pos[k] = fs + 1 + claves.index(k)
    for k, (et, val, fmt, nota) in zip(claves, filas):
        r = pos[k]
        wc.cell(r, 1, et).font = f_n
        v = val.format(**pos) if isinstance(val, str) else val
        c = wc.cell(r, 2, v)
        c.number_format, c.border = fmt, caja
        c.font = f_in if k in ("alfa", "fisher") else f_n
        wc.cell(r, 3, nota).font = Font(name=F, size=9, italic=True, color="555555")
    for k in ("x2", "p", "dec", "V", "OR"):
        wc.cell(pos[k], 2).fill = clave
        wc.cell(pos[k], 1).font = f_b
    wc.cell(pos["fisher"] + 2, 1,
            "Texto azul = dato de entrada (α) o valor calculado fuera de Excel (Fisher). "
            "Fondo amarillo = resultados principales.").font = Font(name=F, size=9, italic=True)

    # Población descriptiva
    fp = pos["fisher"] + 4
    wc.cell(fp, 1, "7. Población (solo descriptiva; la prueba no se pondera por habitantes)").font = f_b
    for j, t in enumerate(["", "Aislada: Sí", "Aislada: No", "Total"]):
        c = wc.cell(fp + 1, 1 + j, t)
        c.font, c.fill, c.border = f_b, hdr, caja
    for i, (t, vi) in enumerate((("Inundable: Sí", 1), ("Inundable: No", 0))):
        wc.cell(fp + 2 + i, 1, t).font = f_b
        for j, va in enumerate((1, 0)):
            wc.cell(fp + 2 + i, 2 + j, f"=SUMIFS({rP},{rI},{vi},{rA},{va})").number_format = "#,##0"
        wc.cell(fp + 2 + i, 4, f"=SUM(B{fp + 2 + i}:C{fp + 2 + i})").number_format = "#,##0"
    wc.cell(fp + 4, 1, "Total").font = f_b
    for j, L in enumerate("BCD"):
        wc.cell(fp + 4, 2 + j, f"=SUM({L}{fp + 2}:{L}{fp + 3})").number_format = "#,##0"

    for row in wc.iter_rows(min_row=4, max_row=fp + 4, max_col=4):
        for c in row:
            if c.font.name != F:
                c.font = f_n
            if c.column > 1 and c.row not in pos.values() and c.number_format == "General" and c.data_type == "f":
                c.number_format = "#,##0"
    wc.column_dimensions["A"].width = 40
    for L in "BCD":
        wc.column_dimensions[L].width = 16
    wc.column_dimensions["C"].width = 18

    # ------------------------------------------------------------- Resumen
    ws = wb.create_sheet("Resumen", 0)
    ws["A1"] = "Prueba Chi-cuadrada: aislamiento hospitalario × zona inundable (localidades rurales, Edomex)"
    ws["A1"].font = f_t
    ws["A3"] = "Escenario"
    ws["B3"] = (f"Tramos con > {info['umbral']:g} % de su longitud inundable (TR100) cerrados "
                f"({info['cerrados']:,d} de {info['tramos']:,d}); destino: {info['hosp']} hospitales públicos "
                "en operación (franja de 20 km); puntos de muestreo cada 15 m.")
    ws["A4"] = "H0"
    ws["B4"] = "Estar en zona inundable y quedar aislada son independientes."
    ws["A5"] = "H1"
    ws["B5"] = "Hay asociación entre estar en zona inundable y quedar aislada."
    filas_r = [
        ("Localidades rurales (N)", "N", "#,##0"),
        ("Chi-cuadrada de Pearson", "x2", "0.000"),
        ("Grados de libertad", "gl", "0"),
        ("Valor p (Pearson)", "p", "0.0000E+00"),
        ("Chi-cuadrada con Yates", "x2y", "0.000"),
        ("Valor p (Yates)", "py", "0.0000E+00"),
        ("Valor p exacto de Fisher", "fisher", "0.0000E+00"),
        ("Decisión (α = 0.05)", "dec", "@"),
        ("V de Cramér", "V", "0.000"),
        ("Razón de momios (OR)", "OR", "0.000"),
        ("OR IC 95 % inferior", "orl", "0.000"),
        ("OR IC 95 % superior", "oru", "0.000"),
        ("Riesgo de aislamiento: inundables", "r1", "0.0%"),
        ("Riesgo de aislamiento: no inundables", "r0", "0.0%"),
        ("Riesgo relativo", "rr", "0.000"),
        ("Frecuencia esperada mínima", "emin", "0.00"),
    ]
    ws["A7"], ws["B7"] = "Resultado", "Valor"
    for c in (ws["A7"], ws["B7"]):
        c.font, c.fill = f_b, hdr
    for i, (et, k, fmt) in enumerate(filas_r, 8):
        ws.cell(i, 1, et).font = f_n
        c = ws.cell(i, 2, f"=Contingencia!B{pos[k]}")
        c.font, c.number_format = f_link, fmt
    fi = 8 + len(filas_r) + 1
    ws.cell(fi, 1, "Interpretación").font = f_b
    for k, t in enumerate(info["interpretacion"]):
        c = ws.cell(fi + 1 + k, 1, t)
        c.font = f_n
        ws.merge_cells(start_row=fi + 1 + k, start_column=1, end_row=fi + 1 + k, end_column=4)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[fi + 1 + k].height = 42
    for r in range(3, 6):
        ws.cell(r, 1).font = f_b
        ws.cell(r, 2).font = f_n
        ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
    ws.row_dimensions[3].height = 42
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 34
    ws.column_dimensions["C"].width = 20
    ws.column_dimensions["D"].width = 20
    ws.cell(fi + 1 + len(info["interpretacion"]) + 1, 1,
            "Texto verde = vínculo a la hoja Contingencia, donde están las fórmulas.").font = \
        Font(name=F, size=9, italic=True)

    # ------------------------------------------------------------- Python
    wp = wb.create_sheet("Python")
    wp.append(["Estadístico (scipy 1.x)", "Valor"])
    for c in wp[1]:
        c.font, c.fill = f_b, hdr
    for k, v in res.items():
        wp.append([k, v if not isinstance(v, (np.floating,)) else float(v)])
    for row in wp.iter_rows(min_row=2):
        for c in row:
            c.font = f_n
        if isinstance(row[1].value, float):
            row[1].number_format = "0.0000E+00" if abs(row[1].value) < 1e-3 and row[1].value != 0 else "0.000000"
    wp.column_dimensions["A"].width = 34
    wp.column_dimensions["B"].width = 18
    wp.cell(wp.max_row + 2, 1, "Valores calculados por el script 12 con scipy.stats (chi2_contingency, "
            "fisher_exact); deben coincidir con la hoja Contingencia.").font = Font(name=F, size=9, italic=True)

    # ------------------------------------------------------------- Método
    wm = wb.create_sheet("Metodo")
    texto = [
        ("Definiciones", ""),
        ("Unidad de análisis", "Localidad rural (< 2,500 hab., ITER 2020), excluidos los renglones agregados 0000/9998/9999."),
        ("Inundable", "El punto de la localidad cae en píxel con valor 0 del ráster tr100_prcent.tif (CENAPRED, TR100, 30 m)."),
        ("Aislada", "Sin ninguna ruta por la red vial a un hospital público en operación tras cerrar los tramos "
                    f"con > {info['umbral']:g} % de su longitud inundable (puntos de muestreo cada 15 m)."),
        ("Red vial", "RNC 2025 (INEGI), capa 20km_red_vial_buffer; sin tramos Deshabilitados, En construcción - "
                     "cerrado ni Cerrada en ambos sentidos; componente conexa principal."),
        ("Hospitales", "CLUES sep-2026: 'DE HOSPITALIZACIÓN', 'EN OPERACION', institución distinta de "
                       "'SERVICIOS MEDICOS PRIVADOS', dentro del polígono 20km_buffer."),
        ("Pruebas", ""),
        ("Chi-cuadrada de Pearson", "X² = Σ (O − E)² / E; gl = 1; p = CHIDIST(X², 1)."),
        ("Corrección de Yates", "X² = Σ (|O − E| − 0.5)² / E; habitual en tablas 2×2."),
        ("Fisher exacta", "Bilateral, scipy.stats.fisher_exact; no depende de aproximación asintótica."),
        ("V de Cramér", "√(X²/N); en 2×2 equivale a |phi|."),
        ("Razón de momios", "(a·d)/(b·c); IC 95 % de Woolf con ln(OR) ± 1.96·√(1/a+1/b+1/c+1/d)."),
        ("Advertencia", "La prueba supone independencia entre observaciones. Las localidades vecinas comparten "
                        "caminos y tienden a aislarse juntas (autocorrelación espacial), lo que infla la "
                        "significancia; leer el valor p con cautela y apoyarse en V de Cramér y OR."),
        ("Fuentes", "rnc2025Edomex.gpkg; CENAPRED_IPI_2015_Inund_Munici.zip; tr100_prcent.tif; "
                    "iter_15csv20.zip; ESTABLECIMIENTO_SALUD_202609.xlsx; scripts 02, 03, 04, 05, 07 y 12."),
    ]
    for i, (k, v) in enumerate(texto, 1):
        a = wm.cell(i, 1, k)
        b = wm.cell(i, 2, v)
        a.font = f_b if v == "" else Font(name=F, size=10, bold=True, color="333333")
        b.font = f_n
        b.alignment = Alignment(wrap_text=True, vertical="top")
        a.alignment = Alignment(vertical="top")
        if v:
            wm.row_dimensions[i].height = 30 if len(v) < 130 else 45
    wm.column_dimensions["A"].width = 24
    wm.column_dimensions["B"].width = 100

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

    print("1/3 Calculando aislamiento y zona inundable de cada localidad rural...")
    it, n_hosp, n_cerr, n_tr = variables_localidades(a)
    print("2/3 Prueba Chi-cuadrada...")
    res, O, E, radj = estadisticos(it, a.alfa)

    interp = [
        (f"Con χ² = {res['chi2_pearson']:,.2f} (gl = 1) y p = {res['p_pearson']:.2e}, "
         + ("se rechaza H0" if res["rechaza_H0"] else "no se rechaza H0")
         + f" al nivel α = {a.alfa:g}: "
         + ("estar en zona inundable y quedar aislada están asociadas." if res["rechaza_H0"]
            else "no hay evidencia de asociación.")),
        (f"La V de Cramér es {res['v_cramer']:.3f} (efecto {efecto(res['v_cramer'])}). "
         f"Las localidades rurales en zona inundable quedan aisladas en el {100 * res['riesgo_aislamiento_inundable']:.1f} % "
         f"de los casos, frente al {100 * res['riesgo_aislamiento_no_inundable']:.1f} % de las no inundables "
         f"(riesgo relativo {res['riesgo_relativo']:.2f}; OR {res['odds_ratio']:.2f}, IC 95 % "
         f"{res['or_ic95_inf']:.2f}–{res['or_ic95_sup']:.2f})."),
        (f"De las {res['a_inund_aislada'] + res['c_seca_aislada']:,d} localidades rurales aisladas, "
         f"{res['c_seca_aislada']:,d} ({100 * res['c_seca_aislada'] / (res['a_inund_aislada'] + res['c_seca_aislada']):.1f} %) "
         "NO están en zona inundable: quedan aisladas porque se cortan los caminos a su alrededor (efecto indirecto)."),
        ("Cautela: las localidades vecinas no son observaciones independientes (comparten caminos), lo que tiende "
         "a exagerar la significancia. El tamaño del efecto es más informativo que el valor p."),
    ]
    info = {"umbral": a.umbral, "cerrados": n_cerr, "tramos": n_tr, "hosp": n_hosp, "interpretacion": interp}

    print("3/3 Escribiendo Excel...")
    ruta_x = SALIDAS / f"{NOMBRE}.xlsx"
    escribir_excel(ruta_x, it, res, info)
    pd.DataFrame([res]).T.rename(columns={0: "valor"}).to_csv(
        SALIDAS / "12_resultados_chi_cuadrada.csv", encoding="utf-8-sig")

    # ---------------------------------------------------------------- Impresión
    print("\n" + "=" * 78)
    print(" PRUEBA CHI-CUADRADA: localidades rurales aisladas × en zona inundable")
    print("=" * 78)
    print(f"  Escenario: tramos > {a.umbral:g} % inundables cerrados ({n_cerr:,d}); {n_hosp} hospitales públicos")
    print(f"\n  Frecuencias observadas (esperadas)      Aislada: Sí        Aislada: No        Total")
    for i, et in enumerate(("Inundable: Sí", "Inundable: No")):
        print(f"  {et:38s}{O[i, 0]:>6,d} ({E[i, 0]:>7.1f})  {O[i, 1]:>6,d} ({E[i, 1]:>7.1f})  {O[i].sum():>7,d}")
    print(f"  {'Total':38s}{O[:, 0].sum():>6,d}{'':11s}{O[:, 1].sum():>6,d}{'':11s}{O.sum():>7,d}")
    print(f"\n  Residuos ajustados: inund-aislada {radj[0, 0]:+.2f}, inund-no {radj[0, 1]:+.2f}, "
          f"seca-aislada {radj[1, 0]:+.2f}, seca-no {radj[1, 1]:+.2f}")
    print(f"\n  Chi-cuadrada de Pearson : {res['chi2_pearson']:>10.3f}  gl = 1   p = {res['p_pearson']:.3e}")
    print(f"  Chi-cuadrada con Yates  : {res['chi2_yates']:>10.3f}           p = {res['p_yates']:.3e}")
    print(f"  Fisher exacta (bilateral):                     p = {res['p_fisher_exacta']:.3e}")
    print(f"  Valor crítico χ²(1, {a.alfa:g}) : {res['chi2_critico']:.3f}  ->  "
          + ("SE RECHAZA H0 (hay asociación)" if res["rechaza_H0"] else "no se rechaza H0"))
    print(f"  Frecuencia esperada mínima: {res['esperada_minima']:.1f} (validez: >= 5)")
    print(f"  V de Cramér: {res['v_cramer']:.3f} ({efecto(res['v_cramer'])}); phi = {res['phi']:+.3f}")
    print(f"  Razón de momios: {res['odds_ratio']:.3f}  IC 95 % [{res['or_ic95_inf']:.3f}, {res['or_ic95_sup']:.3f}]")
    print(f"  Riesgo de aislamiento: inundables {100 * res['riesgo_aislamiento_inundable']:.1f} %, "
          f"no inundables {100 * res['riesgo_aislamiento_no_inundable']:.1f} %; RR = {res['riesgo_relativo']:.3f}")
    print("\n  Interpretación:")
    for t in interp:
        print("   - " + t)
    print(f"\n  Excel: {ruta_x}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
