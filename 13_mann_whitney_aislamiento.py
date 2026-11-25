# -*- coding: utf-8 -*-
"""
Script 13 - Aislamiento por Inundaciones: pruebas U de Mann-Whitney
===================================================================

OBJETIVO
--------
Comparar dos grupos de localidades RURALES (< 2,500 hab.) del Estado de
México: las que quedan AISLADAS de los hospitales públicos en el escenario de
inundación y las que NO quedan aisladas, en dos variables:

  Prueba 1  Tiempo base (red completa, sin inundación) al hospital público
            operativo más cercano (min).
            Pregunta: ¿las localidades que se aíslan ya estaban más lejos (o
            más cerca) del hospital antes de la inundación?
  Prueba 2  Población de la localidad (habitantes, Censo 2020).
            Pregunta: ¿las localidades que se aíslan son más pequeñas o más
            grandes que las que no?

  H0: las distribuciones de la variable son iguales en ambos grupos
      (P(X_aislada > X_no aislada) = 0.5).
  H1: difieren (prueba bilateral).

Se usa Mann-Whitney y no la t de Student porque ambas variables son muy
asimétricas (muchas localidades pequeñas y pocas grandes; tiempos con cola
larga), y la prueba no supone normalidad.

ESCENARIO Y GRUPOS (mismos de los scripts 09, 10 y 12)
------------------------------------------------------
  - Puntos de muestreo cada 15 m (script 03); se cierran los tramos con MÁS
    DEL 50 % de su longitud en zona inundable TR100.
  - Hospitales públicos en operación ('DE HOSPITALIZACIÓN', 'EN OPERACION',
    institución distinta de 'SERVICIOS MEDICOS PRIVADOS') en la franja de
    20 km.
  - Aislada = sin ruta a ningún hospital público en el escenario.
  - Tiempo base = tiempo en la red completa (longitud / VELOCIDAD de la RNC,
    conexión a la red a 5 km/h). Todas las localidades rurales tienen ruta
    en la red completa, así que la variable está definida para ambos grupos.
  Las variables se obtienen con la función del script 12, de modo que los
  grupos son idénticos a los de la prueba Chi-cuadrada.

CÁLCULO (n1 = aisladas, n2 = no aisladas, N = n1 + n2)
-----------------------------------------------------
  1. Rangos de todas las observaciones juntas, de menor a mayor; los empates
     reciben el rango promedio.
  2. R1 = suma de rangos de las aisladas.
     U1 = R1 − n1(n1+1)/2 (estadístico de las aisladas, el que reporta scipy);
     U2 = n1·n2 − U1.
  3. Aproximación normal (adecuada con n1, n2 > 20) con corrección por
     empates y de continuidad:
       μ_U = n1·n2 / 2
       σ_U = √( n1·n2/12 · [ (N+1) − Σ(t³ − t) / (N(N−1)) ] ),
             t = tamaño de cada grupo de empates
       z   = ( max(U1, U2) − μ_U − 0.5 ) / σ_U
       p   = 2 · (1 − Φ(z))   (bilateral)
     Es exactamente lo que hace scipy.stats.mannwhitneyu(method='asymptotic',
     use_continuity=True, alternative='two-sided').
  4. Tamaño del efecto:
       - r = z / √N (sin corrección de continuidad): 0.1 pequeño,
         0.3 mediano, 0.5 grande.
       - Correlación biserial por rangos: r_rb = 2·U1/(n1·n2) − 1
         (de −1 a 1; positivo = aisladas con valores mayores).
       - Probabilidad de superioridad: PS = U1/(n1·n2) = probabilidad de que
         una aislada elegida al azar tenga un valor mayor que una no aislada
         (los empates cuentan 1/2).
  5. Descriptivos por grupo: n, mediana, cuartiles, media, mínimo, máximo.

ADVERTENCIA METODOLÓGICA
------------------------
  Como en el script 12, la prueba supone observaciones independientes; las
  localidades vecinas comparten caminos (autocorrelación espacial), lo que
  infla la significancia. Conviene apoyarse en el tamaño del efecto.

USO
---
  python 13_mann_whitney_aislamiento.py
  python 13_mann_whitney_aislamiento.py --umbral 50 --alfa 0.05

REQUISITOS
----------
  numpy, pandas, networkx, scipy, matplotlib, Pillow, openpyxl.
  En la misma carpeta: scripts 02, 03, 04, 05, 07 y 12.

SALIDAS (carpeta salidas\\)
--------------------------
  13_mann_whitney_aislamiento.xlsx   hojas:
      Resumen      resultados e interpretación de las dos pruebas
      MW_Tiempo    datos ordenados por grupo, rangos y cálculo completo con
                   fórmulas de Excel (prueba 1)
      MW_Poblacion idem para la población (prueba 2)
      Datos        una fila por localidad rural con todas las variables
      Python       resultados de scipy para comparar
      Metodo       definiciones, fórmulas y fuentes
  13_resultados_mann_whitney.csv
"""

import argparse
import importlib.util
import math
import time
from pathlib import Path

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


s12 = _importar("s12", "12_chi_cuadrada_aislamiento_inundable.py")   # variables por localidad
RUTAS = dict(s12.RUTAS)
NOMBRE = "13_mann_whitney_aislamiento"

PRUEBAS = [
    # clave, columna, hoja, etiqueta, unidad, formato
    ("tiempo", "min_base", "MW_Tiempo",
     "Tiempo base al hospital público operativo más cercano", "min", "0.00"),
    ("poblacion", "pob", "MW_Poblacion", "Población de la localidad rural", "hab.", "#,##0"),
]


# ============================================================================
# 1. Estadísticos en Python
# ============================================================================
def mann_whitney(x1, x2):
    """x1 = aisladas, x2 = no aisladas."""
    n1, n2 = len(x1), len(x2)
    N = n1 + n2
    r = stats.mannwhitneyu(x1, x2, alternative="two-sided", method="asymptotic", use_continuity=True)
    U1 = float(r.statistic)
    U2 = n1 * n2 - U1
    _, t = np.unique(np.r_[x1, x2], return_counts=True)
    empates = float((t ** 3 - t).sum())
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 / 12 * ((N + 1) - empates / (N * (N - 1))))
    z_cc = (max(U1, U2) - mu - 0.5) / sigma
    z = (U1 - mu) / sigma
    q = lambda v, p: float(np.percentile(v, p))
    return {
        "n_aisladas": n1, "n_no_aisladas": n2, "N": N,
        "U1_aisladas": U1, "U2_no_aisladas": U2,
        "suma_rangos_aisladas": U1 + n1 * (n1 + 1) / 2,
        "media_U": mu, "sigma_U": sigma, "suma_t3_menos_t": empates,
        "z_con_continuidad": z_cc, "z_sin_continuidad_con_signo": z, "p_valor": float(r.pvalue),
        "r_efecto": abs(z) / math.sqrt(N),
        "correlacion_biserial_rangos": 2 * U1 / (n1 * n2) - 1,
        "prob_superioridad": U1 / (n1 * n2),
        "mediana_aisladas": float(np.median(x1)), "mediana_no_aisladas": float(np.median(x2)),
        "q1_aisladas": q(x1, 25), "q3_aisladas": q(x1, 75),
        "q1_no_aisladas": q(x2, 25), "q3_no_aisladas": q(x2, 75),
        "media_aisladas": float(np.mean(x1)), "media_no_aisladas": float(np.mean(x2)),
        "min_aisladas": float(np.min(x1)), "max_aisladas": float(np.max(x1)),
        "min_no_aisladas": float(np.min(x2)), "max_no_aisladas": float(np.max(x2)),
        "suma_aisladas": float(np.sum(x1)), "suma_no_aisladas": float(np.sum(x2)),
    }


def magnitud(r):
    return "despreciable" if r < 0.1 else "pequeño" if r < 0.3 else "mediano" if r < 0.5 else "grande"


# ============================================================================
# 2. Excel
# ============================================================================
def escribir_excel(ruta, it, res, info, alfa):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    F = "Arial"
    f_n, f_b, f_t = Font(name=F, size=10), Font(name=F, size=10, bold=True), Font(name=F, size=13, bold=True)
    f_i = Font(name=F, size=9, italic=True, color="555555")
    f_in, f_link = Font(name=F, size=10, color="0000FF"), Font(name=F, size=10, color="008000")
    hdr, clave = PatternFill("solid", start_color="D9E1F2"), PatternFill("solid", start_color="FFF2CC")
    fino = Side(style="thin", color="999999")
    caja = Border(left=fino, right=fino, top=fino, bottom=fino)
    wb = Workbook()

    # ------------------------------------------------------------- Datos
    wd = wb.active
    wd.title = "Datos"
    wd.append(["CVEGEO", "Municipio", "Localidad", "Longitud", "Latitud", "Población 2020",
               "Inundable (1=sí)", "Aislada (1=sí)", "Tiempo base (min)", "Tiempo escenario (min)"])
    for r in it.itertuples():
        wd.append([r.CVEGEO, r.NOM_MUN, r.NOM_LOC, round(r.lon, 6), round(r.lat, 6), int(r.pob),
                   int(r.inundable), int(r.aislada), float(r.min_base),
                   None if pd.isna(r.min_escenario) else float(r.min_escenario)])
    for c in wd[1]:
        c.font, c.fill, c.alignment = f_b, hdr, Alignment(wrap_text=True)
    for row in wd.iter_rows(min_row=2):
        for c in row:
            c.font = f_n
        row[5].number_format = "#,##0"
        row[8].number_format = row[9].number_format = "0.00"
    for i, w in enumerate([14, 26, 34, 12, 11, 12, 11, 11, 13, 13], 1):
        wd.column_dimensions[get_column_letter(i)].width = w
    wd.freeze_panes = "A2"
    wd.auto_filter.ref = f"A1:J{len(it) + 1}"

    # ------------------------------------------------------------- hojas de prueba
    pos_hojas = {}
    for clave_p, col, hoja, etiqueta, unidad, fmt in PRUEBAS:
        ws = wb.create_sheet(hoja)
        d = it[["CVEGEO", "aislada", col]].sort_values(["aislada", col], ascending=[False, True])
        n1, n = int((d.aislada == 1).sum()), len(d)
        f0, fl = 2, len(d) + 1                     # filas de datos 2..fl
        fa1, fa2, fb1, fb2 = 2, 1 + n1, 2 + n1, fl  # bloques: aisladas, no aisladas
        ws.append(["CVEGEO", "Aislada (1=sí)", f"{etiqueta} ({unidad})", "Rango (promedio en empates)",
                   "t = observaciones con ese valor"])
        for c in ws[1]:
            c.font, c.fill, c.alignment = f_b, hdr, Alignment(wrap_text=True, vertical="center")
        V = f"$C${f0}:$C${fl}"
        for k, r in enumerate(d.itertuples(), start=f0):
            ws.cell(k, 1, r.CVEGEO).font = f_n
            ws.cell(k, 2, int(r.aislada)).font = f_n
            c = ws.cell(k, 3, float(getattr(r, col)) if fmt != "#,##0" else int(getattr(r, col)))
            c.font, c.number_format = f_n, fmt
            ws.cell(k, 4, f"=RANK(C{k},{V},1)+(COUNTIF({V},C{k})-1)/2").number_format = "0.0"
            ws.cell(k, 5, f"=COUNTIF({V},C{k})").number_format = "0"
        for k in (4, 5):
            for row in ws.iter_rows(min_row=f0, max_row=fl, min_col=k, max_col=k):
                row[0].font = f_n
        for L, w in zip("ABCDE", (14, 10, 22, 14, 14)):
            ws.column_dimensions[L].width = w
        ws.freeze_panes = "A2"

        # Bloque de cálculo (columnas G-I)
        ws["G1"] = f"Prueba U de Mann-Whitney: {etiqueta}"
        ws["G1"].font = f_t
        ws["G2"] = (f"Filas {fa1}-{fa2}: aisladas (n1); filas {fb1}-{fb2}: no aisladas (n2). Datos ordenados "
                    "por grupo; todas las cifras son fórmulas.")
        ws["G2"].font = f_i
        A, B = f"C{fa1}:C{fa2}", f"C{fb1}:C{fb2}"
        filas = [
            ("Descriptivos", None, None, None),
            ("n1 (aisladas)", f"=COUNTIF($B${f0}:$B${fl},1)", "#,##0", ""),
            ("n2 (no aisladas)", f"=COUNTIF($B${f0}:$B${fl},0)", "#,##0", ""),
            ("N", "=I{n1}+I{n2}", "#,##0", ""),
            ("Mediana aisladas", f"=MEDIAN({A})", fmt, ""),
            ("Mediana no aisladas", f"=MEDIAN({B})", fmt, ""),
            ("Q1 aisladas", f"=QUARTILE({A},1)", fmt, "Cuartil 1 (método inclusivo de Excel)"),
            ("Q3 aisladas", f"=QUARTILE({A},3)", fmt, ""),
            ("Q1 no aisladas", f"=QUARTILE({B},1)", fmt, ""),
            ("Q3 no aisladas", f"=QUARTILE({B},3)", fmt, ""),
            ("Media aisladas", f"=AVERAGE({A})", fmt, ""),
            ("Media no aisladas", f"=AVERAGE({B})", fmt, ""),
            ("Mínimo – máximo aisladas", f'=TEXT(MIN({A}),"{fmt}")&" – "&TEXT(MAX({A}),"{fmt}")', "@", ""),
            ("Mínimo – máximo no aisladas", f'=TEXT(MIN({B}),"{fmt}")&" – "&TEXT(MAX({B}),"{fmt}")', "@", ""),
            ("Prueba", None, None, None),
            ("R1 = suma de rangos aisladas", f"=SUMIF($B${f0}:$B${fl},1,$D${f0}:$D${fl})", "#,##0.0", ""),
            ("U1 (aisladas)", "=I{R1}-I{n1}*(I{n1}+1)/2", "#,##0.0", "R1 − n1(n1+1)/2"),
            ("U2 (no aisladas)", "=I{n1}*I{n2}-I{U1}", "#,##0.0", "n1·n2 − U1"),
            ("Σ(t³ − t) de los empates", f"=SUMPRODUCT($E${f0}:$E${fl}^2-1)", "#,##0", "Σ por observación de (t² − 1)"),
            ("μ_U", "=I{n1}*I{n2}/2", "#,##0.0", "n1·n2 / 2"),
            ("σ_U (corregida por empates)", "=SQRT(I{n1}*I{n2}/12*((I{N}+1)-I{tie}/(I{N}*(I{N}-1))))", "#,##0.00", ""),
            ("z (con continuidad)", "=(MAX(I{U1},I{U2})-I{mu}-0.5)/I{sd}", "0.000", "(max(U1,U2) − μ − 0.5)/σ"),
            ("Valor p (bilateral)", "=MIN(1,2*NORMSDIST(-I{z}))", "0.0000E+00", "2·Φ(−z) = 2·(1 − Φ(z))"),
            ("α", alfa, "0.00", "Parámetro (--alfa). Dato de entrada."),
            ("Decisión", '=IF(I{p}<I{alfa},"Se rechaza H0: las distribuciones difieren","No se rechaza H0")', "@", ""),
            ("Tamaño del efecto", None, None, None),
            ("z sin continuidad (con signo)", "=(I{U1}-I{mu})/I{sd}", "0.000", "Positivo: aisladas mayores"),
            ("r = |z| / √N", "=ABS(I{zs})/SQRT(I{N})", "0.000", "0.1 pequeño, 0.3 mediano, 0.5 grande"),
            ("Correlación biserial por rangos", "=2*I{U1}/(I{n1}*I{n2})-1", "0.000", "2·U1/(n1·n2) − 1"),
            ("Probabilidad de superioridad", "=I{U1}/(I{n1}*I{n2})", "0.0%", "P(aislada > no aislada)"),
        ]
        claves = ["h1", "n1", "n2", "N", "med1", "med2", "q11", "q31", "q12", "q32", "m1", "m2", "rg1", "rg2",
                  "h2", "R1", "U1", "U2", "tie", "mu", "sd", "z", "p", "alfa", "dec", "h3", "zs", "r", "rrb", "ps"]
        pos = {k: 4 + i for i, k in enumerate(claves)}
        for k, (et, val, fm, nota) in zip(claves, filas):
            r = pos[k]
            c = ws.cell(r, 7, et)
            if val is None:
                c.font = f_b
                continue
            c.font = f_n
            v = val.format(**pos) if isinstance(val, str) else val
            cel = ws.cell(r, 9, v)
            cel.number_format, cel.border = fm, caja
            cel.font = f_in if k == "alfa" else f_n
            ws.cell(r, 10, nota).font = f_i
        for k in ("med1", "med2", "U1", "z", "p", "dec", "r"):
            ws.cell(pos[k], 9).fill = clave
            ws.cell(pos[k], 7).font = f_b
        ws.column_dimensions["G"].width = 32
        ws.column_dimensions["H"].width = 2
        ws.column_dimensions["I"].width = 22
        ws.column_dimensions["J"].width = 36
        pos_hojas[clave_p] = (hoja, pos)

    # ------------------------------------------------------------- Resumen
    ws = wb.create_sheet("Resumen", 0)
    ws["A1"] = "Pruebas U de Mann-Whitney: localidades rurales aisladas vs. no aisladas (Edomex)"
    ws["A1"].font = f_t
    ws["A3"], ws["B3"] = "Escenario", info["escenario"]
    ws["A4"], ws["B4"] = "H0", "Las distribuciones de la variable son iguales en aisladas y no aisladas."
    ws["A5"], ws["B5"] = "H1", "Las distribuciones difieren (bilateral)."
    for r in (3, 4, 5):
        ws.cell(r, 1).font = f_b
        ws.cell(r, 2).font = f_n
        ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=3)
    ws.row_dimensions[3].height = 42
    ws["A7"] = "Resultado"
    ws["B7"] = "Prueba 1: tiempo base (min)"
    ws["C7"] = "Prueba 2: población (hab.)"
    for c in ws[7][:3]:
        c.font, c.fill, c.alignment = f_b, hdr, Alignment(wrap_text=True)
    filas_r = [("n1 (aisladas)", "n1", "#,##0"), ("n2 (no aisladas)", "n2", "#,##0"),
               ("Mediana aisladas", "med1", None), ("Mediana no aisladas", "med2", None),
               ("Q1 aisladas", "q11", None), ("Q3 aisladas", "q31", None),
               ("Q1 no aisladas", "q12", None), ("Q3 no aisladas", "q32", None),
               ("Media aisladas", "m1", None), ("Media no aisladas", "m2", None),
               ("U1 (aisladas)", "U1", "#,##0.0"), ("U2 (no aisladas)", "U2", "#,##0.0"),
               ("z (con continuidad)", "z", "0.000"), ("Valor p (bilateral)", "p", "0.0000E+00"),
               ("Decisión (α = 0.05)", "dec", "@"), ("r = |z|/√N", "r", "0.000"),
               ("Correlación biserial por rangos", "rrb", "0.000"),
               ("Probabilidad de superioridad", "ps", "0.0%")]
    for i, (et, k, fm) in enumerate(filas_r, 8):
        ws.cell(i, 1, et).font = f_n
        for j, (clave_p, *_r, fmt) in enumerate(PRUEBAS):
            hoja, pos = pos_hojas[clave_p]
            c = ws.cell(i, 2 + j, f"={hoja}!I{pos[k]}")
            c.font, c.number_format = f_link, fm or fmt
            c.alignment = Alignment(wrap_text=True)
    fi = 8 + len(filas_r) + 1
    ws.cell(fi, 1, "Interpretación").font = f_b
    for k, t in enumerate(info["interpretacion"], fi + 1):
        c = ws.cell(k, 1, t)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=k, start_column=1, end_row=k, end_column=3)
        ws.row_dimensions[k].height = 44
    ws.cell(fi + len(info["interpretacion"]) + 2, 1,
            "Texto verde = vínculo a las hojas MW_Tiempo y MW_Poblacion, donde están las fórmulas. "
            "Texto azul = dato de entrada.").font = f_i
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 34
    ws.column_dimensions["C"].width = 34

    # ------------------------------------------------------------- Python
    wp = wb.create_sheet("Python")
    wp.append(["Estadístico (scipy.stats.mannwhitneyu)", "Prueba 1: tiempo base", "Prueba 2: población"])
    for c in wp[1]:
        c.font, c.fill = f_b, hdr
    for k in res["tiempo"]:
        wp.append([k, res["tiempo"][k], res["poblacion"][k]])
    for row in wp.iter_rows(min_row=2):
        for c in row:
            c.font = f_n
        for c in row[1:]:
            if isinstance(c.value, float):
                c.number_format = "0.0000E+00" if 0 < abs(c.value) < 1e-3 else "#,##0.0000"
    wp.column_dimensions["A"].width = 32
    wp.column_dimensions["B"].width = 22
    wp.column_dimensions["C"].width = 22
    wp.cell(wp.max_row + 2, 1, "Calculado por el script 13 con scipy; debe coincidir con las hojas MW_.").font = f_i

    # ------------------------------------------------------------- Método
    wm = wb.create_sheet("Metodo")
    txt = [
        ("Unidad de análisis", "Localidad rural (< 2,500 hab., ITER 2020)."),
        ("Grupos", "Aislada = sin ruta a hospital público operativo tras cerrar los tramos con > 50 % de su longitud "
                   "inundable (TR100; puntos de muestreo cada 15 m). Mismos grupos que el script 12."),
        ("Tiempo base", "Minutos al hospital público operativo más cercano en la red completa (RNC 2025, velocidad "
                        "de la RNC, conexión a la red a 5 km/h)."),
        ("Rangos", "RANK(x, todos, 1) + (COUNTIF(todos, x) − 1)/2 = rango promedio en empates."),
        ("U", "U1 = R1 − n1(n1+1)/2; U2 = n1·n2 − U1."),
        ("Aproximación normal", "μ = n1·n2/2; σ = √(n1·n2/12·[(N+1) − Σ(t³−t)/(N(N−1))]); "
                                "z = (max(U1,U2) − μ − 0.5)/σ; p = 2(1 − Φ(z)). Igual a scipy "
                                "(method='asymptotic', use_continuity=True)."),
        ("Efecto", "r = |z|/√N; r_rb = 2U1/(n1·n2) − 1; PS = U1/(n1·n2)."),
        ("Advertencia", "La prueba supone observaciones independientes; las localidades vecinas comparten caminos "
                        "(autocorrelación espacial), lo que infla la significancia. Apoyarse en el tamaño del efecto."),
        ("Fuentes", "rnc2025Edomex.gpkg; CENAPRED_IPI_2015_Inund_Munici.zip; tr100_prcent.tif; iter_15csv20.zip; "
                    "ESTABLECIMIENTO_SALUD_202609.xlsx; scripts 02, 03, 04, 05, 07, 12 y 13."),
    ]
    for i, (k, v) in enumerate(txt, 1):
        wm.cell(i, 1, k).font = f_b
        c = wm.cell(i, 2, v)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        wm.cell(i, 1).alignment = Alignment(vertical="top")
        wm.row_dimensions[i].height = 30 if len(v) < 120 else 45
    wm.column_dimensions["A"].width = 22
    wm.column_dimensions["B"].width = 100
    wb.save(ruta)


# ============================================================================
# 3. Programa principal
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUTAS.items():
        ap.add_argument(f"--{k}", type=Path, default=v)
    ap.add_argument("--intervalo", type=float, default=s12.s07.m03.INTERVALO)
    ap.add_argument("--umbral", type=float, default=50.0)
    ap.add_argument("--alfa", type=float, default=0.05)
    a = ap.parse_args()
    for k in RUTAS:
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()
    SALIDAS.mkdir(exist_ok=True)

    print("1/3 Aislamiento y tiempo base de cada localidad rural...")
    it, n_hosp, n_cerr, n_tr = s12.variables_localidades(a)
    if not np.isfinite(it["min_base"]).all():
        raise SystemExit("Hay localidades sin ruta en la red completa; el tiempo base no está definido para todas.")

    print("2/3 Pruebas U de Mann-Whitney...")
    ais = it["aislada"] == 1
    res = {clave: mann_whitney(it.loc[ais, col].to_numpy(float), it.loc[~ais, col].to_numpy(float))
           for clave, col, *_ in PRUEBAS}

    rt, rp = res["tiempo"], res["poblacion"]

    def frase(r, var, un, fmt):
        dec = "difieren" if r["p_valor"] < a.alfa else "no difieren significativamente"
        mayor = "mayores" if r["correlacion_biserial_rangos"] > 0 else "menores"
        return (f"{var}: mediana {fmt(r['mediana_aisladas'])} {un} en aisladas vs. {fmt(r['mediana_no_aisladas'])} {un} "
                f"en no aisladas; U = {r['U1_aisladas']:,.0f}, z = {r['z_con_continuidad']:.2f}, "
                f"p = {r['p_valor']:.2e}: las distribuciones {dec}. Las aisladas tienden a valores {mayor} "
                f"(probabilidad de superioridad {100 * r['prob_superioridad']:.1f} %; r = {r['r_efecto']:.3f}, "
                f"efecto {magnitud(r['r_efecto'])}).")

    interp = [
        frase(rt, "Tiempo base", "min", lambda v: f"{v:.1f}"),
        frase(rp, "Población", "hab.", lambda v: f"{v:,.0f}"),
        "Cautela: las localidades vecinas no son observaciones independientes (comparten caminos); el valor p "
        "tiende a exagerarse y el tamaño del efecto es más informativo.",
    ]
    info = {"escenario": (f"Tramos con > {a.umbral:g} % de su longitud inundable (TR100) cerrados ({n_cerr:,d} de "
                          f"{n_tr:,d}); {n_hosp} hospitales públicos operativos (franja de 20 km); puntos de "
                          f"muestreo cada {a.intervalo:g} m; {len(it):,d} localidades rurales."),
            "interpretacion": interp}

    print("3/3 Escribiendo Excel...")
    ruta_x = SALIDAS / f"{NOMBRE}.xlsx"
    escribir_excel(ruta_x, it, res, info, a.alfa)
    pd.DataFrame(res).to_csv(SALIDAS / "13_resultados_mann_whitney.csv", encoding="utf-8-sig")

    print("\n" + "=" * 86)
    print(" PRUEBAS U DE MANN-WHITNEY: localidades rurales aisladas vs. no aisladas")
    print("=" * 86)
    print(f"  {info['escenario']}")
    for (clave, col, hoja, etiqueta, unidad, fmt) in PRUEBAS:
        r = res[clave]
        f = (lambda v: f"{v:,.1f}") if clave == "tiempo" else (lambda v: f"{v:,.0f}")
        print(f"\n  {etiqueta} ({unidad})")
        print(f"    {'':22s}{'n':>7s}{'Mediana':>11s}{'Q1':>10s}{'Q3':>10s}{'Media':>11s}")
        for g, s in (("Aisladas", "aisladas"), ("No aisladas", "no_aisladas")):
            print(f"    {g:22s}{r['n_' + s]:>7,d}{f(r['mediana_' + s]):>11s}{f(r['q1_' + s]):>10s}"
                  f"{f(r['q3_' + s]):>10s}{f(r['media_' + s]):>11s}")
        print(f"    U1 = {r['U1_aisladas']:,.1f}   U2 = {r['U2_no_aisladas']:,.1f}   z = {r['z_con_continuidad']:.3f}   "
              f"p = {r['p_valor']:.3e}  ->  " + ("SE RECHAZA H0" if r["p_valor"] < a.alfa else "no se rechaza H0"))
        print(f"    r = {r['r_efecto']:.3f} ({magnitud(r['r_efecto'])}); biserial por rangos = "
              f"{r['correlacion_biserial_rangos']:+.3f}; P(aislada > no aislada) = {100 * r['prob_superioridad']:.1f} %")
    print("\n  Interpretación:")
    for t in interp:
        print("   - " + t)
    print(f"\n  Excel: {ruta_x}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
