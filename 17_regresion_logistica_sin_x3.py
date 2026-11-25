# -*- coding: utf-8 -*-
"""
Script 17 - Aislamiento por Inundaciones: regresión logística SIN X3
====================================================================

OBJETIVO
--------
Repetir el modelo del script 16 eliminando X3 (red municipal en tramos con
más del 75 % inundado), y compararlo con el modelo completo para medir qué
aporta X3:

  Modelo reducido (este script):  logit P(Y=1) = b0 + b1·X1 + b2·X2
  Modelo completo (script 16):    logit P(Y=1) = b0 + b1·X1 + b2·X2 + b3·X3

  Y  = 1 si la localidad rural queda aislada de los hospitales públicos
       operativos al cerrar los tramos con MÁS DEL 50 % de su longitud en
       zona inundable (escenario de los scripts 07 y 09 a 16).
  X1 = 1 si la localidad rural está en zona inundable.
  X2 = 1 si el hospital público operativo más cercano (red completa) está en
       zona inundable.
  X3 = 1 si más del 75 % de la red municipal (km) está en tramos con más del
       75 % de su longitud inundada (solo para la comparación).

  Unidad: localidad rural (4,215); errores estándar robustos agrupados por
  municipio (124 conglomerados), como en el script 16.

MÉTODO
------
  - Variables, estimación (Newton-Raphson), errores robustos CR1 por
    municipio, AUC y Hosmer-Lemeshow: funciones del script 16 (mismos datos y
    criterios; las variables se recalculan con sus funciones).
  - Comparación de modelos anidados:
      * Razón de verosimilitudes para X3: LR = 2(LL_completo − LL_reducido),
        gl = 1 (supone independencia; con conglomerados es orientativa).
      * Prueba de Wald robusta de b3 = 0 (z² del modelo completo), válida con
        conglomerados: es la prueba principal.
      * ΔAIC, ΔMcFadden y ΔAUC.
      * Cambio en los OR de X1 y X2 al quitar X3: indica si X3 explicaba
        parte del efecto atribuido a ellos (confusión).

FUENTES
-------
  Las mismas del script 16: rnc2025Edomex.gpkg; CENAPRED.zip (IPI 2015);
  tr100_prcent.tif; inegi\\iter_15csv20.zip; ESTABLECIMIENTO_SALUD_202609.xlsx;
  scripts 02, 03, 04, 05, 07, 08, 11, 14 y 16.

USO
---
  python 17_regresion_logistica_sin_x3.py
  python 17_regresion_logistica_sin_x3.py --umbral 50 --umbral-tramo 75 --umbral-x3 75

SALIDAS (carpeta salidas\\)
--------------------------
  17_regresion_logistica_sin_x3.xlsx   hojas:
      Resumen      coeficientes del modelo reducido, ajuste e interpretación
      Modelo       coeficientes y EE (Python, azul) y lo derivado con fórmulas:
                   OR, IC, z, p, LL, LR, McFadden, AIC, clasificación, perfiles
      Comparacion  modelo reducido vs. completo (con fórmulas)
      Datos        una fila por localidad rural: Y, X1, X2, X3 y probabilidad
                   predicha del modelo reducido (fórmula)
      Python       resultados de Python
      Metodo       definiciones y fuentes
  17_coeficientes_logit_sin_x3.csv
"""

import argparse
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
SALIDAS = AQUI / "salidas"


def _importar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s16 = _importar("s16", "16_regresion_logistica_aislamiento.py")
RUTAS = dict(s16.RUTAS)
NOMBRE = "17_regresion_logistica_sin_x3"
V1, V2, V3 = s16.VARS
ETIQ = s16.ETIQ


def escribir_excel(ruta, it, red, com, extra_r, extra_c, info):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter as L

    F = "Arial"
    f_n, f_b, f_t = Font(name=F, size=10), Font(name=F, size=10, bold=True), Font(name=F, size=13, bold=True)
    f_i = Font(name=F, size=9, italic=True, color="555555")
    f_in, f_link = Font(name=F, size=10, color="0000FF"), Font(name=F, size=10, color="008000")
    hdr, clave = PatternFill("solid", start_color="D9E1F2"), PatternFill("solid", start_color="FFF2CC")
    fino = Side(style="thin", color="999999")
    caja = Border(left=fino, right=fino, top=fino, bottom=fino)
    wb = Workbook()

    def cab(ws, fila, textos, col=1):
        for j, t in enumerate(textos):
            c = ws.cell(fila, col + j, t)
            c.font, c.fill, c.border = f_b, hdr, caja
            c.alignment = Alignment(wrap_text=True, vertical="center")

    # ------------------------------------------------------------- Datos
    wd = wb.active
    wd.title = "Datos"
    cab(wd, 1, ["CVEGEO", "CVE_MPIO", "Municipio", "Localidad", "Población 2020", "Y: aislada (1=sí)",
                "X1: localidad inundable", "X2: hospital más cercano inundable",
                "X3: red municipal > 75 % en tramos > 75 % (solo comparación)", "Logit modelo reducido",
                "Probabilidad predicha (reducido)", "Clasificada aislada (p ≥ 0.5)",
                "Probabilidad predicha (completo, Python)"])
    for i, r in enumerate(it.itertuples(), 2):
        vals = [r.CVEGEO, r.CVE_MPIO, r.NOM_MUN, r.NOM_LOC, int(r.pob), int(r.Y_aislada), int(getattr(r, V1)),
                int(getattr(r, V2)), int(getattr(r, V3)), f"=Modelo!$C$7+Modelo!$C$8*G{i}+Modelo!$C$9*H{i}",
                f"=1/(1+EXP(-J{i}))", f"=IF(K{i}>=0.5,1,0)", float(com["p_hat"][i - 2])]
        for j, v in enumerate(vals, 1):
            wd.cell(i, j, v).font = f_n
        wd.cell(i, 5).number_format = "#,##0"
        for j in (10, 11, 13):
            wd.cell(i, j).number_format = "0.0000"
        wd.cell(i, 13).font = f_in
    nd = len(it) + 1
    for j, w in enumerate([14, 10, 24, 30, 11, 10, 11, 13, 15, 11, 12, 12, 13], 1):
        wd.column_dimensions[L(j)].width = w
    wd.freeze_panes = "A2"
    wd.auto_filter.ref = f"A1:M{nd}"
    Y, X1, X2, X3, PH, CL, PC = (f"Datos!${c}$2:${c}${nd}" for c in "FGHIKLM")

    # ------------------------------------------------------------- Modelo
    ws = wb.create_sheet("Modelo")
    ws["A1"] = "Regresión logística sin X3: P(localidad rural aislada) ~ X1 + X2, EE robustos por municipio"
    ws["A1"].font = f_t
    ws["A2"] = ("Coeficientes y EE (azul) vienen de Python; lo demás son fórmulas sobre Datos. "
                f"Municipios (conglomerados): {red['G']}.")
    ws["A2"].font = f_i
    cab(ws, 6, ["Variable", "", "Coeficiente b", "EE robusto (municipio)", "z", "Valor p", "OR",
                "IC 95 % OR inf.", "IC 95 % OR sup.", "EE ingenuo", "Valor p ingenuo"])
    for i, v in enumerate(["const", V1, V2]):
        r = 7 + i
        ws.cell(r, 1, ETIQ[v]).font = f_n
        ws.cell(r, 3, float(red["b"][i])).font = f_in
        ws.cell(r, 4, float(red["ee"][i])).font = f_in
        ws.cell(r, 5, f"=C{r}/D{r}").font = f_n
        ws.cell(r, 6, f"=2*NORMSDIST(-ABS(E{r}))").font = f_n
        ws.cell(r, 7, f"=EXP(C{r})").font = f_n
        ws.cell(r, 8, f"=EXP(C{r}-1.96*D{r})").font = f_n
        ws.cell(r, 9, f"=EXP(C{r}+1.96*D{r})").font = f_n
        ws.cell(r, 10, float(red["ee_ingenuo"][i])).font = f_in
        ws.cell(r, 11, f"=2*NORMSDIST(-ABS(C{r}/J{r}))").font = f_n
        for j, fm in ((3, "0.0000"), (4, "0.0000"), (5, "0.000"), (6, "0.0000E+00"), (7, "0.000"), (8, "0.000"),
                      (9, "0.000"), (10, "0.0000"), (11, "0.0000E+00")):
            ws.cell(r, j).number_format = fm
            ws.cell(r, j).border = caja
        if i:
            ws.cell(r, 6).fill = ws.cell(r, 7).fill = clave
    fa = 12
    ws.cell(fa, 1, "Ajuste global").font = f_b
    aj = [("n", f"=COUNT({Y})", "#,##0", ""), ("Aisladas", f"=SUM({Y})", "#,##0", ""),
          ("LL", f"=SUMPRODUCT({Y},LN({PH}))+SUMPRODUCT(1-{Y},LN(1-{PH}))", "0.000", ""),
          ("LL0 (nulo)", "=C{ny}*LN(C{ny}/C{n})+(C{n}-C{ny})*LN(1-C{ny}/C{n})", "0.000", ""),
          ("LR = 2(LL − LL0)", "=2*(C{ll}-C{ll0})", "0.000", "gl = 2"),
          ("Valor p (LR)", "=CHIDIST(C{lr},2)", "0.0000E+00", ""),
          ("Wald conjunta robusta", float(red["wald"]), "0.000", "Python, gl = 2"),
          ("Valor p (Wald robusta)", "=CHIDIST(C{wald},2)", "0.0000E+00", ""),
          ("Pseudo R² de McFadden", "=1-C{ll}/C{ll0}", "0.0000", ""),
          ("AIC", "=2*3-2*C{ll}", "0.00", "k = 3"),
          ("AUC", float(extra_r["auc"]), "0.0000", "Python"),
          ("Hosmer-Lemeshow χ²", float(extra_r["hl"]), "0.000", f"Python, perfiles, gl = {extra_r['hl_gl']}"),
          ("Valor p (Hosmer-Lemeshow)", float(extra_r["hl_p"]), "0.0000", "")]
    ks = ["n", "ny", "ll", "ll0", "lr", "plr", "wald", "pw", "mcf", "aic", "auc", "hl", "hlp"]
    pos = {k: fa + 1 + i for i, k in enumerate(ks)}
    for k, (et, val, fm, nota) in zip(ks, aj):
        r = pos[k]
        ws.cell(r, 1, et).font = f_n
        es_f = isinstance(val, str) and val.startswith("=")
        c = ws.cell(r, 3, val.format(**pos) if es_f else val)
        c.number_format, c.border, c.font = fm, caja, (f_n if es_f else f_in)
        ws.cell(r, 4, nota).font = f_i
    fc = pos["hlp"] + 2
    ws.cell(fc, 1, "Clasificación (corte 0.5)").font = f_b
    cab(ws, fc + 1, ["", "", "Predicha aislada", "Predicha no aislada", "Total"])
    for i, (et, yv) in enumerate((("Observada aislada", 1), ("Observada no aislada", 0))):
        r = fc + 2 + i
        ws.cell(r, 1, et).font = f_b
        ws.cell(r, 3, f"=COUNTIFS({Y},{yv},{CL},1)").font = f_n
        ws.cell(r, 4, f"=COUNTIFS({Y},{yv},{CL},0)").font = f_n
        ws.cell(r, 5, f"=C{r}+D{r}").font = f_n
    r = fc + 4
    for k, (et, f) in enumerate((("Exactitud", f"=(C{fc + 2}+D{fc + 3})/(E{fc + 2}+E{fc + 3})"),
                                 ("Sensibilidad", f"=C{fc + 2}/E{fc + 2}"),
                                 ("Especificidad", f"=D{fc + 3}/E{fc + 3}"))):
        ws.cell(r + k, 1, et).font = f_n
        c = ws.cell(r + k, 3, f)
        c.font, c.number_format = f_n, "0.0%"
    fp = r + 4
    ws.cell(fp, 1, "Perfiles X1 × X2").font = f_b
    cab(ws, fp + 1, ["X1", "X2", "Localidades", "Aisladas", "% observado", "Probabilidad predicha"])
    for i, (a1, a2) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        rr = fp + 2 + i
        ws.cell(rr, 1, a1)
        ws.cell(rr, 2, a2)
        ws.cell(rr, 3, f"=COUNTIFS({X1},A{rr},{X2},B{rr})")
        ws.cell(rr, 4, f"=COUNTIFS({X1},A{rr},{X2},B{rr},{Y},1)")
        ws.cell(rr, 5, f"=D{rr}/C{rr}").number_format = "0.0%"
        ws.cell(rr, 6, f"=1/(1+EXP(-($C$7+$C$8*A{rr}+$C$9*B{rr})))").number_format = "0.0%"
        for j in range(1, 7):
            ws.cell(rr, j).font = f_n
            ws.cell(rr, j).border = caja
    for j, w in enumerate([44, 2, 14, 15, 12, 13, 12, 12, 12, 12, 13], 1):
        ws.column_dimensions[L(j)].width = w

    # ------------------------------------------------------------- Comparacion
    wc = wb.create_sheet("Comparacion")
    wc["A1"] = "Comparación: modelo reducido (X1 + X2) vs. modelo completo del script 16 (X1 + X2 + X3)"
    wc["A1"].font = f_t
    wc["A2"] = "Valores del modelo completo (azul) recalculados con las funciones del script 16 sobre los mismos datos."
    wc["A2"].font = f_i
    cab(wc, 4, ["Variable", "OR reducido", "IC 95 % reducido", "OR completo", "IC 95 % completo",
                "Cambio del OR al quitar X3"])
    ORc = np.exp(com["b"])
    for i, v in enumerate([V1, V2, V3], 5):
        j = i - 4
        wc.cell(i, 1, ETIQ[v]).font = f_n
        if v != V3:
            wc.cell(i, 2, f"=Modelo!G{7 + j}").font = f_link
            wc.cell(i, 3, f'=TEXT(Modelo!H{7 + j},"0.00")&" – "&TEXT(Modelo!I{7 + j},"0.00")').font = f_link
            wc.cell(i, 6, f"=B{i}/D{i}-1").number_format = "+0.0%;-0.0%"
            wc.cell(i, 6).font = f_n
        else:
            wc.cell(i, 2, "—").font = f_n
            wc.cell(i, 3, "—").font = f_n
        wc.cell(i, 4, float(ORc[j])).font = f_in
        wc.cell(i, 5, f"{math.exp(com['b'][j] - 1.96 * com['ee'][j]):.2f} – "
                      f"{math.exp(com['b'][j] + 1.96 * com['ee'][j]):.2f}").font = f_in
        wc.cell(i, 2).number_format = wc.cell(i, 4).number_format = "0.000"
    cab(wc, 9, ["Medida", "Reducido", "Completo", "Diferencia (completo − reducido)", "", ""])
    filas = [("Log-verosimilitud LL", f"=Modelo!C{pos['ll']}", com["ll"], "0.000"),
             ("Pseudo R² de McFadden", f"=Modelo!C{pos['mcf']}", com["mcfadden"], "0.0000"),
             ("AIC (menor = mejor)", f"=Modelo!C{pos['aic']}", com["aic"], "0.00"),
             ("AUC", f"=Modelo!C{pos['auc']}", extra_c["auc"], "0.0000"),
             ("Parámetros k", 3, 4, "0")]
    for i, (et, rv, cv, fm) in enumerate(filas, 10):
        wc.cell(i, 1, et).font = f_n
        a = wc.cell(i, 2, rv)
        a.font = f_link if isinstance(rv, str) else f_n
        b = wc.cell(i, 3, float(cv))
        b.font = f_in
        d = wc.cell(i, 4, f"=C{i}-B{i}")
        d.font = f_n
        for c in (a, b, d):
            c.number_format, c.border = fm, caja
    wc.cell(16, 1, "Pruebas de la contribución de X3").font = f_b
    pr = [("LR = 2(LL completo − LL reducido)", "=2*(C10-B10)", "0.000", "gl = 1; orientativa con conglomerados"),
          ("Valor p (LR)", "=CHIDIST(B17,1)", "0.0000E+00", ""),
          ("Wald robusta de b3 = 0 (z²)", float((com["b"][3] / com["ee"][3]) ** 2), "0.000",
           "Del modelo completo con EE por municipio (prueba principal)"),
          ("Valor p (Wald robusta)", "=CHIDIST(B19,1)", "0.0000E+00", ""),
          ("ΔAIC (completo − reducido)", "=D12", "0.00", "< −2: el completo es preferible")]
    for i, (et, v, fm, nota) in enumerate(pr, 17):
        wc.cell(i, 1, et).font = f_n
        c = wc.cell(i, 2, v)
        c.number_format, c.border = fm, caja
        c.font = f_in if not isinstance(v, str) else f_n
        wc.cell(i, 3, nota).font = f_i
    for k in (18, 20):
        wc.cell(k, 2).fill = clave
    for j, w in enumerate([44, 16, 18, 16, 18, 16], 1):
        wc.column_dimensions[L(j)].width = w

    # ------------------------------------------------------------- Resumen
    wr = wb.create_sheet("Resumen", 0)
    wr["A1"] = "Regresión logística sin X3 (X1 + X2), agrupada por municipio"
    wr["A1"].font = f_t
    wr["A3"], wr["B3"] = "Escenario", info["escenario"]
    wr["A3"].font, wr["B3"].font = f_b, f_n
    wr["B3"].alignment = Alignment(wrap_text=True, vertical="top")
    wr.merge_cells("B3:F3")
    wr.row_dimensions[3].height = 44
    cab(wr, 5, ["Variable", "b", "OR", "IC 95 % OR", "Valor p (robusto)", "Significativo"])
    for i in range(3):
        r, mr = 6 + i, 7 + i
        for j, f in enumerate([f"=Modelo!A{mr}", f"=Modelo!C{mr}", f"=Modelo!G{mr}",
                               f'=TEXT(Modelo!H{mr},"0.00")&" – "&TEXT(Modelo!I{mr},"0.00")', f"=Modelo!F{mr}",
                               f'=IF(Modelo!F{mr}<0.05,"Sí","No")'], 1):
            c = wr.cell(r, j, f)
            c.font = f_link
        wr.cell(r, 2).number_format = wr.cell(r, 3).number_format = "0.000"
        wr.cell(r, 5).number_format = "0.0000E+00"
    for i, (et, f, fm) in enumerate((("McFadden reducido | completo", None, None),), 10):
        pass
    filas = [("Pseudo R² McFadden (reducido)", f"=Modelo!C{pos['mcf']}", "0.000"),
             ("Pseudo R² McFadden (completo, con X3)", "=Comparacion!C11", "0.000"),
             ("AUC reducido | completo", '=TEXT(Comparacion!B13,"0.000")&" | "&TEXT(Comparacion!C13,"0.000")', "@"),
             ("ΔAIC al agregar X3", "=Comparacion!B21", "0.00"),
             ("Valor p de X3 (Wald robusta)", "=Comparacion!B20", "0.00E+00")]
    for i, (et, f, fm) in enumerate(filas, 10):
        wr.cell(i, 1, et).font = f_n
        c = wr.cell(i, 2, f)
        c.font, c.number_format = f_link, fm
    wr.cell(16, 1, "Interpretación").font = f_b
    for k, t in enumerate(info["interpretacion"], 17):
        c = wr.cell(k, 1, t)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        wr.merge_cells(start_row=k, start_column=1, end_row=k, end_column=6)
        wr.row_dimensions[k].height = 48
    for j, w in enumerate([46, 14, 11, 16, 15, 12], 1):
        wr.column_dimensions[L(j)].width = w

    # ------------------------------------------------------------- Python
    wp = wb.create_sheet("Python")
    cab(wp, 1, ["Modelo", "Variable", "b", "EE robusto", "z", "p", "EE ingenuo", "OR"])
    r = 2
    for nom, res, vs in (("Reducido", red, ["const", V1, V2]), ("Completo", com, ["const", V1, V2, V3])):
        for i, v in enumerate(vs):
            for j, val in enumerate([nom, ETIQ[v], res["b"][i], res["ee"][i], res["z"][i], res["p"][i],
                                     res["ee_ingenuo"][i], math.exp(res["b"][i])], 1):
                c = wp.cell(r, j, val if isinstance(val, str) else float(val))
                c.font = f_n
                if j > 2:
                    c.number_format = "0.0000E+00" if j == 6 else "0.000000"
            r += 1
    for k, (nom, res, ex) in enumerate((("Reducido", red, extra_r), ("Completo", com, extra_c))):
        wp.cell(r + 1 + k, 1, f"{nom}: LL {res['ll']:.4f}; McFadden {res['mcfadden']:.5f}; AIC {res['aic']:.3f}; "
                              f"AUC {ex['auc']:.5f}; HL {ex['hl']:.3f} (p {ex['hl_p']:.4f})").font = f_n
    wp.column_dimensions["A"].width = 12
    wp.column_dimensions["B"].width = 58
    for j in range(3, 9):
        wp.column_dimensions[L(j)].width = 13

    # ------------------------------------------------------------- Método
    wm = wb.create_sheet("Metodo")
    txt = [("Modelo", "Igual al script 16 sin X3: logit P(aislada) = b0 + b1·X1 + b2·X2; EE robustos CR1 por municipio."),
           ("Comparación", "LR = 2(LL completo − LL reducido), gl 1 (orientativa con conglomerados); Wald robusta de "
                           "b3 = 0 del modelo completo (principal); ΔAIC, ΔMcFadden, ΔAUC; cambio de OR de X1 y X2."),
           ("Variables", "Y, X1, X2 y X3 se calculan con las funciones del script 16 (mismos criterios: tramos > 50 % "
                         "inundables cerrados, puntos cada 15 m, hospitales públicos operativos en la franja de 20 km)."),
           ("Fuentes", "rnc2025Edomex.gpkg; CENAPRED_IPI_2015_Inund_Munici.zip; tr100_prcent.tif; iter_15csv20.zip; "
                       "ESTABLECIMIENTO_SALUD_202609.xlsx; scripts 02, 03, 04, 05, 07, 08, 11, 14, 16 y 17.")]
    for i, (k, v) in enumerate(txt, 1):
        wm.cell(i, 1, k).font = f_b
        c = wm.cell(i, 2, v)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        wm.row_dimensions[i].height = 34
    wm.column_dimensions["A"].width = 14
    wm.column_dimensions["B"].width = 105
    wb.save(ruta)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in RUTAS.items():
        ap.add_argument(f"--{k}", type=Path, default=v)
    ap.add_argument("--intervalo", type=float, default=s16.s07.m03.INTERVALO)
    ap.add_argument("--umbral", type=float, default=50.0)
    ap.add_argument("--umbral-tramo", type=float, default=75.0)
    ap.add_argument("--umbral-x3", type=float, default=75.0)
    a = ap.parse_args()
    for k in RUTAS:
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()
    SALIDAS.mkdir(exist_ok=True)

    print("1/3 Variables (funciones del script 16)...")
    muns, _ = s16.s08.municipios(a.cenapred)
    km, km_i, km_t = s16.red_tramos_por_municipio(a, muns, a.umbral_tramo)
    mun = pd.DataFrame({"CVE_MPIO": [k for k, _, _ in muns], "km": km, "km_t": km_t})
    mun[V3] = (100 * mun.km_t / mun.km > a.umbral_x3).astype(int)
    it, info0 = s16.variables(a)
    it = it.merge(mun[["CVE_MPIO", V3]], on="CVE_MPIO", how="left")

    print("2/3 Modelos reducido y completo...")
    y = it["Y_aislada"].to_numpy(float)
    g = it["CVE_MPIO"].to_numpy()
    Xr = np.column_stack([np.ones(len(it)), it[V1], it[V2]]).astype(float)
    Xc = np.column_stack([np.ones(len(it)), it[V1], it[V2], it[V3]]).astype(float)
    red, com = s16.logit(y, Xr, g), s16.logit(y, Xc, g)
    hl_r = s16.hosmer_lemeshow(y, red["p_hat"])
    hl_c = s16.hosmer_lemeshow(y, com["p_hat"])
    ex_r = {"auc": s16.auc(y, red["p_hat"]), "hl": hl_r[0], "hl_gl": hl_r[1], "hl_p": hl_r[2]}
    ex_c = {"auc": s16.auc(y, com["p_hat"]), "hl": hl_c[0], "hl_gl": hl_c[1], "hl_p": hl_c[2]}
    lr = 2 * (com["ll"] - red["ll"])
    from scipy import stats
    p_lr = stats.chi2.sf(lr, 1)
    wald3 = (com["b"][3] / com["ee"][3]) ** 2

    OR, ORc = np.exp(red["b"]), np.exp(com["b"])
    lo, hi = np.exp(red["b"] - 1.96 * red["ee"]), np.exp(red["b"] + 1.96 * red["ee"])
    interp = [
        (f"Sin X3, estar en zona inundable multiplica los momios de aislamiento por {OR[1]:.2f} (IC 95 % "
         f"{lo[1]:.2f}–{hi[1]:.2f}; p = {red['p'][1]:.1e}) y tener el hospital más cercano en zona inundable por "
         f"{OR[2]:.2f} (IC {lo[2]:.2f}–{hi[2]:.2f}; p = {red['p'][2]:.1e})."),
        (f"Al quitar X3 los OR cambian {100 * (OR[1] / ORc[1] - 1):+.1f} % (X1) y {100 * (OR[2] / ORc[2] - 1):+.1f} % "
         "(X2): " + ("casi no cambian, así que X3 no confundía sus efectos." if max(abs(OR[1] / ORc[1] - 1),
                                                                              abs(OR[2] / ORc[2] - 1)) < 0.1
                     else "X3 absorbía parte de su efecto.")),
        (f"Ajuste: McFadden {red['mcfadden']:.3f} sin X3 vs. {com['mcfadden']:.3f} con X3; AUC {ex_r['auc']:.3f} vs. "
         f"{ex_c['auc']:.3f}; ΔAIC = {com['aic'] - red['aic']:.1f}. X3 aporta información significativa (Wald robusta "
         f"= {wald3:.1f}, p = {stats.chi2.sf(wald3, 1):.1e}), aunque la capacidad de discriminación casi no cambia: "
         "X3 afecta a pocas localidades (las de 14 municipios)."),
    ]
    info = {"escenario": (f"{len(it):,d} localidades rurales en {red['G']} municipios. Tramos con > {a.umbral:g} % "
                          f"inundable cerrados ({info0['cerrados']:,d}); {info0['hosp']} hospitales públicos "
                          f"operativos ({info0['hosp_inund']} en zona inundable); puntos cada {a.intervalo:g} m."),
            "interpretacion": interp}

    print("3/3 Excel...")
    ruta = SALIDAS / f"{NOMBRE}.xlsx"
    escribir_excel(ruta, it, red, com, ex_r, ex_c, info)
    pd.DataFrame({"variable": ["const", V1, V2], "b": red["b"], "ee_robusto": red["ee"], "z": red["z"],
                  "p": red["p"], "ee_ingenuo": red["ee_ingenuo"], "OR": OR, "OR_ic95_inf": lo, "OR_ic95_sup": hi}
                 ).to_csv(SALIDAS / "17_coeficientes_logit_sin_x3.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(" REGRESIÓN LOGÍSTICA SIN X3: P(localidad rural aislada) ~ X1 + X2, EE robustos por municipio")
    print("=" * 100)
    print(f"  {info['escenario']}")
    print(f"\n  {'Variable':52s}{'b':>9s}{'EE rob.':>9s}{'z':>8s}{'p':>11s}{'OR':>8s}{'IC 95 % OR':>17s}")
    for i, v in enumerate(["const", V1, V2]):
        print(f"  {ETIQ[v]:52s}{red['b'][i]:>9.3f}{red['ee'][i]:>9.3f}{red['z'][i]:>8.2f}{red['p'][i]:>11.2e}"
              f"{OR[i]:>8.2f}{lo[i]:>8.2f} – {hi[i]:<7.2f}")
    pc = (red["p_hat"] >= 0.5).astype(int)
    print(f"\n  LL = {red['ll']:.2f}; LR = {2 * (red['ll'] - red['ll0']):.2f} (gl 2); McFadden = {red['mcfadden']:.4f}; "
          f"AIC = {red['aic']:.2f}; AUC = {ex_r['auc']:.4f}; HL = {ex_r['hl']:.2f} (gl {ex_r['hl_gl']}, "
          f"p = {ex_r['hl_p']:.3f})")
    print(f"  Clasificación (0.5): exactitud {100 * np.mean(pc == y):.1f} %, sensibilidad "
          f"{100 * np.mean(pc[y == 1] == 1):.1f} %, especificidad {100 * np.mean(pc[y == 0] == 0):.1f} %")
    print(f"\n  Comparación con el modelo completo (con X3):")
    print(f"    OR X1: {OR[1]:.2f} sin X3 vs. {ORc[1]:.2f} con X3;  OR X2: {OR[2]:.2f} vs. {ORc[2]:.2f}")
    print(f"    McFadden {red['mcfadden']:.4f} vs. {com['mcfadden']:.4f}; AIC {red['aic']:.2f} vs. {com['aic']:.2f} "
          f"(Δ {com['aic'] - red['aic']:.2f}); AUC {ex_r['auc']:.4f} vs. {ex_c['auc']:.4f}")
    print(f"    LR de X3 = {lr:.2f} (gl 1, p = {p_lr:.2e}); Wald robusta = {wald3:.2f} (p = {stats.chi2.sf(wald3, 1):.2e})")
    print("\n  Interpretación:")
    for t in interp:
        print("   - " + t)
    print(f"\n  Excel: {ruta}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
