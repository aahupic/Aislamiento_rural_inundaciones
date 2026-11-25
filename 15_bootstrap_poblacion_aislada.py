# -*- coding: utf-8 -*-
"""
Script 15 - Aislamiento por Inundaciones: intervalos Bootstrap (B = 2500)
========================================================================

OBJETIVO
--------
Estimar la incertidumbre (error estándar e intervalos de confianza al 95 %)
de dos indicadores de la población rural (localidades < 2,500 hab.) del
Estado de México en el escenario de inundación:

  P1 = % de población rural aislada
       = Σ pob (localidades aisladas) / Σ pob (todas las rurales) × 100

  P2 = % de población rural aislada FUERA de zona inundable
       = Σ pob (aisladas y fuera de zona inundable) / Σ pob (todas) × 100
       (la parte del aislamiento que es efecto indirecto: personas que no se
       inundan pero quedan incomunicadas)

  Como complemento se reporta P3 = proporción de los aislados que están
  fuera de zona inundable = Σ pob (aisladas fuera) / Σ pob (aisladas) × 100.
  Es otra lectura posible de "% de población rural aislada fuera de zona
  inundable"; P2 y P3 tienen distinto denominador.

ESCENARIO (idéntico a los scripts 09 a 14)
-----------------------------------------
  Puntos de muestreo cada 15 m (script 03); se cierran los tramos con más
  del 50 % de su longitud en zona inundable TR100; destino = hospitales
  públicos operativos en la franja de 20 km; aislada = sin ruta. Zona
  inundable de la localidad = su punto (ITER 2020) en píxel TR100 = 0.
  Las variables se obtienen con la función del script 12.

MÉTODO BOOTSTRAP
----------------
  Bootstrap no paramétrico con B = 2500 réplicas y semilla fija
  (--semilla, por defecto 20261124) para que el resultado sea reproducible.

  A. Bootstrap simple por LOCALIDAD (principal)
     - Cada réplica toma n = 4,215 localidades rurales CON REEMPLAZO.
     - En cada réplica se recalculan P1, P2 y P3 como cocientes de sumas de
       población (estimador de razón: las localidades grandes pesan más,
       igual que en el valor observado).
  B. Bootstrap por CONGLOMERADOS (MUNICIPIO) (sensibilidad)
     - Cada réplica toma municipios con reemplazo (todas las localidades
       rurales de cada municipio elegido). Respeta que las localidades
       vecinas no son independientes (comparten caminos y se aíslan
       juntas), que es la principal advertencia de las pruebas 12 a 14.
       Suele dar intervalos más amplios y más realistas.

  Para cada indicador y método se reporta:
    - valor observado (θ),
    - media de las réplicas y sesgo = media − θ,
    - error estándar bootstrap = desviación estándar de las réplicas,
    - IC 95 % percentil: percentiles 2.5 y 97.5 de las réplicas,
    - IC 95 % básico (pivotal): [2θ − P97.5, 2θ − P2.5],
    - IC 95 % BCa (corregido por sesgo y aceleración; Efron 1987):
        z0 = Φ⁻¹( proporción de réplicas < θ )
        a  = Σ (θ̄(·) − θ(i))³ / ( 6 · [Σ (θ̄(·) − θ(i))²]^1.5 ),
             θ(i) = estimación jackknife sin la unidad i (localidad o
             municipio según el método)
        α1 = Φ( z0 + (z0 + z_0.025) / (1 − a(z0 + z_0.025)) ),
        α2 = Φ( z0 + (z0 + z_0.975) / (1 − a(z0 + z_0.975)) );
        IC = percentiles α1 y α2 de las réplicas.
    - Variabilidad Monte Carlo: con B = 2500 los límites pueden moverse en
      centésimas de punto porcentual si se cambia la semilla.

  Interpretación: el bootstrap mide la incertidumbre por la MUESTRA de
  localidades (qué tan estable es el indicador si el conjunto de
  localidades hubiera sido otro de la misma población). No incluye la
  incertidumbre del modelo (ráster TR100, velocidades, umbral de 50 %), que
  se explora con los escenarios de los scripts 07 y 10.

FUENTES
-------
  rnc2025Edomex.gpkg; CENAPRED.zip (IPI 2015); tr100_prcent.tif;
  inegi\\iter_15csv20.zip; ESTABLECIMIENTO_SALUD_202609.xlsx; scripts 02, 03,
  04, 05, 07 y 12.

USO
---
  python 15_bootstrap_poblacion_aislada.py
  python 15_bootstrap_poblacion_aislada.py --B 2500 --semilla 20261124 --confianza 0.95

SALIDAS (carpeta salidas\\)
--------------------------
  15_bootstrap_poblacion_aislada.xlsx   hojas:
      Resumen       resultados de ambos métodos e interpretación
      Calculos      valores observados, EE, sesgo e IC percentil/básico con
                    fórmulas sobre Datos y Replicas; BCa (Python, en azul)
      Replicas      las 2,500 réplicas de cada indicador y método
      Histogramas   distribución de las réplicas (fórmulas + gráficos)
      Datos         una fila por localidad rural
      Python        resultados de Python para comparar
      Metodo        definiciones y fuentes
  15_resultados_bootstrap.csv
  15_replicas_bootstrap.csv
"""

import argparse
import importlib.util
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


s12 = _importar("s12", "12_chi_cuadrada_aislamiento_inundable.py")
RUTAS = dict(s12.RUTAS)
NOMBRE = "15_bootstrap_poblacion_aislada"

INDICADORES = [
    ("P1", "% población rural aislada", "aisl", "tot"),
    ("P2", "% población rural aislada fuera de zona inundable", "aisl_fuera", "tot"),
    ("P3", "% de los aislados que están fuera de zona inundable", "aisl_fuera", "aisl"),
]
METODOS = [("loc", "Por localidad"), ("mun", "Por municipio (conglomerados)")]


# ============================================================================
# 1. Bootstrap
# ============================================================================
def componentes(it):
    """Sumandos por localidad: población total, aislada y aislada fuera."""
    p = it["pob"].to_numpy(float)
    a = it["aislada"].to_numpy() == 1
    f = it["inundable"].to_numpy() == 0
    return {"tot": p, "aisl": p * a, "aisl_fuera": p * (a & f)}


def razones(sumas):
    return {k: 100 * sumas[num] / sumas[den] for k, _, num, den in INDICADORES}


def bootstrap(comp, grupos, B, rng):
    """Réplicas de P1-P3. grupos = None (por localidad) o etiquetas de
    conglomerado (por municipio)."""
    if grupos is None:
        unidades = dict(comp)
    else:
        codigos, inv = np.unique(grupos, return_inverse=True)
        unidades = {k: np.bincount(inv, weights=v, minlength=len(codigos)) for k, v in comp.items()}
    n = len(unidades["tot"])
    rep = {k: np.empty(B) for k, *_ in INDICADORES}
    bloque = 250
    for b0 in range(0, B, bloque):
        idx = rng.integers(0, n, size=(min(bloque, B - b0), n))
        r = razones({k: v[idx].sum(axis=1) for k, v in unidades.items()})
        for k in rep:
            rep[k][b0:b0 + len(idx)] = r[k]
    tot = {k: v.sum() for k, v in unidades.items()}
    jack = razones({k: tot[k] - unidades[k] for k in unidades})     # jackknife (aceleración BCa)
    return razones(tot), rep, jack, n


def intervalos(theta, rep, jack, conf):
    alfa = 1 - conf
    lo, hi = np.percentile(rep, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    z0 = stats.norm.ppf(np.clip(np.mean(rep < theta) + 0.5 * np.mean(rep == theta), 1e-10, 1 - 1e-10))
    d = jack.mean() - jack
    a = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5) if (d ** 2).sum() > 0 else 0.0
    zl, zh = stats.norm.ppf(alfa / 2), stats.norm.ppf(1 - alfa / 2)
    a1 = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    a2 = stats.norm.cdf(z0 + (z0 + zh) / (1 - a * (z0 + zh)))
    blo, bhi = np.percentile(rep, [100 * a1, 100 * a2])
    return {"observado": theta, "media_replicas": rep.mean(), "sesgo": rep.mean() - theta,
            "error_estandar": rep.std(ddof=1), "ic_percentil_inf": lo, "ic_percentil_sup": hi,
            "ic_basico_inf": 2 * theta - hi, "ic_basico_sup": 2 * theta - lo,
            "ic_bca_inf": blo, "ic_bca_sup": bhi, "z0_bca": z0, "aceleracion_bca": a,
            "alfa1_bca": a1, "alfa2_bca": a2, "min_replica": rep.min(), "max_replica": rep.max()}


# ============================================================================
# 2. Excel
# ============================================================================
def escribir_excel(ruta, it, reps, res, info, B, conf):
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference
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

    # ------------------------------------------------------------- Datos
    wd = wb.active
    wd.title = "Datos"
    wd.append(["CVEGEO", "CVE_MPIO", "Municipio", "Localidad", "Población 2020", "Inundable (1=sí)",
               "Aislada (1=sí)", "Aislada y fuera de zona inundable (1=sí)"])
    for r in it.itertuples():
        wd.append([r.CVEGEO, r.CVE_MPIO, r.NOM_MUN, r.NOM_LOC, int(r.pob), int(r.inundable), int(r.aislada),
                   int(r.aislada == 1 and r.inundable == 0)])
    for c in wd[1]:
        c.font, c.fill, c.alignment = f_b, hdr, Alignment(wrap_text=True)
    for row in wd.iter_rows(min_row=2):
        for c in row:
            c.font = f_n
        row[4].number_format = "#,##0"
    for i, w in enumerate([14, 10, 26, 34, 12, 11, 11, 16], 1):
        wd.column_dimensions[L(i)].width = w
    wd.freeze_panes = "A2"
    nd = len(it) + 1
    wd.auto_filter.ref = f"A1:H{nd}"
    P, A, AF = f"Datos!$E$2:$E${nd}", f"Datos!$G$2:$G${nd}", f"Datos!$H$2:$H${nd}"

    # ------------------------------------------------------------- Replicas
    wr = wb.create_sheet("Replicas")
    cols = [(m, k) for m, _ in METODOS for k, *_ in INDICADORES]
    wr.append(["Réplica"] + [f"{k} – {dict(METODOS)[m]}" for m, k in cols])
    for b in range(B):
        wr.append([b + 1] + [round(float(reps[m][k][b]), 6) for m, k in cols])
    for c in wr[1]:
        c.font, c.fill, c.alignment = f_b, hdr, Alignment(wrap_text=True, vertical="center")
    for row in wr.iter_rows(min_row=2):
        row[0].font = f_n
        for c in row[1:]:
            c.font, c.number_format = f_in, "0.0000"
    wr.column_dimensions["A"].width = 9
    for j in range(2, len(cols) + 2):
        wr.column_dimensions[L(j)].width = 18
    wr.freeze_panes = "B2"
    nr = B + 1
    col_rep = {mk: f"Replicas!${L(j)}$2:${L(j)}${nr}" for j, mk in enumerate(cols, 2)}
    wr.cell(nr + 2, 1, f"Réplicas generadas por el script 15 (NumPy, semilla {info['semilla']}).").font = f_i

    # ------------------------------------------------------------- Calculos
    wc = wb.create_sheet("Calculos")
    wc["A1"] = f"Bootstrap (B = {B:,d}): % de población rural aislada"
    wc["A1"].font = f_t
    wc["A2"] = ("Observados, sesgo, EE e IC percentil y básico son fórmulas sobre las hojas Datos y Replicas. "
                "El IC BCa necesita jackknife y se calcula en Python (texto azul).")
    wc["A2"].font = f_i
    wc["A4"], wc["B4"] = "Nivel de confianza", conf
    wc["A4"].font, wc["B4"].font, wc["B4"].number_format = f_b, f_in, "0%"
    wc["C4"] = "Dato de entrada (--confianza)"
    wc["C4"].font = f_i
    wc["A5"], wc["B5"] = "B (réplicas)", f"=COUNT({col_rep[cols[0]]})"
    wc["A5"].font, wc["B5"].font = f_b, f_n
    num_den = {"P1": (f"SUMPRODUCT({P},{A})", f"SUM({P})"),
               "P2": (f"SUMPRODUCT({P},{AF})", f"SUM({P})"),
               "P3": (f"SUMPRODUCT({P},{AF})", f"SUMPRODUCT({P},{A})")}
    enc = ["Indicador", "Método", "Observado (%)", "Media réplicas (%)", "Sesgo (pp)", "Error estándar (pp)",
           "IC percentil inf.", "IC percentil sup.", "IC básico inf.", "IC básico sup.", "IC BCa inf.",
           "IC BCa sup.", "z0 (BCa)", "a (BCa)"]
    fila = 7
    for j, t in enumerate(enc, 1):
        c = wc.cell(fila, j, t)
        c.font, c.fill, c.border, c.alignment = f_b, hdr, caja, Alignment(wrap_text=True, vertical="center")
    pos = {}
    for k, nombre, *_ in INDICADORES:
        for m, mn in METODOS:
            fila += 1
            pos[(m, k)] = fila
            rr = col_rep[(m, k)]
            num, den = num_den[k]
            vals = [f"{k}: {nombre}", mn, f"=100*{num}/{den}", f"=AVERAGE({rr})", f"=D{fila}-C{fila}",
                    f"=STDEV({rr})", f"=PERCENTILE({rr},(1-$B$4)/2)", f"=PERCENTILE({rr},1-(1-$B$4)/2)",
                    f"=2*C{fila}-H{fila}", f"=2*C{fila}-G{fila}",
                    float(res[(m, k)]["ic_bca_inf"]), float(res[(m, k)]["ic_bca_sup"]),
                    float(res[(m, k)]["z0_bca"]), float(res[(m, k)]["aceleracion_bca"])]
            for j, v in enumerate(vals, 1):
                c = wc.cell(fila, j, v)
                c.border = caja
                c.font = f_in if j >= 11 else f_n
                c.number_format = "0.0000" if j >= 13 else "0.00"
                c.alignment = Alignment(wrap_text=True, vertical="center")
            for j in (3, 6, 7, 8):
                wc.cell(fila, j).fill = clave
    wc.cell(fila + 2, 1, "pp = puntos porcentuales. IC básico = [2θ − P(1−α/2), 2θ − P(α/2)]. BCa: Efron (1987), "
            "aceleración por jackknife de la unidad de remuestreo.").font = f_i
    wc.cell(fila + 3, 1, f"Semilla aleatoria: {info['semilla']} (las réplicas son reproducibles con el script 15). "
            "PERCENTILE de Excel y numpy.percentile usan la misma interpolación lineal.").font = f_i
    for j, w in enumerate([34, 26, 11, 11, 10, 11, 10, 10, 10, 10, 10, 10, 9, 9], 1):
        wc.column_dimensions[L(j)].width = w
    wc.row_dimensions[7].height = 32

    # ------------------------------------------------------------- Histogramas
    wh = wb.create_sheet("Histogramas")
    wh["A1"] = "Distribución de las réplicas bootstrap (frecuencias con fórmulas COUNTIFS)"
    wh["A1"].font = f_t
    nb = 30
    col0 = 1
    for k, nombre, *_ in INDICADORES:
        for m, mn in METODOS:
            rr = col_rep[(m, k)]
            lo = float(np.floor(min(reps[m][k]) * 10) / 10)
            hi = float(np.ceil(max(reps[m][k]) * 10) / 10)
            w = (hi - lo) / nb
            wh.cell(3, col0, f"{k} – {mn}").font = f_b
            for j, t in enumerate(["Desde", "Hasta", "Marca", "Réplicas"]):
                c = wh.cell(4, col0 + j, t)
                c.font, c.fill = f_b, hdr
            for i in range(nb):
                r = 5 + i
                wh.cell(r, col0, round(lo, 6) if i == 0 else f"={L(col0 + 1)}{r - 1}")
                wh.cell(r, col0 + 1, f"={L(col0)}{r}+{w:.6f}")
                wh.cell(r, col0 + 2, f"=({L(col0)}{r}+{L(col0 + 1)}{r})/2")
                op = "<=" if i == nb - 1 else "<"
                wh.cell(r, col0 + 3, f'=COUNTIFS({rr},">="&{L(col0)}{r},{rr},"{op}"&{L(col0 + 1)}{r})')
                for j in range(4):
                    wh.cell(r, col0 + j).font = f_n
                    wh.cell(r, col0 + j).number_format = "0" if j == 3 else "0.00"
            wh.cell(5, col0).font = f_in
            wh.cell(5 + nb, col0 + 2, "Total").font = f_b
            wh.cell(5 + nb, col0 + 3, f"=SUM({L(col0 + 3)}5:{L(col0 + 3)}{4 + nb})").font = f_b
            ch = BarChart()
            ch.type, ch.gapWidth = "col", 5
            ch.title = f"{k} – {mn}"
            ch.y_axis.title, ch.x_axis.title = "Réplicas", "% (marca de clase)"
            ch.add_data(Reference(wh, min_col=col0 + 3, min_row=4, max_row=4 + nb), titles_from_data=True)
            ch.set_categories(Reference(wh, min_col=col0 + 2, min_row=5, max_row=4 + nb))
            ch.legend = None
            ch.roundedCorners = False
            ch.x_axis.delete = ch.y_axis.delete = False
            for t in (ch.title, ch.x_axis.title, ch.y_axis.title):
                t.overlay = False
            ch.width, ch.height = 12, 7
            wh.add_chart(ch, f"{L(col0)}{7 + nb}")
            for j, wd_ in enumerate([9, 9, 9, 9, 2]):
                wh.column_dimensions[L(col0 + j)].width = wd_
            col0 += 6

    # ------------------------------------------------------------- Resumen
    ws = wb.create_sheet("Resumen", 0)
    ws["A1"] = f"Bootstrap (B = {B:,d}) de la población rural aislada – Estado de México"
    ws["A1"].font = f_t
    ws["A3"], ws["B3"] = "Escenario", info["escenario"]
    ws["A3"].font, ws["B3"].font = f_b, f_n
    ws["B3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("B3:G3")
    ws.row_dimensions[3].height = 44
    enc = ["Indicador", "Método", "Observado (%)", "Error estándar (pp)", "IC 95 % percentil", "IC 95 % BCa",
           "Sesgo (pp)"]
    for j, t in enumerate(enc, 1):
        c = ws.cell(5, j, t)
        c.font, c.fill, c.alignment = f_b, hdr, Alignment(wrap_text=True)
    r = 5
    for k, nombre, *_ in INDICADORES:
        for m, mn in METODOS:
            r += 1
            fc = pos[(m, k)]
            vals = [f"{k}: {nombre}", mn, f"=Calculos!C{fc}", f"=Calculos!F{fc}",
                    f'=TEXT(Calculos!G{fc},"0.00")&" – "&TEXT(Calculos!H{fc},"0.00")',
                    f'=TEXT(Calculos!K{fc},"0.00")&" – "&TEXT(Calculos!L{fc},"0.00")', f"=Calculos!E{fc}"]
            for j, v in enumerate(vals, 1):
                c = ws.cell(r, j, v)
                c.font = f_link if j >= 3 else f_n
                c.number_format = "0.00" if j in (3, 4) else "0.000" if j == 7 else "General"
                c.alignment = Alignment(wrap_text=True, vertical="center")
    fi = r + 2
    ws.cell(fi, 1, "Interpretación").font = f_b
    for k, t in enumerate(info["interpretacion"], fi + 1):
        c = ws.cell(k, 1, t)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=k, start_column=1, end_row=k, end_column=7)
        ws.row_dimensions[k].height = 44
    ws.cell(fi + len(info["interpretacion"]) + 2, 1,
            "Verde = vínculo a la hoja Calculos. pp = puntos porcentuales.").font = f_i
    for j, w in enumerate([44, 26, 12, 12, 16, 16, 10], 1):
        ws.column_dimensions[L(j)].width = w

    # ------------------------------------------------------------- Python
    wp = wb.create_sheet("Python")
    claves = list(next(iter(res.values())).keys())
    wp.append(["Estadístico"] + [f"{k} – {dict(METODOS)[m]}" for m, k in cols])
    for kk in claves:
        wp.append([kk] + [float(res[mk][kk]) for mk in cols])
    wp.append(["unidades_remuestreadas"] + [info["n_unidades"][m] for m, k in cols])
    for c in wp[1]:
        c.font, c.fill, c.alignment = f_b, hdr, Alignment(wrap_text=True)
    for row in wp.iter_rows(min_row=2):
        for c in row:
            c.font = f_n
            if isinstance(c.value, float):
                c.number_format = "0.0000"
    wp.column_dimensions["A"].width = 24
    for j in range(2, len(cols) + 2):
        wp.column_dimensions[L(j)].width = 18

    # ------------------------------------------------------------- Método
    wm = wb.create_sheet("Metodo")
    txt = [
        ("Indicadores", "P1 = Σpob aisladas / Σpob rural × 100. P2 = Σpob aisladas fuera de zona inundable / "
                        "Σpob rural × 100. P3 = Σpob aisladas fuera / Σpob aisladas × 100 (otra lectura de P2)."),
        ("Escenario", "Tramos con > 50 % de su longitud inundable (TR100; puntos cada 15 m) cerrados; hospitales "
                      "públicos operativos (franja de 20 km); aislada = sin ruta. Localidades rurales < 2,500 hab."),
        ("Bootstrap", f"B = {B:,d} réplicas, semilla {info['semilla']}. Por localidad: se remuestrean las 4,215 "
                      "localidades con reemplazo. Por municipio: se remuestrean municipios completos "
                      "(conglomerados), lo que respeta la dependencia espacial entre localidades vecinas."),
        ("Intervalos", "Percentil: percentiles α/2 y 1−α/2 de las réplicas. Básico: 2θ − percentiles. BCa: "
                       "corrección de sesgo z0 y aceleración a por jackknife (Efron, 1987)."),
        ("Alcance", "Mide la incertidumbre muestral de las localidades; no la del modelo (ráster, velocidades, "
                    "umbral), que se explora con los escenarios de los scripts 07 y 10."),
        ("Fuentes", "rnc2025Edomex.gpkg; CENAPRED_IPI_2015_Inund_Munici.zip; tr100_prcent.tif; iter_15csv20.zip; "
                    "ESTABLECIMIENTO_SALUD_202609.xlsx; scripts 02, 03, 04, 05, 07, 12 y 15."),
    ]
    for i, (k, v) in enumerate(txt, 1):
        wm.cell(i, 1, k).font = f_b
        c = wm.cell(i, 2, v)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        wm.cell(i, 1).alignment = Alignment(vertical="top")
        wm.row_dimensions[i].height = 45
    wm.column_dimensions["A"].width = 14
    wm.column_dimensions["B"].width = 105
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
    ap.add_argument("--B", type=int, default=2500, help="número de réplicas bootstrap")
    ap.add_argument("--semilla", type=int, default=20261124)
    ap.add_argument("--confianza", type=float, default=0.95)
    a = ap.parse_args()
    for k in RUTAS:
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()
    SALIDAS.mkdir(exist_ok=True)

    print("1/3 Aislamiento y zona inundable de cada localidad rural...")
    it, n_hosp, n_cerr, n_tr = s12.variables_localidades(a)
    it["CVE_MPIO"] = it["ENTIDAD"] + it["MUN"]

    print(f"2/3 Bootstrap B = {a.B:,d} (por localidad y por municipio)...")
    rng = np.random.default_rng(a.semilla)
    comp = componentes(it)
    reps, res, n_unid = {}, {}, {}
    for m, grupos in (("loc", None), ("mun", it["CVE_MPIO"].to_numpy())):
        obs, rep, jack, n = bootstrap(comp, grupos, a.B, rng)
        reps[m], n_unid[m] = rep, n
        for k, *_ in INDICADORES:
            res[(m, k)] = intervalos(obs[k], rep[k], jack[k], a.confianza)

    tot = {k: v.sum() for k, v in comp.items()}
    r1, r2, r3 = res[("loc", "P1")], res[("loc", "P2")], res[("loc", "P3")]
    m1, m2, m3 = res[("mun", "P1")], res[("mun", "P2")], res[("mun", "P3")]
    if m3["ic_bca_inf"] > 50:
        cierre_p3 = "Aun con el método más prudente el límite inferior supera el 50 %: la mayoría de los aislados lo está por efecto indirecto."
    else:
        cierre_p3 = ("El intervalo por municipio incluye el 50 %: no puede afirmarse con 95 % de confianza que la "
                     "mayoría de los aislados esté fuera de zona inundable.")
    interp = [
        (f"P1: el {r1['observado']:.2f} % de la población rural queda aislada ({int(tot['aisl']):,d} de "
         f"{int(tot['tot']):,d} hab.). IC 95 % BCa por localidad: {r1['ic_bca_inf']:.2f} – {r1['ic_bca_sup']:.2f} % "
         f"(EE {r1['error_estandar']:.2f} pp); por municipio: {m1['ic_bca_inf']:.2f} – {m1['ic_bca_sup']:.2f} % "
         f"(EE {m1['error_estandar']:.2f} pp)."),
        (f"P2: el {r2['observado']:.2f} % de la población rural queda aislada sin estar en zona inundable "
         f"({int(tot['aisl_fuera']):,d} hab.). IC 95 % BCa por localidad: {r2['ic_bca_inf']:.2f} – "
         f"{r2['ic_bca_sup']:.2f} %; por municipio: {m2['ic_bca_inf']:.2f} – {m2['ic_bca_sup']:.2f} %."),
        (f"P3: de los aislados, el {r3['observado']:.1f} % está fuera de zona inundable (IC 95 % BCa por "
         f"localidad {r3['ic_bca_inf']:.1f} – {r3['ic_bca_sup']:.1f} %; por municipio "
         f"{m3['ic_bca_inf']:.1f} – {m3['ic_bca_sup']:.1f} %). " + cierre_p3),
        ("El intervalo por municipio es más amplio porque considera que las localidades de un mismo municipio se "
         "aíslan juntas (no son independientes); es la estimación más prudente para reportar."),
    ]
    info = {"escenario": (f"Localidades rurales (< 2,500 hab.): {len(it):,d}. Tramos con > {a.umbral:g} % de su "
                          f"longitud inundable (TR100) cerrados ({n_cerr:,d}); {n_hosp} hospitales públicos operativos "
                          f"(franja de 20 km); puntos de muestreo cada {a.intervalo:g} m."),
            "interpretacion": interp, "semilla": a.semilla, "n_unidades": n_unid}

    print("3/3 Escribiendo Excel y CSV...")
    ruta_x = SALIDAS / f"{NOMBRE}.xlsx"
    escribir_excel(ruta_x, it, reps, res, info, a.B, a.confianza)
    pd.DataFrame({f"{k}_{m}": v for (m, k), v in res.items()}).to_csv(
        SALIDAS / "15_resultados_bootstrap.csv", encoding="utf-8-sig")
    pd.DataFrame({f"{k}_{m}": reps[m][k] for m, _ in METODOS for k, *_ in INDICADORES}).round(6).to_csv(
        SALIDAS / "15_replicas_bootstrap.csv", index_label="replica", encoding="utf-8-sig")

    print("\n" + "=" * 112)
    print(f" BOOTSTRAP (B = {a.B:,d}, semilla {a.semilla}) - población rural aislada, Estado de México")
    print("=" * 112)
    print(f"  {info['escenario']}")
    print(f"  Unidades remuestreadas: {n_unid['loc']:,d} localidades / {n_unid['mun']:,d} municipios")
    print(f"\n  {'Indicador':52s}{'Método':16s}{'Obs. %':>8s}{'EE pp':>7s}{'IC 95 % percentil':>19s}"
          f"{'IC 95 % BCa':>17s}")
    for k, nombre, *_ in INDICADORES:
        for m, mn in METODOS:
            r = res[(m, k)]
            print(f"  {(k + ': ' + nombre)[:51]:52s}{mn[:15]:16s}{r['observado']:>8.2f}{r['error_estandar']:>7.2f}"
                  f"{r['ic_percentil_inf']:>9.2f} – {r['ic_percentil_sup']:<7.2f}{r['ic_bca_inf']:>7.2f} – "
                  f"{r['ic_bca_sup']:<7.2f}")
    print("\n  Interpretación:")
    for t in interp:
        print("   - " + t)
    print(f"\n  Excel: {ruta_x}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
