# -*- coding: utf-8 -*-
"""
Script 16 - Aislamiento por Inundaciones: regresión logística agrupada por municipio
===================================================================================

(Sustituye al borrador anterior "16_regresion_aislamiento_municipal.py", una
regresión lineal con el municipio como unidad. Por indicación del usuario,
2026-11-24, se cambia a regresión logística con la LOCALIDAD como unidad y
errores agrupados por municipio.)

OBJETIVO
--------
Estimar cómo cambia la probabilidad de que una localidad rural quede AISLADA
de los hospitales públicos operativos según tres factores de exposición, y
qué tanto pesa cada uno controlando por los otros dos:

  Y  = 1 si la localidad rural queda aislada (sin ruta a ningún hospital
       público operativo) al cerrar los tramos con MÁS DEL 50 % de su
       longitud en zona inundable TR100 (escenario usado en los scripts 07 y
       09 a 15); 0 si no.
  X1 = 1 si la localidad rural está en zona inundable (su punto del ITER cae
       en píxel TR100 = 0).
  X2 = 1 si el hospital público operativo más cercano a la localidad (en la
       red completa, sin inundación) está en zona inundable.
  X3 = 1 si más del 75 % de la red vial del MUNICIPIO de la localidad (en km)
       está formada por TRAMOS con más del 75 % de su longitud inundada
       (corrección del usuario, 2026-11-24: "red en tramos con más del 75 %
       inundado").

  Modelo:  logit P(Y=1) = b0 + b1·X1 + b2·X2 + b3·X3

  "Agrupada por municipio": las localidades de un mismo municipio comparten
  caminos y tienden a aislarse juntas (no son independientes; ver scripts 12
  a 15). Por eso los errores estándar se calculan ROBUSTOS POR
  CONGLOMERADO (municipio), que es la forma estándar de tratar observaciones
  agrupadas en una regresión logística. Se reportan también los errores
  ingenuos (sin agrupar) para mostrar la diferencia.

DEFINICIÓN DETALLADA DE LAS VARIABLES
-------------------------------------
  Escenario común: RNC 2025, tramos válidos de la componente principal,
  puntos de muestreo cada 15 m (script 03); ráster tr100_prcent.tif
  (0 = inundable); hospitales públicos operativos ('DE HOSPITALIZACIÓN',
  'EN OPERACION', institución distinta de 'SERVICIOS MEDICOS PRIVADOS') en la
  franja de 20 km; localidades rurales < 2,500 hab. (ITER 2020).

  X2: en la red completa se calculan dos tiempos por localidad: al hospital
      inundable más cercano y al hospital seco más cercano (tiempo = longitud
      / VELOCIDAD RNC; conexión a la red a 5 km/h). El hospital más cercano es
      inundable si el primer tiempo es menor. Hospital inundable = su punto en
      píxel TR100 = 0. (Criterio tomado del borrador anterior.)
  X3: "red en tramos con más del 75 % inundado".
      a. % inundado de cada TRAMO = puntos de muestreo (cada <= 15 m) en
         píxel inundable / puntos del tramo (mismo cálculo de los scripts 07
         y 11). Tramo "muy inundado" si ese % > 75 (--umbral-tramo).
      b. Cada punto de muestreo vale L/n metros y se asigna al municipio
         donde cae; lleva la marca de su tramo. Así, un tramo que cruza un
         límite reparte su longitud entre municipios.
      c. % de la red municipal en tramos muy inundados =
         km en tramos con % inundado > 75 / km de red del municipio × 100.
      d. X3 = 1 si ese % > 75 (--umbral-x3).
      Como sensibilidad se ajusta el modelo con el % continuo del paso c (por
      cada 10 puntos porcentuales).
  Municipio de la localidad = su clave del ITER (ENTIDAD + MUN).

ESTIMACIÓN
----------
  1. Máxima verosimilitud por Newton-Raphson (IRLS):
       p = 1/(1+exp(−Xb));  b ← b + (X'WX)⁻¹ X'(y − p),  W = diag(p(1−p)).
  2. Errores estándar:
       - ingenuos (modelo):        V = (X'WX)⁻¹
       - robustos por municipio:   V = (X'WX)⁻¹ [Σ_g s_g s_g'] (X'WX)⁻¹ · c,
         s_g = Σ_{i∈g} x_i (y_i − p_i),  c = G/(G−1) · (n−1)/(n−k)
         (corrección CR1, la misma de Stata 'vce(cluster)').
  3. Para cada coeficiente: z = b/EE, p bilateral = 2·Φ(−|z|),
     razón de momios OR = exp(b) e IC 95 % = exp(b ± 1.96·EE).
  4. Ajuste global:
       - log-verosimilitud LL y LL0 del modelo nulo (solo constante),
       - razón de verosimilitudes LR = 2(LL − LL0), gl = 3,
       - prueba de Wald conjunta con la matriz robusta (gl = 3),
       - pseudo R² de McFadden = 1 − LL/LL0,
       - AIC = 2k − 2LL,
       - AUC (área bajo la curva ROC) = probabilidad de que una aislada
         tenga mayor probabilidad predicha que una no aislada,
       - tabla de clasificación con punto de corte 0.5,
       - Hosmer-Lemeshow con los grupos distintos de probabilidad predicha
         (con tres predictores binarios hay como máximo 8 perfiles; se
         usan los perfiles en lugar de deciles).
  5. Diagnóstico: correlación entre predictores y tabla de cada perfil
     (combinación X1, X2, X3) con su % observado y predicho de aislamiento.

CRITERIOS QUE CONVIENE DEBATIR
------------------------------
  - Los errores robustos por municipio corrigen la inferencia, pero los
    coeficientes son los de un modelo "promedio poblacional". Un modelo
    mixto (efecto aleatorio por municipio) o GEE daría coeficientes
    condicionales; no se usa porque statsmodels no está instalado en el
    entorno (red desactivada) y el modelo robusto es el estándar para este
    propósito.
  - X3 es una característica del municipio, no de la localidad: su efecto se
    estima con solo los municipios que superan el 75 % (pocos); su
    intervalo será amplio.
  - Asociación, no causalidad: el aislamiento lo produce el modelo de red;
    la regresión resume qué factores de exposición lo acompañan.

FUENTES
-------
  rnc2025Edomex.gpkg; CENAPRED.zip (IPI 2015, municipios); tr100_prcent.tif;
  inegi\\iter_15csv20.zip; ESTABLECIMIENTO_SALUD_202609.xlsx; scripts 02, 03,
  04, 05, 07, 08, 11, 12 y 14.

USO
---
  python 16_regresion_logistica_aislamiento.py
  python 16_regresion_logistica_aislamiento.py --umbral 50 --umbral-tramo 75 --umbral-x3 75

REQUISITOS
----------
  numpy, pandas, networkx, scipy, matplotlib, Pillow, openpyxl.

SALIDAS (carpeta salidas\\)
--------------------------
  16_regresion_logistica_aislamiento.xlsx   hojas:
      Resumen        coeficientes, OR, ajuste e interpretación
      Modelo         coeficientes y EE (Python, azul) y todo lo derivado con
                     fórmulas: OR, IC, z, p, LL, LR, McFadden, AIC,
                     clasificación, perfiles
      Datos          una fila por localidad rural: Y, X1, X2, X3, municipio y
                     probabilidad predicha (fórmula)
      Municipios     km de red, km en tramos > 75 % inundados y X3 por municipio
      Sensibilidad   modelo con X3 continuo y modelo sin agrupar
      Python         resultados de Python para comparar
      Metodo         definiciones y fuentes
  16_coeficientes_logit.csv
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
s07 = _importar("s07", "07_aislamiento_por_umbral_inundacion.py")
s08 = _importar("s08", "08_mapa_tramos_inundables_hospitales.py")
s11 = _importar("s11", "11_tramos_criticos_cierre_individual.py")
s14 = _importar("s14", "14_spearman_red_inundable_aislamiento.py")
RUTAS = dict(s07.RUTAS)
NOMBRE = "16_regresion_logistica_aislamiento"
VARS = ["X1_localidad_inundable", "X2_hospital_cercano_inundable", "X3_red_municipal_gt75"]
ETIQ = {"const": "Constante",
        "X1_localidad_inundable": "X1: localidad rural en zona inundable",
        "X2_hospital_cercano_inundable": "X2: hospital público más cercano en zona inundable",
        "X3_red_municipal_gt75": "X3: > 75 % de la red municipal en tramos > 75 % inundados",
        "X3_pct_red_tramos_x10": "X3c: % red municipal en tramos > 75 % inundados (por cada 10 pp)"}


# ============================================================================
# 1. Variables por localidad
# ============================================================================
def variables(a):
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
    it["CVE_MPIO"] = it["ENTIDAD"] + it["MUN"]
    lx, ly = s05.lcc(it["lon"], it["lat"])
    it["X1_localidad_inundable"] = R.valores(lx, ly, inund).astype(int)
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
    h["nodo"], h["con_min"] = asignar(hx, hy)

    nl, cn = it["nodo"].to_numpy(), it["con_min"].to_numpy()
    G0 = s07.grafo(tr, np.ones(len(tr), bool))
    Ge = s07.grafo(tr, ~cerrados)
    t_inund = s07.tiempos(G0, h.loc[h.inundable, "nodo"].to_numpy(), h.loc[h.inundable, "con_min"].to_numpy(), nl, cn)
    t_seco = s07.tiempos(G0, h.loc[~h.inundable, "nodo"].to_numpy(), h.loc[~h.inundable, "con_min"].to_numpy(), nl, cn)
    t_esc = s07.tiempos(Ge, h["nodo"].to_numpy(), h["con_min"].to_numpy(), nl, cn)
    it["min_base"] = np.round(np.minimum(t_inund, t_seco), 2)
    it["X2_hospital_cercano_inundable"] = (t_inund < t_seco).astype(int)
    it["Y_aislada"] = (~np.isfinite(t_esc)).astype(int)
    info = {"hosp": len(h), "hosp_inund": int(h.inundable.sum()), "cerrados": int(cerrados.sum()),
            "tramos": len(tr)}
    return it, info


def red_tramos_por_municipio(a, muns, umbral_tramo):
    """Km de red, km en píxel inundable y km en tramos con % inundado > umbral,
    por municipio. Cada punto de muestreo (cada <= 15 m) vale L/n metros, se
    asigna al municipio donde cae y lleva el % inundado de su tramo."""
    R = s05.Raster(a.tr100)
    inund = R.a == s05.VALOR_INUNDABLE
    etiq = s11.etiquetas_municipio(R, muns)
    con = sqlite3.connect(f"file:{a.gpkg.as_posix()}?mode=ro", uri=True)
    filas = con.execute(f'SELECT ESTATUS, CONDICION, CIRCULA, UNION_INI, UNION_FIN, geom '
                        f'FROM "{s07.CAPA_RED}"').fetchall()
    validos = [f for f in filas if not s07.m03.excluido(f[0], f[1], f[2])]
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from((f[3], f[4]) for f in validos)
    comp = max(nx.connected_components(G), key=len)
    lon_l, lat_l, w_l, t_l = [], [], [], []
    i = 0
    for f in validos:
        if f[3] not in comp or f[3] == f[4]:
            continue
        L, pts = s07.red02.puntos_a_lo_largo(s07.red02.gpkg_a_lineas(f[5]), a.intervalo)
        lon_l.append(pts[:, 0]); lat_l.append(pts[:, 1])
        w_l.append(np.full(len(pts), L / len(pts))); t_l.append(np.full(len(pts), i))
        i += 1
    x, y = s05.lcc(np.concatenate(lon_l), np.concatenate(lat_l))
    w, t = np.concatenate(w_l), np.concatenate(t_l)
    ii = R.valores(x, y, inund).astype(float)
    pct_tramo = 100 * np.bincount(t, weights=ii, minlength=i) / np.bincount(t, minlength=i)
    muy = pct_tramo[t] > umbral_tramo
    lab = R.valores(x, y, etiq)
    m = len(muns) + 1
    km = np.bincount(lab, weights=w, minlength=m) / 1000
    km_inund = np.bincount(lab, weights=w * ii, minlength=m) / 1000
    km_tr = np.bincount(lab, weights=w * muy, minlength=m) / 1000
    return km[1:], km_inund[1:], km_tr[1:]


# ============================================================================
# 2. Regresión logística con errores robustos por conglomerado
# ============================================================================
def logit(y, X, grupos=None, iters=50):
    n, k = X.shape
    b = np.zeros(k)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        paso = np.linalg.solve(H, X.T @ (y - p))
        b += paso
        if np.max(np.abs(paso)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ b))
    H = X.T @ (X * (p * (1 - p))[:, None])
    Hinv = np.linalg.inv(H)
    V = Hinv.copy()
    G = None
    if grupos is not None:
        codigos, inv = np.unique(grupos, return_inverse=True)
        G = len(codigos)
        S = np.zeros((G, k))
        np.add.at(S, inv, X * (y - p)[:, None])
        c = G / (G - 1) * (n - 1) / (n - k)
        V = Hinv @ (S.T @ S) @ Hinv * c
    ee = np.sqrt(np.diag(V))
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    p0 = y.mean()
    ll0 = float(n * (p0 * math.log(p0) + (1 - p0) * math.log(1 - p0)))
    wald = float(b[1:] @ np.linalg.solve(V[1:, 1:], b[1:]))
    z = b / ee
    return {"b": b, "ee": ee, "ee_ingenuo": np.sqrt(np.diag(Hinv)), "V": V, "z": z,
            "p": 2 * stats.norm.sf(np.abs(z)), "p_hat": p, "ll": ll, "ll0": ll0,
            "lr": 2 * (ll - ll0), "p_lr": float(stats.chi2.sf(2 * (ll - ll0), k - 1)),
            "wald": wald, "p_wald": float(stats.chi2.sf(wald, k - 1)),
            "mcfadden": 1 - ll / ll0, "aic": 2 * k - 2 * ll, "n": n, "k": k, "G": G}


def auc(y, p):
    r = stats.rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def hosmer_lemeshow(y, p):
    df = pd.DataFrame({"y": y, "p": np.round(p, 10)})
    g = df.groupby("p").agg(n=("y", "size"), obs=("y", "sum")).reset_index()
    g["esp"] = g.n * g.p
    hl = float((((g.obs - g.esp) ** 2) / (g.esp * (1 - g.p))).sum())
    gl = len(g) - 2
    return hl, gl, float(stats.chi2.sf(hl, gl)) if gl > 0 else np.nan


# ============================================================================
# 3. Excel
# ============================================================================
def escribir_excel(ruta, it, mun, res, res_c, res_n, extra, info):
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

    def cabecera(ws, fila, textos, col=1):
        for j, t in enumerate(textos):
            c = ws.cell(fila, col + j, t)
            c.font, c.fill, c.border = f_b, hdr, caja
            c.alignment = Alignment(wrap_text=True, vertical="center")

    # ------------------------------------------------------------- Municipios
    wm = wb.active
    wm.title = "Municipios"
    cabecera(wm, 1, ["CVE_MPIO", "Municipio", "Km de red", "Km en tramos > 75 % inundados",
                     "% de la red en tramos > 75 % inundados", "X3 (1 = % > umbral)", "Localidades rurales",
                     "Aisladas", "Km en píxel inundable (referencia)"])
    for i, r in enumerate(mun.itertuples(), 2):
        wm.cell(i, 1, r.CVE_MPIO)
        wm.cell(i, 2, r.municipio)
        wm.cell(i, 3, round(r.km, 3)).font = f_in
        wm.cell(i, 4, round(r.km_tramos_gt, 3)).font = f_in
        wm.cell(i, 9, round(r.km_inund, 3)).font = f_in
        wm.cell(i, 9).number_format = "#,##0.0"
        wm.cell(i, 5, f"=D{i}/C{i}*100")
        wm.cell(i, 6, f"=IF(E{i}>Modelo!$C$4,1,0)")
        wm.cell(i, 7, f"=COUNTIF(Datos!$B:$B,A{i})")
        wm.cell(i, 8, f"=COUNTIFS(Datos!$B:$B,A{i},Datos!$F:$F,1)")
        for j in (1, 2, 5, 6, 7, 8):
            wm.cell(i, j).font = f_n
        for j, fm in ((3, "#,##0.0"), (4, "#,##0.0"), (5, "0.00"), (6, "0"), (7, "#,##0"), (8, "#,##0")):
            wm.cell(i, j).number_format = fm
    nm = len(mun) + 1
    wm.cell(nm + 2, 1, "Km en azul: calculados por el script 16 (puntos cada 15 m sobre la RNC y ráster TR100; "
            "tramo con % inundado > umbral de tramo, Modelo!C5). El resto son fórmulas; X3 usa el umbral de la "
            "hoja Modelo (celda C4).").font = f_i
    for j, w in enumerate([10, 28, 12, 14, 14, 11, 11, 10, 14], 1):
        wm.column_dimensions[L(j)].width = w
    wm.freeze_panes = "A2"

    # ------------------------------------------------------------- Datos
    wd = wb.create_sheet("Datos", 0)
    cabecera(wd, 1, ["CVEGEO", "CVE_MPIO", "Municipio", "Localidad", "Población 2020",
                     "Y: aislada (1=sí)", "X1: localidad inundable", "X2: hospital más cercano inundable",
                     "X3: red municipal > umbral", "Min. base al hospital", "Logit (b'x)",
                     "Probabilidad predicha", "Clasificada aislada (p ≥ 0.5)"])
    for i, r in enumerate(it.itertuples(), 2):
        fila_m = int(mun.index[mun.CVE_MPIO == r.CVE_MPIO][0]) + 2
        vals = [r.CVEGEO, r.CVE_MPIO, r.NOM_MUN, r.NOM_LOC, int(r.pob), int(r.Y_aislada),
                int(r.X1_localidad_inundable), int(r.X2_hospital_cercano_inundable),
                f"=Municipios!F{fila_m}", float(r.min_base),
                f"=Modelo!$C$8+Modelo!$C$9*G{i}+Modelo!$C$10*H{i}+Modelo!$C$11*I{i}",
                f"=1/(1+EXP(-K{i}))", f"=IF(L{i}>=0.5,1,0)"]
        for j, v in enumerate(vals, 1):
            c = wd.cell(i, j, v)
            c.font = f_n
        wd.cell(i, 5).number_format = "#,##0"
        wd.cell(i, 10).number_format = "0.00"
        wd.cell(i, 11).number_format = "0.0000"
        wd.cell(i, 12).number_format = "0.0000"
    nd = len(it) + 1
    for j, w in enumerate([14, 10, 24, 30, 11, 10, 11, 13, 12, 11, 10, 11, 12], 1):
        wd.column_dimensions[L(j)].width = w
    wd.freeze_panes = "A2"
    wd.auto_filter.ref = f"A1:M{nd}"
    Y, X1, X2, X3, PH, CL = (f"Datos!${c}$2:${c}${nd}" for c in "FGHILM")

    # ------------------------------------------------------------- Modelo
    ws = wb.create_sheet("Modelo")
    ws["A1"] = "Regresión logística: P(localidad rural aislada) con errores robustos por municipio"
    ws["A1"].font = f_t
    ws["A2"] = ("Coeficientes y errores estándar (azul) vienen de Python (máxima verosimilitud; EE robustos por "
                "conglomerado CR1). Todo lo demás son fórmulas sobre las hojas Datos y Municipios.")
    ws["A2"].font = f_i
    ws["A4"], ws["C4"] = "Umbral de X3 (% de la red municipal en tramos muy inundados)", info["umbral_x3"]
    ws["A5"], ws["C5"] = "Umbral de tramo muy inundado (% de su longitud)", info["umbral_tramo"]
    ws["A6"], ws["C6"] = "Municipios (conglomerados) con localidades rurales", res["G"]
    ws["A4"].font = ws["A5"].font = ws["A6"].font = f_b
    ws["C4"].font = ws["C5"].font = ws["C6"].font = f_in
    ws["D5"] = "Informativo: cambiarlo exige volver a correr el script (--umbral-tramo)."
    ws["D5"].font = f_i
    cabecera(ws, 7, ["Variable", "", "Coeficiente b", "EE robusto (municipio)", "z", "Valor p",
                     "Razón de momios OR", "IC 95 % OR inf.", "IC 95 % OR sup.", "EE ingenuo (sin agrupar)",
                     "Valor p ingenuo"])
    nombres = ["const"] + VARS
    for i, v in enumerate(nombres):
        r = 8 + i
        ws.cell(r, 1, ETIQ[v]).font = f_n
        ws.cell(r, 3, float(res["b"][i])).font = f_in
        ws.cell(r, 4, float(res["ee"][i])).font = f_in
        ws.cell(r, 5, f"=C{r}/D{r}")
        ws.cell(r, 6, f"=2*NORMSDIST(-ABS(E{r}))")
        ws.cell(r, 7, f"=EXP(C{r})")
        ws.cell(r, 8, f"=EXP(C{r}-1.96*D{r})")
        ws.cell(r, 9, f"=EXP(C{r}+1.96*D{r})")
        ws.cell(r, 10, float(res["ee_ingenuo"][i])).font = f_in
        ws.cell(r, 11, f"=2*NORMSDIST(-ABS(C{r}/J{r}))")
        for j, fm in ((3, "0.0000"), (4, "0.0000"), (5, "0.000"), (6, "0.0000E+00"), (7, "0.000"),
                      (8, "0.000"), (9, "0.000"), (10, "0.0000"), (11, "0.0000E+00")):
            c = ws.cell(r, j)
            c.number_format, c.border = fm, caja
            if c.font.color is None or c.font.color.rgb != "000000FF":
                c.font = c.font if c.font.color and c.font.color.rgb == "000000FF" else f_n
        if i > 0:
            ws.cell(r, 7).fill = ws.cell(r, 6).fill = clave
    ws.cell(8, 7).value = f"=EXP(C8)"
    ws.cell(8, 7).number_format = "0.000"
    ws.cell(13, 1, "OR de la constante = momios de aislamiento cuando X1 = X2 = X3 = 0.").font = f_i

    fa = 15
    ws.cell(fa, 1, "Ajuste global").font = f_b
    ajuste = [
        ("n (localidades rurales)", f"=COUNT({Y})", "#,##0", ""),
        ("Localidades aisladas", f"=SUM({Y})", "#,##0", ""),
        ("Log-verosimilitud del modelo (LL)", f"=SUMPRODUCT({Y},LN({PH}))+SUMPRODUCT(1-{Y},LN(1-{PH}))",
         "0.000", "Σ y·ln p + (1−y)·ln(1−p)"),
        ("Log-verosimilitud nula (LL0)", "=C{ny}*LN(C{ny}/C{n})+(C{n}-C{ny})*LN(1-C{ny}/C{n})", "0.000",
         "Solo constante: p = proporción de aisladas"),
        ("Razón de verosimilitudes LR = 2(LL − LL0)", "=2*(C{ll}-C{ll0})", "0.000", "gl = 3"),
        ("Valor p (LR)", "=CHIDIST(C{lr},3)", "0.0000E+00", ""),
        ("Wald conjunta (EE robustos)", float(res["wald"]), "0.000", "Python: b' V⁻¹ b, gl = 3"),
        ("Valor p (Wald robusta)", "=CHIDIST(C{wald},3)", "0.0000E+00", ""),
        ("Pseudo R² de McFadden", "=1-C{ll}/C{ll0}", "0.0000", "0.2–0.4 se considera buen ajuste"),
        ("AIC", "=2*4-2*C{ll}", "0.00", "2k − 2LL, k = 4"),
        ("AUC (área bajo la curva ROC)", float(extra["auc"]), "0.0000",
         "Python; 0.7–0.8 aceptable, 0.8–0.9 excelente"),
        ("Hosmer-Lemeshow χ² (perfiles)", float(extra["hl"]), "0.000", f"Python; gl = {extra['hl_gl']}"),
        ("Valor p (Hosmer-Lemeshow)", float(extra["hl_p"]) if np.isfinite(extra["hl_p"]) else "N/D",
         "0.0000", "p > 0.05: no hay evidencia de mal ajuste"),
    ]
    cl_ = ["n", "ny", "ll", "ll0", "lr", "plr", "wald", "pwald", "mcf", "aic", "auc", "hl", "hlp"]
    pos = {k: fa + 1 + i for i, k in enumerate(cl_)}
    for k, (et, val, fm, nota) in zip(cl_, ajuste):
        r = pos[k]
        ws.cell(r, 1, et).font = f_n
        v = val.format(**pos) if isinstance(val, str) and val.startswith("=") else val
        c = ws.cell(r, 3, v)
        c.number_format, c.border = fm, caja
        c.font = f_in if not (isinstance(val, str) and val.startswith("=")) else f_n
        ws.cell(r, 4, nota).font = f_i
    for k in ("lr", "plr", "mcf", "auc"):
        ws.cell(pos[k], 3).fill = clave

    fc = pos["hlp"] + 2
    ws.cell(fc, 1, "Tabla de clasificación (punto de corte 0.5)").font = f_b
    cabecera(ws, fc + 1, ["", "", "Predicha aislada", "Predicha no aislada", "Total"])
    for i, (et, yv) in enumerate((("Observada aislada", 1), ("Observada no aislada", 0))):
        r = fc + 2 + i
        ws.cell(r, 1, et).font = f_b
        ws.cell(r, 3, f"=COUNTIFS({Y},{yv},{CL},1)")
        ws.cell(r, 4, f"=COUNTIFS({Y},{yv},{CL},0)")
        ws.cell(r, 5, f"=C{r}+D{r}")
    r = fc + 4
    ws.cell(r, 1, "Exactitud global").font = f_n
    ws.cell(r, 3, f"=(C{fc + 2}+D{fc + 3})/E{fc + 2}/1+0*E{fc + 3}")
    ws.cell(r, 3).value = f"=(C{fc + 2}+D{fc + 3})/(E{fc + 2}+E{fc + 3})"
    ws.cell(r + 1, 1, "Sensibilidad (aisladas bien clasificadas)").font = f_n
    ws.cell(r + 1, 3, f"=C{fc + 2}/E{fc + 2}")
    ws.cell(r + 2, 1, "Especificidad (no aisladas bien clasificadas)").font = f_n
    ws.cell(r + 2, 3, f"=D{fc + 3}/E{fc + 3}")
    for rr in range(r, r + 3):
        ws.cell(rr, 3).number_format = "0.0%"
        ws.cell(rr, 3).font = f_n

    fp = r + 4
    ws.cell(fp, 1, "Perfiles: % de aislamiento observado y predicho por combinación de factores").font = f_b
    cabecera(ws, fp + 1, ["X1 localidad inundable", "X2 hospital inundable", "X3 red > umbral",
                          "Localidades", "Aisladas", "% observado", "Probabilidad predicha"])
    for i, (a1, a2, a3) in enumerate([(x1, x2, x3) for x1 in (0, 1) for x2 in (0, 1) for x3 in (0, 1)]):
        rr = fp + 2 + i
        ws.cell(rr, 1, a1)
        ws.cell(rr, 2, a2)
        ws.cell(rr, 3, a3)
        ws.cell(rr, 4, f"=COUNTIFS({X1},A{rr},{X2},B{rr},{X3},C{rr})")
        ws.cell(rr, 5, f"=COUNTIFS({X1},A{rr},{X2},B{rr},{X3},C{rr},{Y},1)")
        ws.cell(rr, 6, f'=IF(D{rr}>0,E{rr}/D{rr},"")')
        ws.cell(rr, 7, f"=1/(1+EXP(-($C$8+$C$9*A{rr}+$C$10*B{rr}+$C$11*C{rr})))")
        for j in range(1, 8):
            ws.cell(rr, j).font = f_n
            ws.cell(rr, j).border = caja
        ws.cell(rr, 6).number_format = ws.cell(rr, 7).number_format = "0.0%"

    fx = fp + 11
    ws.cell(fx, 1, "Descriptivo: % de aisladas según cada factor (sin controlar por los otros)").font = f_b
    cabecera(ws, fx + 1, ["Factor", "", "Aisladas si = 1", "Aisladas si = 0", "n con = 1", "n con = 0"])
    for i, (et, rg) in enumerate((("X1 localidad inundable", X1), ("X2 hospital inundable", X2),
                                  ("X3 red municipal > umbral", X3))):
        rr = fx + 2 + i
        ws.cell(rr, 1, et).font = f_n
        ws.cell(rr, 3, f"=COUNTIFS({rg},1,{Y},1)/COUNTIF({rg},1)").number_format = "0.0%"
        ws.cell(rr, 4, f"=COUNTIFS({rg},0,{Y},1)/COUNTIF({rg},0)").number_format = "0.0%"
        ws.cell(rr, 5, f"=COUNTIF({rg},1)").number_format = "#,##0"
        ws.cell(rr, 6, f"=COUNTIF({rg},0)").number_format = "#,##0"
    fz = fx + 6
    ws.cell(fz, 1, "Correlación entre predictores (phi)").font = f_b
    cabecera(ws, fz + 1, ["", "", "X1", "X2", "X3"])
    rgs = [X1, X2, X3]
    for i in range(3):
        ws.cell(fz + 2 + i, 1, f"X{i + 1}").font = f_b
        for j in range(3):
            ws.cell(fz + 2 + i, 3 + j, f"=CORREL({rgs[i]},{rgs[j]})").number_format = "0.000"
    for row in ws.iter_rows(min_row=fc, max_row=fz + 4):
        for c in row:
            if c.font is None or c.font.name != F:
                c.font = f_n
    for j, w in enumerate([46, 2, 14, 15, 12, 13, 13, 12, 12, 14, 13], 1):
        ws.column_dimensions[L(j)].width = w
    ws.row_dimensions[7].height = 32

    # ------------------------------------------------------------- Sensibilidad
    wsn = wb.create_sheet("Sensibilidad")
    wsn["A1"] = "Modelos de sensibilidad (Python)"
    wsn["A1"].font = f_t
    fila = 3
    for titulo, rr, nombres_ in (
            ("A. X3 continuo: % de la red municipal en tramos > 75 % inundados (por cada 10 puntos "
             "porcentuales), EE robustos por municipio", res_c, ["const", VARS[0], VARS[1], "X3_pct_red_tramos_x10"]),
            ("B. Modelo principal con EE ingenuos (sin agrupar por municipio)", res_n, ["const"] + VARS)):
        wsn.cell(fila, 1, titulo).font = f_b
        cabecera(wsn, fila + 1, ["Variable", "Coeficiente b", "EE", "z", "Valor p", "OR", "IC 95 % inf.",
                                 "IC 95 % sup."])
        for i, v in enumerate(nombres_):
            r = fila + 2 + i
            b, e = float(rr["b"][i]), float(rr["ee"][i])
            vals = [ETIQ[v], b, e, b / e, float(rr["p"][i]), math.exp(b), math.exp(b - 1.96 * e),
                    math.exp(b + 1.96 * e)]
            for j, val in enumerate(vals, 1):
                c = wsn.cell(r, j, val)
                c.font = f_n if j == 1 else f_in
                c.number_format = "0.0000E+00" if j == 5 else "0.0000"
        r = fila + 2 + len(nombres_)
        wsn.cell(r, 1, f"LL = {rr['ll']:.3f}; McFadden = {rr['mcfadden']:.4f}; AIC = {rr['aic']:.2f}; "
                       f"LR = {rr['lr']:.2f} (p = {rr['p_lr']:.2e})").font = f_i
        fila = r + 3
    wsn.column_dimensions["A"].width = 50
    for j in range(2, 9):
        wsn.column_dimensions[L(j)].width = 13

    # ------------------------------------------------------------- Resumen
    wr = wb.create_sheet("Resumen", 0)
    wr["A1"] = "Regresión logística: aislamiento de localidades rurales (agrupada por municipio)"
    wr["A1"].font = f_t
    for i, (k, v) in enumerate((("Modelo", "logit P(aislada) = b0 + b1·X1 + b2·X2 + b3·X3; EE robustos por "
                                           "conglomerado (municipio)."),
                                ("Escenario", info["escenario"])), 3):
        wr.cell(i, 1, k).font = f_b
        c = wr.cell(i, 2, v)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        wr.merge_cells(start_row=i, start_column=2, end_row=i, end_column=6)
    wr.row_dimensions[4].height = 56
    cabecera(wr, 6, ["Variable", "b", "OR", "IC 95 % OR", "Valor p (robusto)", "Significativo (α = 0.05)"])
    for i in range(4):
        r = 7 + i
        mr = 8 + i
        wr.cell(r, 1, f"=Modelo!A{mr}").font = f_link
        wr.cell(r, 2, f"=Modelo!C{mr}").font = f_link
        wr.cell(r, 3, f"=Modelo!G{mr}").font = f_link
        wr.cell(r, 4, f'=TEXT(Modelo!H{mr},"0.00")&" – "&TEXT(Modelo!I{mr},"0.00")').font = f_link
        wr.cell(r, 5, f"=Modelo!F{mr}").font = f_link
        wr.cell(r, 6, f'=IF(Modelo!F{mr}<0.05,"Sí","No")').font = f_link
        wr.cell(r, 2).number_format = "0.000"
        wr.cell(r, 3).number_format = "0.000"
        wr.cell(r, 5).number_format = "0.0000E+00"
    for i, (et, k, fm) in enumerate((("Pseudo R² de McFadden", "mcf", "0.000"), ("AUC", "auc", "0.000"),
                                     ("LR (gl = 3)", "lr", "0.00"), ("Valor p (LR)", "plr", "0.00E+00")), 12):
        wr.cell(i, 1, et).font = f_n
        c = wr.cell(i, 2, f"=Modelo!C{pos[k]}")
        c.font, c.number_format = f_link, fm
    fi = 17
    wr.cell(fi, 1, "Interpretación").font = f_b
    for k, t in enumerate(info["interpretacion"], fi + 1):
        c = wr.cell(k, 1, t)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        wr.merge_cells(start_row=k, start_column=1, end_row=k, end_column=6)
        wr.row_dimensions[k].height = 48
    wr.cell(fi + len(info["interpretacion"]) + 2, 1,
            "Verde = vínculo a la hoja Modelo. OR > 1: el factor aumenta los momios de quedar aislada.").font = f_i
    for j, w in enumerate([50, 11, 11, 16, 15, 14], 1):
        wr.column_dimensions[L(j)].width = w

    # ------------------------------------------------------------- Python
    wp = wb.create_sheet("Python")
    cabecera(wp, 1, ["Variable", "b", "EE robusto", "z", "p", "EE ingenuo", "OR"])
    for i, v in enumerate(["const"] + VARS, 2):
        j = i - 2
        for c, val in enumerate([ETIQ[v], res["b"][j], res["ee"][j], res["z"][j], res["p"][j],
                                 res["ee_ingenuo"][j], math.exp(res["b"][j])], 1):
            cel = wp.cell(i, c, float(val) if not isinstance(val, str) else val)
            cel.font = f_n
            if c > 1:
                cel.number_format = "0.0000E+00" if c == 5 else "0.000000"
    otros = [("LL", res["ll"]), ("LL0", res["ll0"]), ("LR", res["lr"]), ("p LR", res["p_lr"]),
             ("Wald robusta", res["wald"]), ("p Wald", res["p_wald"]), ("McFadden", res["mcfadden"]),
             ("AIC", res["aic"]), ("AUC", extra["auc"]), ("Hosmer-Lemeshow", extra["hl"]),
             ("gl HL", extra["hl_gl"]), ("p HL", extra["hl_p"]), ("n", res["n"]), ("Municipios", res["G"])]
    for i, (k, v) in enumerate(otros, 8):
        wp.cell(i, 1, k).font = f_n
        wp.cell(i, 2, float(v)).font = f_n
    wp.column_dimensions["A"].width = 50
    for j in range(2, 8):
        wp.column_dimensions[L(j)].width = 14

    # ------------------------------------------------------------- Método
    wt = wb.create_sheet("Metodo")
    txt = [
        ("Unidad", "Localidad rural (< 2,500 hab., ITER 2020); conglomerado = municipio (clave del ITER)."),
        ("Y", "1 = sin ruta a hospital público operativo tras cerrar los tramos con > 50 % de su longitud inundable "
              "(TR100; puntos cada 15 m); hospitales en la franja de 20 km."),
        ("X1", "1 = el punto de la localidad cae en píxel TR100 = 0."),
        ("X2", "1 = el hospital público operativo más cercano en la red completa (tiempo = longitud / velocidad "
               "RNC; conexión a 5 km/h) está en píxel TR100 = 0."),
        ("X3", "1 = más del 75 % de los km de red vial del municipio están en tramos con más del 75 % de su longitud "
               "inundada (% por tramo con puntos cada 15 m; cada punto, con la marca de su tramo, se asigna al "
               "municipio donde cae). Umbrales en Modelo!C4 y C5."),
        ("Estimación", "Máxima verosimilitud (Newton-Raphson). EE robustos por conglomerado: "
                       "(X'WX)⁻¹ Σ s_g s_g' (X'WX)⁻¹ · G/(G−1)·(n−1)/(n−k). p = 2Φ(−|z|); OR = exp(b); "
                       "IC = exp(b ± 1.96·EE)."),
        ("Ajuste", "LR = 2(LL − LL0), gl = 3; McFadden = 1 − LL/LL0; AIC = 2k − 2LL; AUC; Hosmer-Lemeshow por "
                   "perfiles de probabilidad predicha."),
        ("Advertencia", "Asociación, no causalidad. X3 es municipal y se estima con pocos municipios; su IC es amplio."),
        ("Fuentes", "rnc2025Edomex.gpkg; CENAPRED_IPI_2015_Inund_Munici.zip; tr100_prcent.tif; iter_15csv20.zip; "
                    "ESTABLECIMIENTO_SALUD_202609.xlsx; scripts 02, 03, 04, 05, 07, 08, 11, 14 y 16."),
    ]
    for i, (k, v) in enumerate(txt, 1):
        wt.cell(i, 1, k).font = f_b
        c = wt.cell(i, 2, v)
        c.font, c.alignment = f_n, Alignment(wrap_text=True, vertical="top")
        wt.cell(i, 1).alignment = Alignment(vertical="top")
        wt.row_dimensions[i].height = 32 if len(v) < 120 else 46
    wt.column_dimensions["A"].width = 14
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
    ap.add_argument("--umbral", type=float, default=50.0, help="cierre de tramos (% inundado >)")
    ap.add_argument("--umbral-tramo", type=float, default=75.0,
                    help="%% inundado para considerar un tramo muy inundado (defecto 75)")
    ap.add_argument("--umbral-x3", type=float, default=75.0,
                    help="%% de la red municipal en tramos muy inundados para X3 = 1 (defecto 75)")
    a = ap.parse_args()
    for k in RUTAS:
        if not getattr(a, k).exists():
            raise SystemExit(f"No existe: {getattr(a, k)}")
    t0 = time.time()
    SALIDAS.mkdir(exist_ok=True)

    print("1/4 Red municipal inundable (puntos cada 15 m)...")
    muns, _ = s08.municipios(a.cenapred)
    km, km_inund, km_tr = red_tramos_por_municipio(a, muns, a.umbral_tramo)
    mun = pd.DataFrame({"CVE_MPIO": [k for k, _, _ in muns],
                        "municipio": [s11.texto_utf8(n) for _, n, _ in muns], "km": km, "km_inund": km_inund,
                        "km_tramos_gt": km_tr})
    mun["pct_red_tramos_gt"] = 100 * mun.km_tramos_gt / mun.km
    mun["X3_red_municipal_gt75"] = (mun.pct_red_tramos_gt > a.umbral_x3).astype(int)

    print("2/4 Aislamiento, localidad inundable y hospital más cercano inundable...")
    it, info0 = variables(a)
    it = it.merge(mun[["CVE_MPIO", "pct_red_tramos_gt", "X3_red_municipal_gt75"]], on="CVE_MPIO", how="left")
    if it["X3_red_municipal_gt75"].isna().any():
        raise SystemExit("Hay localidades cuyo municipio no está en el marco municipal.")

    print("3/4 Regresión logística...")
    y = it["Y_aislada"].to_numpy(float)
    X = np.column_stack([np.ones(len(it))] + [it[v].to_numpy(float) for v in VARS])
    g = it["CVE_MPIO"].to_numpy()
    res = logit(y, X, g)
    res_n = logit(y, X, None)
    Xc = np.column_stack([np.ones(len(it)), it[VARS[0]], it[VARS[1]], it["pct_red_tramos_gt"] / 10]).astype(float)
    res_c = logit(y, Xc, g)
    hl, hl_gl, hl_p = hosmer_lemeshow(y, res["p_hat"])
    extra = {"auc": auc(y, res["p_hat"]), "hl": hl, "hl_gl": hl_gl, "hl_p": hl_p}

    OR = np.exp(res["b"])
    lo, hi = np.exp(res["b"] - 1.96 * res["ee"]), np.exp(res["b"] + 1.96 * res["ee"])
    frases = []
    for i, v in enumerate(VARS, 1):
        sig = "significativo" if res["p"][i] < 0.05 else "no significativo"
        frases.append(f"{ETIQ[v]}: OR = {OR[i]:.2f} (IC 95 % {lo[i]:.2f}–{hi[i]:.2f}; p = {res['p'][i]:.2e}, "
                      f"{sig}). Controlando por los otros factores, multiplica los momios de quedar aislada por "
                      f"{OR[i]:.2f}.")
    interp = frases + [
        (f"Ajuste: LR = {res['lr']:.1f} (gl = 3, p = {res['p_lr']:.1e}); McFadden = {res['mcfadden']:.3f}; "
         f"AUC = {extra['auc']:.3f}. Los errores robustos por municipio son "
         f"{np.mean(res['ee'][1:] / res['ee_ingenuo'][1:]):.1f} veces los ingenuos en promedio: la agrupación "
         "importa y debe reportarse."),
        "Asociación, no causalidad: el aislamiento lo produce el modelo de red; la regresión resume qué factores "
        "de exposición lo acompañan. X3 se estima con pocos municipios.",
    ]
    info = {"umbral_x3": a.umbral_x3, "umbral_tramo": a.umbral_tramo, "interpretacion": interp,
            "escenario": (f"{len(it):,d} localidades rurales en {res['G']} municipios. Tramos con > {a.umbral:g} % "
                          f"inundable cerrados ({info0['cerrados']:,d}); {info0['hosp']} hospitales públicos "
                          f"operativos ({info0['hosp_inund']} en zona inundable); puntos cada {a.intervalo:g} m; "
                          f"X3 = más del {a.umbral_x3:g} % de la red municipal en tramos con más del "
                          f"{a.umbral_tramo:g} % inundado "
                          f"({int(mun.X3_red_municipal_gt75.sum())} municipios).")}

    print("4/4 Escribiendo Excel...")
    ruta_x = SALIDAS / f"{NOMBRE}.xlsx"
    escribir_excel(ruta_x, it, mun, res, res_c, res_n, extra, info)
    pd.DataFrame({"variable": ["const"] + VARS, "b": res["b"], "ee_robusto": res["ee"], "z": res["z"],
                  "p": res["p"], "ee_ingenuo": res["ee_ingenuo"], "OR": OR, "OR_ic95_inf": lo,
                  "OR_ic95_sup": hi}).to_csv(SALIDAS / "16_coeficientes_logit.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 104)
    print(" REGRESIÓN LOGÍSTICA: P(localidad rural aislada), errores robustos agrupados por municipio")
    print("=" * 104)
    print(f"  {info['escenario']}")
    print(f"\n  Descriptivo (% aisladas):")
    for v in VARS:
        s1 = it.loc[it[v] == 1, "Y_aislada"]
        s0 = it.loc[it[v] == 0, "Y_aislada"]
        print(f"    {ETIQ[v]:52s} = 1: {100 * s1.mean():5.1f} % (n = {len(s1):,d})   "
              f"= 0: {100 * s0.mean():5.1f} % (n = {len(s0):,d})")
    print(f"\n  {'Variable':52s}{'b':>9s}{'EE rob.':>9s}{'z':>8s}{'p':>11s}{'OR':>8s}{'IC 95 % OR':>17s}{'EE ing.':>9s}")
    for i, v in enumerate(["const"] + VARS):
        print(f"  {ETIQ[v]:52s}{res['b'][i]:>9.3f}{res['ee'][i]:>9.3f}{res['z'][i]:>8.2f}{res['p'][i]:>11.2e}"
              f"{OR[i]:>8.2f}{lo[i]:>8.2f} – {hi[i]:<7.2f}{res['ee_ingenuo'][i]:>8.3f}")
    print(f"\n  LL = {res['ll']:.2f}; LL0 = {res['ll0']:.2f}; LR = {res['lr']:.2f} (gl 3, p = {res['p_lr']:.2e}); "
          f"Wald robusta = {res['wald']:.2f} (p = {res['p_wald']:.2e})")
    print(f"  McFadden = {res['mcfadden']:.4f}; AIC = {res['aic']:.2f}; AUC = {extra['auc']:.4f}; "
          f"Hosmer-Lemeshow = {hl:.2f} (gl {hl_gl}, p = {hl_p:.3f})")
    pc = (res["p_hat"] >= 0.5).astype(int)
    print(f"  Clasificación (0.5): exactitud {100 * np.mean(pc == y):.1f} %, sensibilidad "
          f"{100 * np.mean(pc[y == 1] == 1):.1f} %, especificidad {100 * np.mean(pc[y == 0] == 0):.1f} %")
    print(f"\n  Sensibilidad X3 continuo (por cada 10 pp de red en tramos muy inundados): OR = {math.exp(res_c['b'][3]):.3f} "
          f"(IC {math.exp(res_c['b'][3] - 1.96 * res_c['ee'][3]):.3f}–{math.exp(res_c['b'][3] + 1.96 * res_c['ee'][3]):.3f}, "
          f"p = {res_c['p'][3]:.2e}); McFadden = {res_c['mcfadden']:.4f}")
    print("\n  Interpretación:")
    for t in interp:
        print("   - " + t)
    print(f"\n  Excel: {ruta_x}   [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
