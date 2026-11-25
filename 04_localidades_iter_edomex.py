# -*- coding: utf-8 -*-
"""
Script 04 - Aislamiento por Inundaciones: localidades del Estado de México
=========================================================================
Fuente: inegi\\iter_15csv20.zip -> ITER_15CSV20.csv
(Censo de Población y Vivienda 2020, INEGI, Principales resultados por
localidad - ITER, entidad 15).

Indicadores:
  1. Localidades con coordenadas: total, habitantes y municipios.
  2. Localidades con coordenadas y menos de 2,500 habitantes (umbral INEGI de
     localidad rural): total, habitantes y municipios que las incluyen.

Criterios (explícitos para que puedan debatirse):
  - Se descartan los renglones agregados del ITER, que no son localidades:
        LOC = '0000' (total de la entidad / del municipio)
        LOC = '9998' (localidades de una vivienda, agrupadas)
        LOC = '9999' (localidades de dos viviendas, agrupadas)
  - "Con coordenadas" = LONGITUD y LATITUD presentes y convertibles de
    grados-minutos-segundos a grados decimales.
  - "Menos de 2,500 habitantes" = POBTOT < 2500 (POBTOT = 0 incluido, porque
    el ITER lista localidades sin población registrada; se reporta aparte).
  - Municipios = claves MUN distintas entre las localidades seleccionadas.
  - Como contexto se compara con la población total de la entidad
    (renglón MUN = '000', LOC = '0000').

Uso:
  python 04_localidades_iter_edomex.py
  python 04_localidades_iter_edomex.py RUTA\\iter_15csv20.zip   (o el .csv)

Requisitos: pandas.
Salidas (carpeta salidas\\):
  04_resumen_localidades_edomex.csv
  04_localidades_con_coordenadas_edomex.csv  (CVEGEO, nombres, lon/lat
      decimales, POBTOT y bandera de < 2,500 hab.)
"""

import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent          # D:\ARTUAEM
RUTA_DEFECTO = BASE_DIR / "inegi" / "iter_15csv20.zip"
SALIDAS = Path(__file__).resolve().parent / "salidas"
UMBRAL_RURAL = 2500
LOC_AGREGADOS = {"0000", "9998", "9999"}

_DMS = re.compile(r"""^\s*(\d+)\D+(\d+)\D+([\d.]+)\D*([NSEOW])\s*$""", re.I)


def dms_a_decimal(texto):
    """'99°41\'15.123" W' -> -99.687534. Devuelve NaN si no se puede leer."""
    if pd.isna(texto):
        return float("nan")
    m = _DMS.match(str(texto))
    if not m:
        return float("nan")
    g, mi, s, h = m.groups()
    v = float(g) + float(mi) / 60 + float(s) / 3600
    return -v if h.upper() in ("S", "W", "O") else v


def leer_iter(ruta: Path) -> pd.DataFrame:
    if ruta.suffix.lower() == ".zip":
        with zipfile.ZipFile(ruta) as z:
            nombre = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            datos = z.read(nombre)
        print(f"Fuente: {ruta} -> {nombre}")
    else:
        datos = ruta.read_bytes()
        print(f"Fuente: {ruta}")
    return pd.read_csv(io.BytesIO(datos), encoding="utf-8-sig", dtype=str)


def imprimir_bloque(titulo, df, pob_entidad):
    hab = int(df["POBTOT_n"].sum())
    print(f"\n{titulo}")
    print("-" * len(titulo))
    print(f"  Localidades : {len(df):>12,d}")
    print(f"  Habitantes  : {hab:>12,d}  ({100 * hab / pob_entidad:.2f} % de la población estatal)")
    print(f"  Municipios  : {df['MUN'].nunique():>12,d}")
    return {"localidades": len(df), "habitantes": hab, "municipios": df["MUN"].nunique(),
            "pct_poblacion_estatal": round(100 * hab / pob_entidad, 4)}


def main():
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else RUTA_DEFECTO
    if not ruta.exists():
        sys.exit(f"No existe el archivo: {ruta}")
    d = leer_iter(ruta)

    d["POBTOT_n"] = pd.to_numeric(d["POBTOT"], errors="coerce")
    pob_entidad = int(d.loc[(d["MUN"] == "000") & (d["LOC"] == "0000"), "POBTOT_n"].iloc[0])
    mun_total = d.loc[d["MUN"] != "000", "MUN"].nunique()

    loc = d[~d["LOC"].isin(LOC_AGREGADOS)].copy()
    loc["lon"] = loc["LONGITUD"].map(dms_a_decimal)
    loc["lat"] = loc["LATITUD"].map(dms_a_decimal)
    con = loc[loc["lon"].notna() & loc["lat"].notna()].copy()
    rural = con[con["POBTOT_n"] < UMBRAL_RURAL]

    print("\n" + "=" * 62)
    print(" LOCALIDADES - ESTADO DE MÉXICO (ITER 2020)")
    print("=" * 62)
    print(f"  Población total de la entidad   : {pob_entidad:>12,d}")
    print(f"  Municipios en la entidad        : {mun_total:>12,d}")
    print(f"  Localidades en el ITER          : {len(loc):>12,d}  "
          f"(sin coordenadas: {len(loc) - len(con):,d})")

    r1 = imprimir_bloque("1. Localidades con coordenadas", con, pob_entidad)
    r2 = imprimir_bloque(f"2. Localidades con coordenadas y menos de {UMBRAL_RURAL:,d} habitantes",
                         rural, pob_entidad)
    print(f"  (incluye {int((rural['POBTOT_n'] == 0).sum()):,d} localidades con POBTOT = 0; "
          f"{len(rural) / len(con) * 100:.2f} % de las localidades con coordenadas)")

    sin = loc[loc["lon"].isna() | loc["lat"].isna()]
    if len(sin):
        print(f"\nNota: {len(sin):,d} localidad(es) sin coordenadas suman "
              f"{int(sin['POBTOT_n'].sum()):,d} hab.; las agrupadas de 1 y 2 viviendas "
              f"(LOC 9998/9999) suman "
              f"{int(d.loc[(d['MUN'] == '000') & d['LOC'].isin({'9998', '9999'}), 'POBTOT_n'].sum()):,d} hab.")

    SALIDAS.mkdir(exist_ok=True)
    pd.DataFrame([r1, r2], index=["con_coordenadas", f"con_coordenadas_menos_{UMBRAL_RURAL}"]) \
        .to_csv(SALIDAS / "04_resumen_localidades_edomex.csv", encoding="utf-8-sig")
    con["CVEGEO"] = con["ENTIDAD"] + con["MUN"] + con["LOC"]
    con[f"menos_{UMBRAL_RURAL}"] = con["POBTOT_n"] < UMBRAL_RURAL
    con[["CVEGEO", "ENTIDAD", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "lon", "lat", "ALTITUD",
         "POBTOT", f"menos_{UMBRAL_RURAL}"]] \
        .to_csv(SALIDAS / "04_localidades_con_coordenadas_edomex.csv", index=False, encoding="utf-8-sig")
    print(f"\nArchivos guardados en: {SALIDAS}")


if __name__ == "__main__":
    main()
