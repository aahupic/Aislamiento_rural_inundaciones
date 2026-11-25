# -*- coding: utf-8 -*-
"""
Script 01 - Aislamiento por Inundaciones: procesamiento de datos
================================================================
Eventos de lluvias, inundaciones y tormentas severas en el Estado de
México, 2000-2022,
a partir de la base CENAPRED "BASE_IMPACTO_SOCIOECONOMICO_DESASTRES_2000_2022".

Indicadores calculados:
  1. Eventos de lluvias, inundaciones y tormentas severas (número de registros)
  2. Defunciones
  3. Población afectada (miles)
  4. Viviendas dañadas (miles)
  5. Daños totales ($, pesos corrientes; la base reporta millones de pesos)

Criterios de procesamiento (explícitos para que puedan debatirse):
  - Fenómeno: 'Tipo de fenómeno' normalizado (minúsculas, sin acentos, sin
    espacios sobrantes) igual a 'lluvias', 'inundacion' o 'tormenta severa'
    (esta última incluida por decisión del 2026-11-20). Otros fenómenos
    hidrometeorológicos asociados (ciclón tropical, tormenta tropical,
    granizada) NO se incluyen en el total principal, pero se reportan
    aparte como análisis de sensibilidad.
  - Estado: se separa el campo 'Estado' por comas / ' y ' y se considera
    Edomex si algún componente es 'mexico' o 'estado de mexico'
    (se excluyen 'Ciudad de México', 'CDMX', 'Distrito Federal').
    * Escenario A (principal): registros SOLO del Estado de México.
    * Escenario B: además, registros multiestatales que incluyen al
      Edomex. Ojo: en ellos las cifras corresponden a TODO el evento y no
      pueden prorratearse por estado, por lo que B sobreestima.
    * 'Todo el país' / 'Varios Estados' no se asignan a Edomex.
  - Valores no numéricos ('SD', 'NSR', 'NR', 'SC', 'SR', vacíos, texto
    libre) se tratan como sin dato (NaN) y no suman; se reporta cuántos hay.
    Números con espacios de miles ('19 362') se convierten correctamente.
  - El archivo fuente tiene extensión .csv pero es un libro de Excel (.xlsx);
    se lee como Excel. Puede leerse directamente desde CENAPRED.zip.

Uso:
  python 01_lluvias_inundaciones_edomex.py                 # busca ..\\CENAPRED.zip
  python 01_lluvias_inundaciones_edomex.py RUTA_ARCHIVO    # .zip, .csv(xlsx) o .xlsx

Requisitos: pandas, openpyxl
"""

import io
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # D:\ARTUAEM
RUTA_DEFECTO = BASE_DIR / "CENAPRED.zip"
NOMBRE_FUENTE = "BASE_IMPACTO_SOCIOECONOMICO_DESASTRES_2000_2022"
SALIDAS = Path(__file__).resolve().parent / "salidas"

ANIO_MIN, ANIO_MAX = 2000, 2022
TIPOS_PRINCIPALES = {"lluvias", "inundacion", "tormenta severa"}
TIPOS_SENSIBILIDAD = {"ciclon tropical", "tormenta tropical", "granizada"}
NOMBRES_EDOMEX = {"mexico", "estado de mexico", "edo. de mexico", "edomex", "edo mex"}
SIN_ASIGNAR = {"todo el pais", "varios estados"}

COL = {
    "anio": "Año",
    "tipo": "Tipo de fenómeno",
    "estado": "Estado",
    "def": "Defunciones",
    "pob": "Población afectada",
    "viv": "Viviendas dañadas",
    "danos": "Total de daños (millones de pesos)",
}


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def normalizar(texto) -> str:
    """Minúsculas, sin acentos y sin espacios redundantes."""
    if pd.isna(texto):
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip().lower()


def a_numero(valor):
    """Convierte a float; códigos sin dato o texto libre -> NaN."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return float("nan")
    if isinstance(valor, (int, float)):
        return float(valor)
    s = str(valor).strip().replace(",", "")
    s = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", s)     # '19 362' -> '19362'
    try:
        return float(s)
    except ValueError:
        return float("nan")


def leer_fuente(ruta: Path) -> pd.DataFrame:
    """Lee la base desde .zip (miembro con NOMBRE_FUENTE) o desde el archivo."""
    if ruta.suffix.lower() == ".zip":
        with zipfile.ZipFile(ruta) as z:
            miembros = [n for n in z.namelist() if NOMBRE_FUENTE.lower() in n.lower()]
            if not miembros:
                sys.exit(f"No se encontró {NOMBRE_FUENTE} dentro de {ruta}")
            datos = z.read(miembros[0])
        print(f"Fuente: {ruta} -> {miembros[0]}")
    else:
        datos = ruta.read_bytes()
        print(f"Fuente: {ruta}")

    if datos[:2] == b"PK":                          # es un .xlsx (aunque diga .csv)
        df = pd.read_excel(io.BytesIO(datos), sheet_name=0)
    else:                                           # CSV real
        df = pd.read_csv(io.BytesIO(datos), encoding="latin-1")
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def estados_de(valor) -> list:
    partes = re.split(r",| y ", normalizar(valor))
    return [p.strip() for p in partes if p.strip()]


def resumir(df: pd.DataFrame) -> dict:
    return {
        "Eventos de lluvias, inundaciones y tormentas severas": len(df),
        "Defunciones": df["def_n"].sum(),
        "Población afectada (miles)": df["pob_n"].sum() / 1_000,
        "Viviendas dañadas (miles)": df["viv_n"].sum() / 1_000,
        "Daños totales ($)": df["danos_n"].sum() * 1_000_000,
    }


def imprimir(titulo: str, r: dict, df: pd.DataFrame):
    print(f"\n{titulo}")
    print("-" * len(titulo))
    print(f"  Eventos de lluvias, inundaciones y tormentas severas : {r['Eventos de lluvias, inundaciones y tormentas severas']:>14,d}")
    print(f"  Defunciones                                          : {r['Defunciones']:>14,.0f}")
    print(f"  Población afectada (miles)                           : {r['Población afectada (miles)']:>14,.2f}")
    print(f"  Viviendas dañadas (miles)                            : {r['Viviendas dañadas (miles)']:>14,.2f}")
    print(f"  Daños totales ($)                                    : {r['Daños totales ($)']:>14,.0f}")
    sd = {k: int(df[f'{k}_n'].isna().sum()) for k in ("def", "pob", "viv", "danos")}
    print(f"  Registros sin dato -> defunciones {sd['def']}, población {sd['pob']}, "
          f"viviendas {sd['viv']}, daños {sd['danos']}")


# ----------------------------------------------------------------------------
# Proceso principal
# ----------------------------------------------------------------------------
def main():
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else RUTA_DEFECTO
    if not ruta.exists():
        sys.exit(f"No existe el archivo: {ruta}")

    df = leer_fuente(ruta)
    print(f"Registros totales en la base: {len(df):,}")

    # Columnas numéricas limpias
    for k in ("def", "pob", "viv", "danos"):
        df[f"{k}_n"] = df[COL[k]].map(a_numero)
    df["anio_n"] = pd.to_numeric(df[COL["anio"]], errors="coerce")
    df["tipo_n"] = df[COL["tipo"]].map(normalizar)
    df["estados_l"] = df[COL["estado"]].map(estados_de)
    df["es_edomex"] = df["estados_l"].map(lambda l: any(e in NOMBRES_EDOMEX for e in l))
    df["solo_edomex"] = df["es_edomex"] & (df["estados_l"].map(len) == 1)

    periodo = df["anio_n"].between(ANIO_MIN, ANIO_MAX)
    lluv_inund = df["tipo_n"].isin(TIPOS_PRINCIPALES)

    a = df[periodo & lluv_inund & df["solo_edomex"]].copy()
    b = df[periodo & lluv_inund & df["es_edomex"]].copy()
    sens = df[periodo & df["tipo_n"].isin(TIPOS_SENSIBILIDAD) & df["solo_edomex"]].copy()
    nac = df[periodo & lluv_inund & df["estados_l"].map(lambda l: any(e in SIN_ASIGNAR for e in l))]

    print("\n" + "=" * 66)
    print(f" LLUVIAS, INUNDACIONES Y TORMENTAS SEVERAS - EDO. MÉX., {ANIO_MIN}-{ANIO_MAX}")
    print("=" * 66)

    ra = resumir(a)
    imprimir("Escenario A (principal): registros solo Edomex", ra, a)
    rb = resumir(b)
    imprimir(f"Escenario B: A + {len(b) - len(a)} evento(s) multiestatal(es) con Edomex "
             "(cifras de todo el evento)", rb, b)

    print("\nDesglose por tipo (escenario A):")
    print(a.groupby("tipo_n")
           .agg(eventos=("tipo_n", "size"), defunciones=("def_n", "sum"),
                poblacion=("pob_n", "sum"), viviendas=("viv_n", "sum"),
                danos_mdp=("danos_n", "sum")).to_string())

    print("\nSerie anual (escenario A):")
    anual = (a.groupby("anio_n")
              .agg(eventos=("tipo_n", "size"), defunciones=("def_n", "sum"),
                   poblacion=("pob_n", "sum"), viviendas=("viv_n", "sum"),
                   danos_mdp=("danos_n", "sum"))
              .reindex(range(ANIO_MIN, ANIO_MAX + 1), fill_value=0))
    anual.index = anual.index.astype(int)
    anual.index.name = "anio"
    print(anual.to_string(float_format=lambda v: f"{v:,.2f}"))

    print("\nSensibilidad - otros fenómenos hidrometeorológicos asociados (solo Edomex):")
    if sens.empty:
        print("  Sin registros (ciclón tropical, tormenta tropical, granizada).")
    else:
        print(sens.groupby("tipo_n").agg(eventos=("tipo_n", "size"), defunciones=("def_n", "sum"),
                                         poblacion=("pob_n", "sum"), viviendas=("viv_n", "sum"),
                                         danos_mdp=("danos_n", "sum")).to_string())
    print(f"\nNota: {len(nac)} registro(s) de lluvias/inundaciones/tormentas marcados como "
          "'Todo el país'/'Varios Estados' no se asignan a Edomex.")

    # Salidas
    SALIDAS.mkdir(exist_ok=True)
    pd.DataFrame([ra, rb], index=["A_solo_edomex", "B_incluye_multiestatales"]) \
        .to_csv(SALIDAS / "01_resumen_lluvias_inundaciones_edomex.csv", encoding="utf-8-sig")
    anual.to_csv(SALIDAS / "01_serie_anual_lluvias_inundaciones_edomex.csv", encoding="utf-8-sig")
    cols = [c for c in df.columns if not c.endswith(("_n", "_l")) and c not in ("es_edomex", "solo_edomex")]
    b[cols + ["solo_edomex"]].to_csv(SALIDAS / "01_eventos_lluvias_inundaciones_edomex.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"\nArchivos guardados en: {SALIDAS}")


if __name__ == "__main__":
    main()
