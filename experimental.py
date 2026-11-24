"""
experimental.py — Lectura de datos medidos en la maqueta y comparación con el modelo.

Las predicciones se calculan con fisica.prediccion_frenado, tomando la velocidad
medida al INICIO de la zona de frenado (v0), como indica el documento guía
(sección 14). El error porcentual usa la ecuación (16) con tratamiento del
denominador nulo (ver fisica.error_porcentual).
"""

from __future__ import annotations

import io
import math
import unicodedata

import numpy as np
import pandas as pd

import fisica

# Columnas estándar del archivo de ensayos
COLUMNAS = [
    "ensayo", "repeticion", "theta_deg", "masa_kg", "v0_m_s", "modelo",
    "mu_ef", "Fb_N", "d_exp_m", "t_exp_s", "observaciones",
]
NUMERICAS = ["repeticion", "theta_deg", "masa_kg", "v0_m_s", "mu_ef", "Fb_N", "d_exp_m", "t_exp_s"]

# Sinónimos aceptados en los encabezados (sin tildes, minúsculas)
SINONIMOS = {
    "ensayo": ["ensayo", "id", "identificacion", "id_ensayo", "prueba"],
    "repeticion": ["repeticion", "rep", "repeticiones", "n"],
    "theta_deg": ["theta_deg", "theta", "angulo", "angulo_deg", "theta (deg)", "theta(°)", "angulo (°)"],
    "masa_kg": ["masa_kg", "masa", "m", "m_kg", "masa (kg)"],
    "v0_m_s": ["v0_m_s", "v0", "velocidad", "velocidad_inicial", "v0 (m/s)"],
    "modelo": ["modelo", "modo_frenado", "tipo"],
    "mu_ef": ["mu_ef", "mu", "coef", "coeficiente", "muef", "μef", "mu_ef_calibrado"],
    "Fb_N": ["fb_n", "fb", "fuerza_frenado", "fuerza", "fb (n)"],
    "d_exp_m": ["d_exp_m", "d_exp", "distancia", "distancia_frenado", "df_exp", "d_f", "d (m)"],
    "t_exp_s": ["t_exp_s", "t_exp", "tiempo", "tiempo_frenado", "tf_exp", "t_f", "t (s)"],
    "observaciones": ["observaciones", "obs", "notas", "comentarios"],
}


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    return t.strip().lower()


def tabla_vacia(filas: int = 0) -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNAS})
    df["ensayo"] = df["ensayo"].astype("object")
    df["modelo"] = df["modelo"].astype("object")
    df["observaciones"] = df["observaciones"].astype("object")
    if filas:
        df = pd.concat([df, pd.DataFrame([{"modelo": "A"}] * filas)], ignore_index=True)
    return df[COLUMNAS]


def _a_numero(serie: pd.Series) -> pd.Series:
    """Convierte texto con coma o punto decimal a número (NaN si no es número)."""
    if serie.dtype.kind in "if":
        return serie.astype(float)
    s = serie.astype(str).str.strip().str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def normalizar_tabla(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas por sinónimos, agrega las faltantes y convierte tipos."""
    mapa = {}
    for col in df.columns:
        n = _normalizar(col)
        for estandar, sins in SINONIMOS.items():
            if n in [_normalizar(s) for s in sins] and estandar not in mapa.values():
                mapa[col] = estandar
                break
    df = df.rename(columns=mapa)
    for c in COLUMNAS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNAS].copy()
    for c in NUMERICAS:
        df[c] = _a_numero(df[c])
    df["modelo"] = df["modelo"].fillna("A").astype(str).str.strip().str.upper().str[:1]
    df.loc[~df["modelo"].isin(["A", "B"]), "modelo"] = "A"
    df["ensayo"] = df["ensayo"].fillna("").astype(str).str.strip()
    df["observaciones"] = df["observaciones"].fillna("").astype(str)
    # quitar filas completamente vacías
    vacias = df[["theta_deg", "masa_kg", "v0_m_s", "d_exp_m", "t_exp_s"]].isna().all(axis=1) & (df["ensayo"] == "")
    return df[~vacias].reset_index(drop=True)


def leer_csv(contenido: bytes | str) -> pd.DataFrame:
    """Lee un CSV con separador ',' o ';' y decimal '.' o ','."""
    if isinstance(contenido, bytes):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                texto = contenido.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    else:
        texto = contenido
    primera = texto.splitlines()[0] if texto.strip() else ""
    sep = ";" if primera.count(";") >= primera.count(",") and ";" in primera else ","
    df = pd.read_csv(io.StringIO(texto), sep=sep, dtype=str, skipinitialspace=True)
    return normalizar_tabla(df)


def comparar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Calcula la predicción del modelo para cada fila y los errores.

    Devuelve (tabla_resultados, avisos).
    """
    df = normalizar_tabla(df)
    avisos: list[str] = []
    filas = []
    for i, r in df.iterrows():
        nombre = r["ensayo"] or f"fila {i + 1}"
        faltan = [c for c in ("theta_deg", "masa_kg", "v0_m_s") if pd.isna(r[c])]
        if r["modelo"] == "A" and pd.isna(r["mu_ef"]):
            faltan.append("mu_ef")
        if r["modelo"] == "B" and pd.isna(r["Fb_N"]):
            faltan.append("Fb_N")
        if faltan:
            avisos.append(f"{nombre}: faltan datos ({', '.join(faltan)}); no se calcula la predicción.")
            pred = {"a": np.nan, "t_f": np.nan, "d_f": np.nan, "se_detiene": None}
        elif r["masa_kg"] <= 0 or r["v0_m_s"] < 0 or not (0 <= r["theta_deg"] < 90):
            avisos.append(f"{nombre}: valores fuera de rango (masa > 0, v0 ≥ 0, 0 ≤ θ < 90°).")
            pred = {"a": np.nan, "t_f": np.nan, "d_f": np.nan, "se_detiene": None}
        else:
            pred = fisica.prediccion_frenado(
                r["theta_deg"], r["masa_kg"], r["v0_m_s"], r["modelo"],
                mu_ef=r["mu_ef"] if r["modelo"] == "A" else 0.0,
                Fb=r["Fb_N"] if r["modelo"] == "B" else 0.0,
            )
            if not pred["se_detiene"]:
                avisos.append(f"{nombre}: con esos parámetros el modelo predice que NO se detiene.")
        d_sim = pred["d_f"] if pred["d_f"] is not None and math.isfinite(pred["d_f"]) else np.nan
        t_sim = pred["t_f"] if pred["t_f"] is not None and math.isfinite(pred["t_f"]) else np.nan
        d_exp, t_exp = r["d_exp_m"], r["t_exp_s"]
        err_d = fisica.error_porcentual(d_sim, d_exp)
        err_t = fisica.error_porcentual(t_sim, t_exp)
        if math.isfinite(d_exp) and abs(d_exp) < 1e-12 and math.isfinite(d_sim) and d_sim > 0:
            avisos.append(f"{nombre}: d_exp = 0 → el error relativo no está definido; se reporta la diferencia absoluta.")
        mu_cal = fisica.mu_ef_calibrado(r["v0_m_s"], d_exp, r["theta_deg"]) if all(
            math.isfinite(v) for v in (r["v0_m_s"], d_exp, r["theta_deg"])) else np.nan
        filas.append({
            "ensayo": r["ensayo"], "repeticion": r["repeticion"], "theta_deg": r["theta_deg"],
            "masa_kg": r["masa_kg"], "v0_m_s": r["v0_m_s"], "modelo": r["modelo"],
            "mu_ef": r["mu_ef"], "Fb_N": r["Fb_N"],
            "a_sim_m_s2": pred["a"],
            "d_exp_m": d_exp, "d_sim_m": d_sim,
            "dif_d_m": d_sim - d_exp if math.isfinite(d_sim) and math.isfinite(d_exp) else np.nan,
            "error_d_pct": err_d,
            "t_exp_s": t_exp, "t_sim_s": t_sim,
            "dif_t_s": t_sim - t_exp if math.isfinite(t_sim) and math.isfinite(t_exp) else np.nan,
            "error_t_pct": err_t,
            "mu_ef_calibrado_ec15": mu_cal,
            "observaciones": r["observaciones"],
        })
    return pd.DataFrame(filas), avisos


def resumen_por_ensayo(res: pd.DataFrame) -> pd.DataFrame:
    """
    Promedia las repeticiones de cada ensayo (ec. 16a): media y desviación
    estándar muestral; la predicción se recalcula con v0 promedio.
    """
    if res.empty:
        return res
    grupos = []
    for nombre, g in res.groupby("ensayo", sort=False):
        n = len(g)
        fila = {"ensayo": nombre, "n": n}
        for c in ("theta_deg", "masa_kg", "v0_m_s", "mu_ef", "Fb_N", "d_exp_m", "t_exp_s"):
            vals = g[c].dropna()
            fila[c] = vals.mean() if len(vals) else np.nan
            if c in ("v0_m_s", "d_exp_m", "t_exp_s"):
                fila[c.replace("_m_s", "").replace("_m", "").replace("_s", "") + "_desv"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        modelo = g["modelo"].iloc[0]
        fila["modelo"] = modelo
        ok = all(math.isfinite(fila[c]) for c in ("theta_deg", "masa_kg", "v0_m_s")) and (
            math.isfinite(fila["mu_ef"]) if modelo == "A" else math.isfinite(fila["Fb_N"]))
        if ok and fila["masa_kg"] > 0:
            p = fisica.prediccion_frenado(fila["theta_deg"], fila["masa_kg"], fila["v0_m_s"], modelo,
                                          mu_ef=fila["mu_ef"] if modelo == "A" else 0.0,
                                          Fb=fila["Fb_N"] if modelo == "B" else 0.0)
            fila["d_sim_m"] = p["d_f"] if math.isfinite(p["d_f"]) else np.nan
            fila["t_sim_s"] = p["t_f"] if math.isfinite(p["t_f"]) else np.nan
        else:
            fila["d_sim_m"] = np.nan
            fila["t_sim_s"] = np.nan
        fila["error_d_pct"] = fisica.error_porcentual(fila["d_sim_m"], fila["d_exp_m"])
        fila["error_t_pct"] = fisica.error_porcentual(fila["t_sim_s"], fila["t_exp_s"])
        grupos.append(fila)
    return pd.DataFrame(grupos)


def a_csv(df: pd.DataFrame, excel_es: bool = True) -> bytes:
    """CSV para Excel en español (; y coma decimal) o CSV estándar."""
    if excel_es:
        return df.to_csv(index=False, sep=";", decimal=",", float_format="%.6g").encode("utf-8-sig")
    return df.to_csv(index=False, float_format="%.6g").encode("utf-8")
