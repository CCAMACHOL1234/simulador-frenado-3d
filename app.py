"""
app.py — Laboratorio virtual: frenado de un vehículo en una pista con pendiente.

Proyecto final de Dinámica · Ingeniería Civil · Universidad Nacional de Colombia.

Ejecutar:   streamlit run app.py
(o con iniciar_simulador.bat en Windows)

Arquitectura
------------
* fisica.py      → modelo matemático (única fuente de verdad).
* graficas.py    → figuras Plotly.
* experimental.py→ lectura de CSV y comparación con ensayos.
* escena3d.html  → maqueta 3D (Three.js). Recibe las muestras t, x, v, a
                   calculadas en Python y solo las reproduce (interpola).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import experimental as exp
import fisica
import graficas

RAIZ = Path(__file__).parent

st.set_page_config(
    page_title="Simulador de frenado en pendiente 3D",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Valores predeterminados de los controles
# ---------------------------------------------------------------------------
DEFAULTS = {
    "theta": 5.0,
    "modelo": "A · Resistencia proporcional a la normal (μef·N)",
    "mu": 0.30,
    "nivel_B": "B2 · medio",
    "B1": 0.60, "B2": 1.00, "B3": 1.50,
    "Fb_pers": 1.00,
    "masa": 0.35,
    "modo": "Velocidad al iniciar el frenado (v₀)",
    "v0": 1.00,
    "s_suelta": 0.50,
    "L_fren": 0.80,
    "mu_r": 0.0,
    "t_r": 0.0,
}
MODELOS = ["A · Resistencia proporcional a la normal (μef·N)", "B · Fuerza de frenado constante (Fb)"]
NIVELES = ["B1 · suave", "B2 · medio", "B3 · fuerte", "Personalizado"]
MODOS = ["Velocidad al iniciar el frenado (v₀)", "Soltar desde el reposo"]

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("escenarios", [])
st.session_state.setdefault("nonce", 0)
st.session_state.setdefault("tabla_exp", exp.tabla_vacia(3))
st.session_state.setdefault("editor_ver", 0)


def poner(clave, valor):
    st.session_state[clave] = valor


def restaurar():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def simular_otra_vez():
    st.session_state.nonce += 1
    st.session_state.autoplay = True


def coma(texto: str) -> str:
    """Cambia el punto decimal por coma en los mensajes del modelo."""
    return re.sub(r"(?<=\d)\.(?=\d)", ",", texto)


def f(v, d=3, unidad=""):
    """Formato numérico con coma decimal."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    s = f"{v:.{d}f}".replace(".", ",").replace("-", "−")
    return f"{s} {unidad}".strip()


# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
  .block-container{padding-top:3.2rem;padding-bottom:2rem;max-width:1700px}
  [data-testid="stSidebar"]{background:#0b2545}
  [data-testid="stSidebar"] *{color:#e8eef7}
  [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"]{background:#8db3e2;border-color:#8db3e2}
  [data-testid="stSidebar"] button{background:#13315c;border:1px solid #2c4d7a}
  [data-testid="stSidebar"] button:hover{border-color:#8db3e2;background:#1b3f72}
  [data-testid="stSidebar"] [data-testid="stExpander"]{background:#0f2e55;border-color:#2c4d7a}
  [data-testid="stSidebar"] input{color:#0b2545!important}
  [data-testid="stSidebar"] [data-baseweb="select"] *{color:#0b2545}
  [data-testid="stSidebar"] h3{font-size:.95rem;letter-spacing:.06em;text-transform:uppercase;color:#8db3e2!important;margin:.6rem 0 .2rem}
  .encabezado{background:linear-gradient(110deg,#0b2545 0%,#13315c 60%,#1f6fd1 130%);color:#fff;padding:16px 22px;border-radius:12px;margin-bottom:12px}
  .encabezado .sup{font-size:.72rem;letter-spacing:.22em;color:#8db3e2;text-transform:uppercase;font-weight:600}
  .encabezado h1{font-size:1.55rem;margin:.15rem 0 .1rem;color:#fff;padding:0}
  .encabezado .sub{font-size:.88rem;color:#cfdcf0}
  .tarjetas{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
  .tarjeta{background:#fff;border:1px solid #d6dfeb;border-top:3px solid #1f6fd1;border-radius:9px;padding:8px 11px;min-width:0}
  .tarjeta .et{font-size:.72rem;color:#5f6b7a;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .tarjeta .va{font-size:1.18rem;font-weight:700;color:#0b2545;font-variant-numeric:tabular-nums;white-space:nowrap}
  .tarjeta .no{font-size:.7rem;color:#5f6b7a}
  .tarjeta.n{border-top-color:#ef7d1a}.tarjeta.v{border-top-color:#2e9d5b}.tarjeta.r{border-top-color:#d63a3a}.tarjeta.g{border-top-color:#9aa5b4}
  .estado{border-radius:11px;padding:12px 14px;color:#fff;margin-bottom:9px}
  .estado .t1{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;opacity:.85}
  .estado .t2{font-size:1.15rem;font-weight:800;margin:.1rem 0}
  .estado .t3{font-size:.82rem;opacity:.95;line-height:1.35}
  .nota{font-size:.78rem;color:#5f6b7a}
  .subt{font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#13315c;margin:10px 0 6px}
  div[data-testid="stTabs"] button p{font-weight:600}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# A. PANEL LATERAL — CONTROLES
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.markdown("## ⚙️ Controles del experimento")

# --- Ángulo -----------------------------------------------------------------
sb.markdown("### 1 · Inclinación")
sb.slider("Ángulo θ (°)", 0.0, 20.0, step=0.5, key="theta", format="%.1f",
          help="0° a 20° con resolución de 0,5°. La pista 3D gira alrededor de la bisagra.")
c1, c2, c3 = sb.columns(3)
c1.button("3°", on_click=poner, args=("theta", 3.0), width="stretch", help="Ensayo A1 de la maqueta")
c2.button("5°", on_click=poner, args=("theta", 5.0), width="stretch", help="Ensayo A2 de la maqueta")
c3.button("7°", on_click=poner, args=("theta", 7.0), width="stretch", help="Ensayo A3 de la maqueta")
theta = st.session_state.theta
sb.caption(
    f"Pendiente p = 100·tan θ = **{f(fisica.pendiente_porcentaje(theta), 2)} %** · "
    f"altura del soporte h = L·sen θ = **{f(1.80 * math.sin(math.radians(theta)) * 100, 1)} cm**"
    + ("  ·  ⭐ configuración de la maqueta física" if theta in (3.0, 5.0, 7.0) else "")
)

# --- Modelo de frenado --------------------------------------------------------
sb.markdown("### 2 · Frenado")
sb.radio("Modelo de la fuerza de frenado", MODELOS, key="modelo")
modelo = "A" if st.session_state.modelo.startswith("A") else "B"
if modelo == "A":
    sb.slider("Coeficiente efectivo μef", 0.05, 0.90, step=0.01, key="mu", format="%.2f",
              help="Agrupa la fricción del patín, la rodadura y otras pérdidas (H6).")
    c1, c2, c3 = sb.columns(3)
    c1.button("Baja", on_click=poner, args=("mu", 0.15), width="stretch", help="μef = 0,15")
    c2.button("Media", on_click=poner, args=("mu", 0.30), width="stretch", help="μef = 0,30")
    c3.button("Alta", on_click=poner, args=("mu", 0.60), width="stretch", help="μef = 0,60")
    sb.caption("Valores de **simulación**: baja 0,15 · media 0,30 · alta 0,60. No corresponden a un "
               "material real hasta calibrarlos con la maqueta (ec. 15).")
else:
    sb.radio("Nivel de frenado del patín", NIVELES, key="nivel_B", horizontal=False)
    with sb.expander("Configurar niveles B1, B2, B3 (por calibrar)"):
        st.number_input("B1 · suave (N)", 0.0, 20.0, step=0.05, key="B1", format="%.2f")
        st.number_input("B2 · medio (N)", 0.0, 20.0, step=0.05, key="B2", format="%.2f")
        st.number_input("B3 · fuerte (N)", 0.0, 20.0, step=0.05, key="B3", format="%.2f")
        st.caption("Valores provisionales. Se reemplazan por los medidos con el dinamómetro "
                   "(Fb = μp·Np para cada posición del tornillo).")
    if st.session_state.nivel_B == "Personalizado":
        sb.slider("Fuerza de frenado Fb (N)", 0.0, 10.0, step=0.01, key="Fb_pers")
nivel = st.session_state.nivel_B
Fb = {"B1 · suave": st.session_state.B1, "B2 · medio": st.session_state.B2,
      "B3 · fuerte": st.session_state.B3}.get(nivel, st.session_state.Fb_pers)

# --- Masa -----------------------------------------------------------------
sb.markdown("### 3 · Vehículo")
sb.slider("Masa del carrito m (kg)", 0.10, 2.00, step=0.01, key="masa", format="%.2f")
sb.caption(f"Peso mg = **{f(st.session_state.masa * fisica.G, 3)} N**")

# --- Zona de frenado ----------------------------------------------------------
sb.markdown("### 4 · Geometría de la pista")
sb.slider("Longitud de la zona de frenado (m)", 0.40, 1.20, step=0.05, key="L_fren",
          help="La pista total (1,80 m) y la zona de seguridad (0,20 m) se mantienen; "
               "la zona de aceleración se ajusta: L_acel = 1,60 − L_fren.")
L_acel = 1.80 - 0.20 - st.session_state.L_fren
sb.caption(f"Aceleración **{f(L_acel, 2)} m** · Frenado **{f(st.session_state.L_fren, 2)} m** · Seguridad **0,20 m** · Total **1,80 m**")

# --- Condición inicial --------------------------------------------------------
sb.markdown("### 5 · Condición inicial")
sb.radio("Modo", MODOS, key="modo",
         help="Elija UNA forma de definir la velocidad al iniciar el frenado; así se evitan condiciones contradictorias.")
modo = "v0" if st.session_state.modo.startswith("Velocidad") else "reposo"
s_max = round(L_acel - 0.09, 2)
if modo == "v0":
    sb.slider("Velocidad al iniciar el frenado v₀ (m/s)", 0.0, 3.0, step=0.01, key="v0", format="%.2f")
    sb.caption("El carrito aparece en el inicio de la zona de frenado con v₀. (Se ignora la zona de aceleración.)")
else:
    if st.session_state.s_suelta > s_max:
        st.session_state.s_suelta = s_max
    sb.slider("Posición de suelta: distancia antes de la zona de frenado (m)", 0.05, s_max, step=0.01, key="s_suelta",
              help="Se suelta desde el reposo; la velocidad de entrada se calcula con la zona de aceleración.")
    sb.caption("v₀ **no** se introduce: se calcula con v = √(2·a·s) en la zona de aceleración.")

# --- Avanzado -----------------------------------------------------------------
with sb.expander("Opciones avanzadas"):
    st.slider("Rodadura fuera de la zona de frenado μr", 0.0, 0.05, step=0.001, key="mu_r", format="%.3f",
              help="Solo actúa en la zona de aceleración y en la de seguridad. En la zona de frenado ya está "
                   "incluida en μef / Fb (H6), así que no se cuenta dos veces. 0 = despreciada.")
    st.slider("Tiempo de reacción t_r (s) — solo para d_total", 0.0, 2.0, step=0.1, key="t_r",
              help="d_total = v₀·t_r + d_f (ec. 14). No afecta la animación (en la maqueta no hay conductor).")

sb.markdown("---")
c1, c2 = sb.columns(2)
c1.button("↺ Restaurar valores", on_click=restaurar, width="stretch")
c2.button("↻ Simular nuevamente", on_click=simular_otra_vez, width="stretch", type="primary")

# ---------------------------------------------------------------------------
# Cálculo (Python = fuente de verdad)
# ---------------------------------------------------------------------------
P = fisica.Parametros(
    theta_deg=float(theta), masa=float(st.session_state.masa), modelo=modelo,
    mu_ef=float(st.session_state.mu), Fb=float(Fb), mu_r=float(st.session_state.mu_r),
    modo=modo, v0=float(st.session_state.v0), s_suelta=float(st.session_state.s_suelta),
    t_reaccion=float(st.session_state.t_r), L_fren=float(st.session_state.L_fren),
)
R = fisica.simular(P)
if not R.valido:
    st.error("Parámetros inválidos: " + " ".join(R.errores))
    st.stop()
FZ = R.fuerzas

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="encabezado">
  <div class="sup">Universidad Nacional de Colombia · Dinámica · Ingeniería Civil</div>
  <h1>Simulación y análisis dinámico del frenado de un vehículo en carreteras con pendiente</h1>
  <div class="sub">Laboratorio virtual 3D · Segunda ley de Newton · a = g (sen θ − μef cos θ) · d<sub>f</sub> = v₀² / (2|a|)</div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# B. ESCENA 3D  +  C. PANEL DE RESULTADOS
# ---------------------------------------------------------------------------
def datos_escena(res: fisica.Resultado) -> dict:
    p = res.params

    def lst(a, d=6):
        return [round(float(v), d) for v in a]

    def fin(v):
        return None if v is None or not math.isfinite(v) else float(v)

    return {
        "theta_deg": p.theta_deg,
        "pendiente_pct": fisica.pendiente_porcentaje(p.theta_deg),
        "masa": p.masa, "modelo": p.modelo, "g": p.g,
        "geo": {
            "L_total": p.L_total, "ancho": 0.30, "L_fren": p.L_fren, "L_seg": p.L_seg,
            "largo_carro": p.largo_carro, "espesor_tope": p.espesor_tope,
            "x_ini_fren": p.x_ini_fren, "x_fin_fren": p.x_fin_fren, "x_max": p.x_max,
        },
        "fuerzas": {
            "peso": FZ["peso"], "normal": FZ["normal"], "par": FZ["peso_paralelo"],
            "perp": FZ["peso_perpendicular"], "Fb": FZ["fuerza_frenado"], "Frod": FZ["fuerza_rodadura"],
        },
        "tray": {"t": lst(res.t), "x": lst(res.x), "v": lst(res.v), "a": lst(res.a),
                 "fase": [int(v) for v in res.fase], "F": lst(res.F_res)},
        "eventos": {
            "x_inicio": res.x_inicio, "t_ini_fren": fin(res.t_ini_fren), "t_det": fin(res.t_detencion),
            "x_det": fin(res.x_detencion) if res.resultado == fisica.RES_DETENIDO else None,
            "d_real": fin(res.d_frenado_real), "d_f_teo": fin(res.d_f_teorico), "t_tope": fin(res.t_tope),
        },
        "resultado": res.resultado,
        "mensaje": coma(res.mensaje),
        "nonce": st.session_state.nonce,
        "autoplay": bool(st.session_state.pop("autoplay", False)),
    }


def mostrar_html(html: str, alto: int) -> None:
    """Inserta la escena 3D. Usa st.iframe (Streamlit reciente) o components.html."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=alto)
    else:
        components.html(html, height=alto, scrolling=False)


@st.cache_data(show_spinner=False)
def _leer_plantilla(_mtime: float) -> str:
    return (RAIZ / "escena3d.html").read_text(encoding="utf-8")


def plantilla_html() -> str:
    """Lee escena3d.html (se vuelve a leer si el archivo cambia)."""
    return _leer_plantilla((RAIZ / "escena3d.html").stat().st_mtime)


col3d, colres = st.columns([3.1, 1.25], gap="medium")

with col3d:
    datos = datos_escena(R)
    html = plantilla_html().replace("/*__DATOS__*/null", json.dumps(datos, ensure_ascii=False))
    mostrar_html(html, 790)
    st.caption("🖱️ Arrastrar: girar · Rueda: zoom · Clic derecho o Shift + arrastrar: desplazar · Barra espaciadora: pausar/reanudar · "
               "Clic sobre las mini-gráficas: ir a ese instante. La animación interpola las muestras calculadas en Python.")

COLOR_RES = {
    fisica.RES_DETENIDO: "#2e9d5b", fisica.RES_INSUFICIENTE: "#d63a3a",
    fisica.RES_EXCEDE: "#d63a3a", fisica.RES_SIN_MOVIMIENTO: "#5f6b7a",
}
TITULO_RES = {
    fisica.RES_DETENIDO: "Detenido", fisica.RES_INSUFICIENTE: "Frenado insuficiente",
    fisica.RES_EXCEDE: "Excede longitud disponible", fisica.RES_SIN_MOVIMIENTO: "Preparado · sin movimiento",
}


def tarjeta(et, va, no="", cls=""):
    return f'<div class="tarjeta {cls}"><div class="et">{et}</div><div class="va">{va}</div><div class="no">{no}</div></div>'


with colres:
    st.markdown(
        f'<div class="estado" style="background:{COLOR_RES[R.resultado]}">'
        f'<div class="t1">Estado del experimento</div><div class="t2">{TITULO_RES[R.resultado]}</div>'
        f'<div class="t3">{coma(R.mensaje)}</div></div>', unsafe_allow_html=True)
    for adv in R.advertencias:
        st.warning(coma(adv), icon="⚠️")

    usa = (R.d_frenado_real / P.L_fren * 100) if R.d_frenado_real is not None else None
    t_f = R.t_f_teorico
    d_f = R.d_f_teorico
    st.markdown('<div class="subt">Frenado (ec. 7 y 11)</div>', unsafe_allow_html=True)
    cards = [
        tarjeta("Aceleración en frenado", f(R.a_frenado, 3, "m/s²"), "negativa = desacelera", "n"),
        tarjeta("Velocidad al frenar v₀", f(R.v_entrada, 3, "m/s"), "dada" if modo == "v0" else "calculada (suelta)", "v"),
        tarjeta("Tiempo de frenado t_f", f(t_f, 3, "s") if math.isfinite(t_f) else "∞ · no se detiene", "v₀ / |a|", "n"),
        tarjeta("Distancia de frenado d_f", f(d_f, 3, "m") if math.isfinite(d_f) else "∞ · no se detiene", "v₀² / (2|a|)", "n"),
        tarjeta("Uso de la zona", f(usa, 1, "%") if usa is not None else "> 100 %" if R.resultado != fisica.RES_SIN_MOVIMIENTO else "—",
                f"de {f(P.L_fren, 2)} m disponibles", "r" if usa is None else ""),
        tarjeta("Aceleración previa", f(R.a_libre, 3, "m/s²") if modo == "reposo" else "—",
                "zona de aceleración" if modo == "reposo" else "no aplica (modo v₀)", "g"),
    ]
    if P.t_reaccion > 0:
        cards.append(tarjeta("Detención total", f(R.d_total_reaccion, 3, "m") if math.isfinite(R.d_total_reaccion) else "∞",
                             f"v₀·t_r + d_f (t_r = {f(P.t_reaccion, 1)} s)", "g"))
    st.markdown('<div class="tarjetas">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="subt">Fuerzas (N)</div>', unsafe_allow_html=True)
    crit_txt = (f"θc = {f(fisica.angulo_critico_deg(P.mu_ef), 1)}°" if modelo == "A" else
                (f"θc = {f(math.degrees(math.asin(Fb / FZ['peso'])), 1)}°" if FZ["peso"] > 0 and Fb / FZ["peso"] < 1 else "θc > 90°"))
    cards = [
        tarjeta("Peso mg", f(FZ["peso"], 3, "N"), f"m = {f(P.masa, 2)} kg", ""),
        tarjeta("Normal N = mg cos θ", f(FZ["normal"], 3, "N"), "", ""),
        tarjeta("mg sen θ (a favor)", f(FZ["peso_paralelo"], 3, "N"), f"θ = {f(theta, 1)}° · p = {f(fisica.pendiente_porcentaje(theta), 2)} %", ""),
        tarjeta("Fuerza de frenado Fb", f(FZ["fuerza_frenado"], 3, "N"),
                f"μef·N (modelo A)" if modelo == "A" else f"constante ({nivel.split(' ')[0]})", "n"),
        tarjeta("μ equivalente Fb/N", f(FZ["mu_equivalente"], 3), f"tan θ = {f(math.tan(P.theta), 3)}", "g"),
        tarjeta("Ángulo crítico", crit_txt, "por encima no se detiene", "g"),
    ]
    st.markdown('<div class="tarjetas">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="subt">Escenarios</div>', unsafe_allow_html=True)
    nombre_def = f"Escenario {len(st.session_state.escenarios) + 1}"
    nombre = st.text_input("Nombre del escenario", value="", placeholder=nombre_def, label_visibility="collapsed")
    if st.button("💾 Guardar escenario", width="stretch", type="primary"):
        d = fisica.resumen_dict(R)
        d["Escenario"] = nombre.strip() or nombre_def
        d["L_fren"] = P.L_fren
        st.session_state.escenarios.append(d)
        st.toast(f"Guardado: {d['Escenario']}", icon="💾")
    st.caption(f"{len(st.session_state.escenarios)} escenario(s) guardado(s). Ver pestaña **Escenarios**.")

# ---------------------------------------------------------------------------
# Pestañas inferiores
# ---------------------------------------------------------------------------
tabs = st.tabs(["📈 Gráficas del movimiento", "📊 Análisis paramétrico", "🗂️ Escenarios",
                "🧪 Comparación experimental", "📘 Modelo y ecuaciones", "✅ Verificación"])

# --- 1. Gráficas x, v, a ----------------------------------------------------
with tabs[0]:
    st.caption("Curvas calculadas por el modelo. Zona azul: aceleración · zona naranja: frenado. "
               "Las líneas discontinuas marcan el inicio del frenado y el instante exacto de detención. "
               "Las mini-gráficas bajo la escena 3D muestran las mismas curvas con un marcador sincronizado con la animación.")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(graficas.serie_temporal(R, "x"), width="stretch")
    c2.plotly_chart(graficas.serie_temporal(R, "v"), width="stretch")
    c3.plotly_chart(graficas.serie_temporal(R, "a"), width="stretch")
    with st.expander("Ver tabla de muestras (t, x, v, a)"):
        df_m = pd.DataFrame({"t (s)": R.t, "x (m)": R.x, "v (m/s)": R.v, "a (m/s²)": R.a,
                             "fase": [fisica.NOMBRE_FASE[int(k)] for k in R.fase], "F resistente (N)": R.F_res})
        st.dataframe(df_m, width="stretch", height=300)
        st.download_button("⬇️ Descargar muestras (CSV)", exp.a_csv(df_m), "trayectoria.csv", "text/csv")

# --- 2. Análisis paramétrico ------------------------------------------------
with tabs[1]:
    if modelo == "A":
        extra = st.multiselect("Comparar además con μef =", [0.15, 0.20, 0.25, 0.40, 0.50, 0.60], default=[0.25, 0.40],
                               help="Curvas adicionales con los demás parámetros constantes (como la figura 6 del documento).")
    else:
        extra = []
    st.plotly_chart(graficas.distancia_vs_pendiente(P, extra), width="stretch")
    b = fisica.barrido(P, "theta_deg", [0, 3, 5, 7])
    tabla = pd.DataFrame({
        "θ (°)": [0, 3, 5, 7],
        "p (%)": [round(fisica.pendiente_porcentaje(t), 2) for t in (0, 3, 5, 7)],
        "v de entrada (m/s)": np.round(b["v_entrada"], 3),
        "a (m/s²)": np.round(b["a_frenado"], 3),
        "t_f (s)": [f(v, 3) if np.isfinite(v) else "no se detiene" for v in b["t_f"]],
        "d_f (m)": [f(v, 3) if np.isfinite(v) else "no se detiene" for v in b["d_f"]],
        "¿Cabe en la zona?": ["Sí" if c else "No" for c in b["cabe"]],
    })
    st.dataframe(tabla, width="stretch", hide_index=True)

    st.plotly_chart(graficas.distancia_vs_friccion(P), width="stretch")
    # Sensibilidad
    if modelo == "A":
        st.markdown("**Sensibilidad de d_f a μef** (demás variables constantes)")
        mus = sorted(set([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, round(P.mu_ef, 2)]))
        bm = fisica.barrido(P, "mu_ef", mus)
        ref = R.d_f_teorico
        filas = []
        for m_, d_ in zip(mus, bm["d_f"]):
            var = (d_ - ref) / ref * 100 if np.isfinite(d_) and math.isfinite(ref) and ref > 0 else None
            filas.append({"μef": f(m_, 2), "d_f (m)": f(d_, 3) if np.isfinite(d_) else "no se detiene",
                          "variación vs. actual (%)": (("+" if var > 0 else "") + f(var, 1)) if var is not None else "—",
                          "actual": "◀" if abs(m_ - P.mu_ef) < 1e-9 else ""})
        st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
    st.markdown("**La masa en los dos modelos**")
    st.plotly_chart(graficas.distancia_vs_masa(P, Fb if modelo == "B" else st.session_state.B2), width="stretch")
    st.info("Modelo A: Fb = μef·m·g·cos θ crece con m, de modo que **m se cancela** en a = g(sen θ − μef cos θ): "
            "cambiar solo la masa no cambia la desaceleración. Modelo B: con Fb constante, a = g sen θ − Fb/m "
            "**sí depende de m**: un carrito más pesado frena peor con el mismo patín.", icon="💡")

# --- 3. Escenarios ----------------------------------------------------------
with tabs[2]:
    esc = st.session_state.escenarios
    c1, c2, c3 = st.columns([1.3, 1, 1])

    def cargar_ejemplos():
        base = P.copia(modelo="A", modo="v0", v0=1.0, masa=0.35, L_fren=0.80, mu_r=0.0)
        ejemplos = [("Ejemplo 1 · θ=3°, μ=0,30", dict(theta_deg=3.0, mu_ef=0.30)),
                    ("Ejemplo 2 · θ=7°, μ=0,30", dict(theta_deg=7.0, mu_ef=0.30)),
                    ("Ejemplo 3 · θ=7°, μ=0,20", dict(theta_deg=7.0, mu_ef=0.20))]
        for n, cambios in ejemplos:
            r = fisica.simular(base.copia(**cambios))
            d = fisica.resumen_dict(r)
            d["Escenario"] = n
            st.session_state.escenarios.append(d)

    c1.button("➕ Cargar los 3 escenarios de ejemplo", on_click=cargar_ejemplos, width="stretch")
    if c3.button("🗑️ Eliminar todos", width="stretch", disabled=not esc):
        st.session_state.escenarios = []
        st.rerun()
    if not esc:
        st.info("Aún no hay escenarios. Ajuste los controles y pulse **Guardar escenario** en el panel de resultados, "
                "o cargue los ejemplos.")
    else:
        nombres = [e["Escenario"] for e in esc]
        ref_nombre = c2.selectbox("Referencia para el % de variación", nombres, index=0)
        ref = next(e for e in esc if e["Escenario"] == ref_nombre)

        def var_pct(v, r):
            if v is None or r is None or not math.isfinite(v) or not math.isfinite(r) or abs(r) < 1e-12:
                return None
            return (v - r) / abs(r) * 100

        filas = []
        for e in esc:
            filas.append({
                "Escenario": e["Escenario"],
                "θ (°)": e["theta_deg"], "p (%)": round(e["pendiente_pct"], 2),
                "Modelo": e["modelo"],
                "μef / Fb": (f"μ = {e['mu_ef']:.2f}" if e["modelo"] == "A" else f"Fb = {e['Fb']:.2f} N").replace(".", ","),
                "m (kg)": e["masa"],
                "v₀ entrada (m/s)": round(e["v_entrada"], 3),
                "a (m/s²)": round(e["a_frenado"], 3),
                "t_f (s)": e["t_f"], "d_f (m)": e["d_f"],
                "¿Se detiene en la zona?": "Sí" if e["se_detiene"] else "No",
                "Δ a (%)": var_pct(e["a_frenado"], ref["a_frenado"]),
                "Δ t_f (%)": var_pct(e["t_f"], ref["t_f"]),
                "Δ d_f (%)": var_pct(e["d_f"], ref["d_f"]),
            })
        T = pd.DataFrame(filas)
        st.dataframe(
            T, width="stretch", hide_index=True,
            column_config={
                "t_f (s)": st.column_config.NumberColumn(format="%.3f"),
                "d_f (m)": st.column_config.NumberColumn(format="%.3f"),
                "Δ a (%)": st.column_config.NumberColumn(format="%+.1f %%"),
                "Δ t_f (%)": st.column_config.NumberColumn(format="%+.1f %%"),
                "Δ d_f (%)": st.column_config.NumberColumn(format="%+.1f %%"),
            })
        st.caption("t_f y d_f vacíos = el vehículo no desacelera (no hay tiempo finito de detención). "
                   "Δ% = (valor − referencia)/|referencia| × 100.")
        g1, g2, g3 = st.columns(3)
        T["|a| (m/s²)"] = T["a (m/s²)"].abs()
        g1.plotly_chart(graficas.barras_escenarios(T, "d_f (m)", "Distancia de frenado", "d_f (m)"), width="stretch")
        g2.plotly_chart(graficas.barras_escenarios(T, "t_f (s)", "Tiempo de frenado", "t_f (s)"), width="stretch")
        g3.plotly_chart(graficas.barras_escenarios(T, "a (m/s²)", "Aceleración en frenado", "a (m/s²)"), width="stretch")
        c1, c2 = st.columns([2, 1])
        borrar = c1.multiselect("Seleccionar escenarios para eliminar", nombres)
        if c2.button("Eliminar seleccionados", disabled=not borrar, width="stretch"):
            st.session_state.escenarios = [e for e in esc if e["Escenario"] not in borrar]
            st.rerun()
        st.download_button("⬇️ Exportar tabla de escenarios (CSV)", exp.a_csv(T), "escenarios.csv", "text/csv")

# --- 4. Comparación experimental -------------------------------------------
with tabs[3]:
    st.markdown("#### Comparación experimental")
    st.markdown(
        "Introduzca o cargue los datos **medidos en la maqueta**. Para cada fila se calcula la predicción del modelo con "
        "las mismas condiciones (v₀ medida al **inicio** de la zona de frenado) y el error "
        "`|X_sim − X_exp| / |X_exp| × 100` (ec. 16). Si `X_exp = 0`, el error relativo no está definido y se "
        "reporta la diferencia absoluta.")
    st.caption("Columnas: ensayo, repeticion, theta_deg, masa_kg, v0_m_s, modelo (A/B), mu_ef, Fb_N, d_exp_m, t_exp_s, "
               "observaciones. Se acepta separador `,` o `;` y coma o punto decimal.")
    c1, c2, c3 = st.columns([2, 1, 1])
    archivo = c1.file_uploader("Cargar CSV de ensayos", type=["csv", "txt"])
    if archivo is not None and st.session_state.get("archivo_leido") != (archivo.name, archivo.size):
        try:
            st.session_state.tabla_exp = exp.leer_csv(archivo.getvalue())
            st.session_state.archivo_leido = (archivo.name, archivo.size)
            st.session_state.editor_ver += 1
            st.success(f"Leídas {len(st.session_state.tabla_exp)} filas de {archivo.name}.")
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {e}")
    if c2.button("Cargar plantilla de formato", width="stretch",
                 help="Carga datos/ejemplo.csv: valores FICTICIOS solo para ver el formato."):
        st.session_state.tabla_exp = exp.leer_csv((RAIZ / "datos" / "ejemplo.csv").read_bytes())
        st.session_state.editor_ver += 1
    if c3.button("Vaciar tabla", width="stretch"):
        st.session_state.tabla_exp = exp.tabla_vacia(3)
        st.session_state.editor_ver += 1
        st.session_state.archivo_leido = None

    editada = st.data_editor(
        st.session_state.tabla_exp, num_rows="dynamic", width="stretch",
        key=f"editor_{st.session_state.editor_ver}",
        column_config={
            "ensayo": st.column_config.TextColumn("Ensayo (ID)"),
            "repeticion": st.column_config.NumberColumn("Rep.", format="%d", min_value=1, step=1),
            "theta_deg": st.column_config.NumberColumn("θ (°)", min_value=0.0, max_value=89.0, format="%.2f"),
            "masa_kg": st.column_config.NumberColumn("m (kg)", min_value=0.0, format="%.3f"),
            "v0_m_s": st.column_config.NumberColumn("v₀ (m/s)", min_value=0.0, format="%.3f"),
            "modelo": st.column_config.SelectboxColumn("Modelo", options=["A", "B"], default="A"),
            "mu_ef": st.column_config.NumberColumn("μef (A)", min_value=0.0, format="%.3f"),
            "Fb_N": st.column_config.NumberColumn("Fb (N) (B)", min_value=0.0, format="%.3f"),
            "d_exp_m": st.column_config.NumberColumn("d_f exp (m)", min_value=0.0, format="%.3f"),
            "t_exp_s": st.column_config.NumberColumn("t_f exp (s)", min_value=0.0, format="%.3f"),
            "observaciones": st.column_config.TextColumn("Observaciones"),
        })
    res_exp, avisos = exp.comparar(editada)
    if res_exp.empty:
        st.info("La tabla está vacía. Escriba los datos de los ensayos o cargue un CSV. "
                "**No se muestran datos experimentales ficticios como si fueran mediciones.**")
    else:
        if res_exp["observaciones"].astype(str).str.contains("FICTICIO", case=False).any():
            st.warning("La tabla contiene filas marcadas como **FICTICIAS** (plantilla de formato). "
                       "No las use como resultados experimentales.", icon="⚠️")
        for a in avisos:
            st.caption("• " + a)
        st.markdown("**Resultados por fila**")
        st.dataframe(res_exp, width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                    ["a_sim_m_s2", "d_exp_m", "d_sim_m", "dif_d_m", "t_exp_s", "t_sim_s", "dif_t_s",
                                     "mu_ef_calibrado_ec15", "error_d_pct", "error_t_pct"]})
        agg = exp.resumen_por_ensayo(res_exp)
        if not agg.empty and (agg["n"] > 1).any():
            st.markdown("**Promedio por ensayo (media ± desviación estándar, ec. 16a)**")
            st.dataframe(agg, width="stretch", hide_index=True,
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in agg.columns
                                        if c not in ("ensayo", "n", "modelo")})
        c1, c2 = st.columns(2)
        c1.plotly_chart(graficas.comparacion_experimental(res_exp), width="stretch")
        with c2:
            st.markdown("**Calibración de μef (método 1, ec. 15)**")
            st.latex(r"\mu_{ef}=\frac{|a|+g\,\mathrm{sen}\,\theta}{g\cos\theta},\qquad |a|=\frac{v_0^2}{2\,d_{exp}}")
            st.caption("La columna `mu_ef_calibrado_ec15` estima μef de cada ensayo a partir de v₀ y d_exp medidos. "
                       "Sirve para calibrar los niveles B1, B2 y B3.")
            st.download_button("⬇️ Exportar comparación (CSV Excel ES: ; y ,)", exp.a_csv(res_exp, True),
                               "comparacion_experimental.csv", "text/csv", width="stretch")
            st.download_button("⬇️ Exportar comparación (CSV estándar)", exp.a_csv(res_exp, False),
                               "comparacion_experimental_std.csv", "text/csv", width="stretch")
            if not agg.empty:
                st.download_button("⬇️ Exportar promedios por ensayo (CSV)", exp.a_csv(agg, True),
                                   "promedios_por_ensayo.csv", "text/csv", width="stretch")

# --- 5. Modelo y ecuaciones -------------------------------------------------
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Diagrama de cuerpo libre y ecuaciones")
        st.markdown("Eje **x** paralelo a la pista, **positivo en el descenso**; eje **y** perpendicular, hacia afuera.")
        st.latex(r"N = mg\cos\theta \qquad mg_{\parallel}=mg\,\mathrm{sen}\,\theta \qquad mg_{\perp}=mg\cos\theta")
        st.latex(r"\Sigma F_x = ma \;\Rightarrow\; mg\,\mathrm{sen}\,\theta - F_b = ma")
        st.latex(r"\text{Modelo A: } F_b=\mu_{ef}N \;\Rightarrow\; a = g\,(\mathrm{sen}\,\theta-\mu_{ef}\cos\theta)")
        st.latex(r"\text{Modelo B: } F_b=\text{cte} \;\Rightarrow\; a = g\,\mathrm{sen}\,\theta - F_b/m")
        st.latex(r"v=v_0+at \qquad x=x_0+v_0t+\tfrac12at^2 \qquad v^2=v_0^2+2a\,\Delta x")
        st.latex(r"a<0:\quad t_f=\frac{v_0}{|a|}\qquad d_f=\frac{v_0^2}{2|a|}")
        st.latex(r"\text{Se detiene (modelo A, descenso)} \iff \mu_{ef}>\tan\theta \qquad p(\%)=100\tan\theta")
        st.latex(r"\text{Zonas sin patín: } a = g\,(\mathrm{sen}\,\theta-\mu_r\cos\theta)")
    with c2:
        st.markdown("#### Desarrollo numérico del escenario actual")
        s, c = math.sin(P.theta), math.cos(P.theta)
        st.markdown(f"- sen θ = {f(s, 5)}; cos θ = {f(c, 5)}; tan θ = {f(math.tan(P.theta), 5)}")
        st.markdown(f"- mg = {f(P.masa, 3)} · 9,81 = **{f(FZ['peso'], 4)} N**;  N = mg cos θ = **{f(FZ['normal'], 4)} N**")
        if modelo == "A":
            st.markdown(f"- a = 9,81 · ({f(s, 5)} − {f(P.mu_ef, 3)} · {f(c, 5)}) = **{f(R.a_frenado, 4)} m/s²**")
            st.markdown(f"- Fb = μef·N = {f(P.mu_ef, 3)} · {f(FZ['normal'], 4)} = **{f(FZ['fuerza_frenado'], 4)} N**")
        else:
            st.markdown(f"- a = 9,81 · {f(s, 5)} − {f(Fb, 3)} / {f(P.masa, 3)} = **{f(R.a_frenado, 4)} m/s²**")
        if modo == "reposo":
            st.markdown(f"- Zona de aceleración: a₁ = {f(R.a_libre, 4)} m/s²; v₀ = √(2·a₁·s) = √(2·{f(R.a_libre, 4)}·{f(P.s_suelta, 2)}) "
                        f"= **{f(R.v_entrada, 4)} m/s**")
        if math.isfinite(R.t_f_teorico):
            st.markdown(f"- t_f = {f(R.v_entrada, 3)} / {f(abs(R.a_frenado), 4)} = **{f(R.t_f_teorico, 4)} s**")
            st.markdown(f"- d_f = {f(R.v_entrada, 3)}² / (2·{f(abs(R.a_frenado), 4)}) = **{f(R.d_f_teorico, 4)} m** "
                        f"({'cabe' if R.d_f_teorico <= P.L_fren else 'NO cabe'} en {f(P.L_fren, 2)} m)")
        else:
            st.markdown("- a ≥ 0 → **no hay desaceleración neta**: no existe t_f finito.")
        st.markdown("#### Hipótesis y simplificaciones")
        st.markdown("""
- **H1** Movimiento rectilíneo · **H2** partícula (la rotación de las ruedas en 3D es solo visual) ·
  **H3** Fb constante · **H4** pendiente constante · **H5** sin resistencia del aire.
- **H6** La rodadura está **incluida en μef / Fb** dentro de la zona de frenado. Fuera de ella el patín no
  toca la pista y solo actúa μr (0 por defecto). Así **no se cuenta dos veces**.
- La fricción del patín **no** se aplica en la zona de aceleración.
- **Normal total ≠ precarga del patín.** En el modelo A, μef se define respecto a N = mg cos θ;
  la precarga Np del tornillo solo fija el valor de Fb = μp·Np (modelo B).
- **Detención:** se calcula el instante exacto t = v₀/|a|; después v = 0 y a = 0 (no se extrapola la parábola,
  no hay movimiento hacia atrás).
- **Reposo estático:** se exige Fs,max ≥ mg sen θ. Se toma μs = μef (o Fs,max = Fb), lo cual es conservador
  porque en la realidad μs ≥ μk. Con esta hipótesis, si el carrito logra detenerse también permanece quieto.
- **Tope final:** no es un sistema de frenado. Si el carrito llega a él, la simulación termina en el contacto y se
  informa la velocidad de impacto; el choque no se modela.
- **Punto de referencia:** x es la posición del centro del carrito (donde está el patín). d_f es una diferencia de
  posiciones, así que no cambia si se mide con el frente del carrito.
""")

# --- 6. Verificación --------------------------------------------------------
with tabs[5]:
    st.markdown("#### Casos de comprobación (calculados en vivo con `fisica.py`)")
    casos = []
    ref = fisica.prediccion_frenado(5.0, 0.35, 1.0, "A", mu_ef=0.30)
    casos.append(("Caso del documento: v₀=1 m/s, θ=5°, μ=0,30", "a ≈ −2,08; t_f ≈ 0,48 s; d_f ≈ 0,24 m",
                  f"a = {f(ref['a'], 3)}; t_f = {f(ref['t_f'], 3)}; d_f = {f(ref['d_f'], 3)}",
                  abs(ref["a"] + 2.08) < 0.005 and abs(ref["t_f"] - 0.48) < 0.005 and abs(ref["d_f"] - 0.24) < 0.005))
    for th, esperado in ((0, 0.170), (3, 0.0), (7, 0.0)):
        r = fisica.prediccion_frenado(th, 0.35, 1.0, "A", mu_ef=0.30)
        a_teo = 9.81 * (math.sin(math.radians(th)) - 0.30 * math.cos(math.radians(th)))
        casos.append((f"θ = {th}° (μ=0,30, v₀=1)", f"a = {f(a_teo, 3)}", f"a = {f(r['a'], 3)}; d_f = {f(r['d_f'], 3)}",
                      abs(r["a"] - a_teo) < 1e-12 and r["se_detiene"]))
    r = fisica.simular(fisica.Parametros(theta_deg=10, mu_ef=0.15))
    casos.append(("Fricción insuficiente (θ=10°, μ=0,15)", "no se detiene, t_f = ∞", f"{r.resultado}; t_f = {r.t_f_teorico}",
                  r.resultado == fisica.RES_INSUFICIENTE and math.isinf(r.t_f_teorico)))
    r = fisica.simular(fisica.Parametros(v0=0.0))
    casos.append(("v₀ = 0 con μ > tan θ", "permanece en reposo, d_f = 0", f"{r.resultado}; d = {r.d_frenado_real}",
                  r.resultado == fisica.RES_DETENIDO and r.d_frenado_real == 0))
    ra1 = fisica.simular(fisica.Parametros(masa=0.2)); ra2 = fisica.simular(fisica.Parametros(masa=1.5))
    rb1 = fisica.simular(fisica.Parametros(masa=0.2, modelo="B", Fb=1.0)); rb2 = fisica.simular(fisica.Parametros(masa=1.5, modelo="B", Fb=1.0))
    casos.append(("Masa variable · modelo A", "a no cambia", f"{f(ra1.a_frenado, 4)} vs {f(ra2.a_frenado, 4)}", abs(ra1.a_frenado - ra2.a_frenado) < 1e-12))
    casos.append(("Masa variable · modelo B", "a cambia", f"{f(rb1.a_frenado, 4)} vs {f(rb2.a_frenado, 4)}", abs(rb1.a_frenado - rb2.a_frenado) > 0.1))
    r = fisica.simular(fisica.Parametros(v0=2.0, mu_ef=0.25))
    casos.append(("d_f > 0,8 m (v₀=2, μ=0,25)", "excede la zona, no atraviesa el tope",
                  f"{r.resultado}; d_f teórico = {f(r.d_f_teorico, 3)}; x máx = {f(float(r.x.max()), 3)} ≤ {f(r.params.x_max, 3)}",
                  r.resultado == fisica.RES_EXCEDE and r.x.max() <= r.params.x_max + 1e-12))
    r = fisica.simular(fisica.Parametros())
    casos.append(("Detención exacta sin movimiento inverso", "v ≥ 0 y x no decrece",
                  f"min v = {f(float(r.v.min()), 4)}; x monótona = {bool(np.all(np.diff(r.x) >= -1e-12))}",
                  r.v.min() >= 0 and np.all(np.diff(r.x) >= -1e-12) and r.x[-1] == r.x_detencion))
    i = np.argmin(np.abs(r.t - r.t_detencion))
    casos.append(("Coherencia gráficas / indicadores", "x(t_det) − x_ini_fren = d_f",
                  f"{f(r.x[i] - r.params.x_ini_fren, 4)} vs {f(r.d_f_teorico, 4)}",
                  abs(r.x[i] - r.params.x_ini_fren - r.d_f_teorico) < 1e-9))
    df_c = pd.DataFrame(casos, columns=["Caso", "Esperado", "Obtenido", "¿Correcto?"])
    df_c["¿Correcto?"] = df_c["¿Correcto?"].map(lambda b: "✅ Sí" if b else "❌ No")
    st.dataframe(df_c, width="stretch", hide_index=True)
    st.caption("Las pruebas automatizadas completas están en `pruebas/test_fisica.py` (ejecutar `python -m pytest pruebas`). "
               "El cambio de controles durante la reproducción se maneja recargando la escena 3D con el nuevo "
               "escenario completo; nunca se mezclan muestras de dos escenarios.")

st.markdown('<p class="nota" style="text-align:center;margin-top:1.5rem">Simulador académico · Los resultados son '
            'predicciones del modelo; su validez se comprueba con la maqueta física.</p>', unsafe_allow_html=True)
