"""
graficas.py — Figuras Plotly del simulador (todas calculadas con fisica.py).
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import fisica

AZUL = "#1f6fd1"
NAVY = "#0b2545"
VERDE = "#2e9d5b"
NARANJA = "#ef7d1a"
ROJO = "#d63a3a"
GRIS = "#5f6b7a"
MORADO = "#8e44ad"
PALETA = [AZUL, NARANJA, VERDE, MORADO, ROJO, "#14a3b8", "#b8860b", GRIS]


def _base(fig: go.Figure, titulo: str, xt: str, yt: str, alto: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15, color=NAVY), x=0.01),
        height=alto, margin=dict(l=60, r=20, t=50, b=50),
        paper_bgcolor="white", plot_bgcolor="#fbfcfe",
        separators=",.",  # coma decimal
        font=dict(family="Segoe UI, Arial", color=NAVY, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text=xt, gridcolor="#e3e9f1", zeroline=True, zerolinecolor="#9aa5b4", showline=True, linecolor="#9aa5b4")
    fig.update_yaxes(title_text=yt, gridcolor="#e3e9f1", zeroline=True, zerolinecolor="#9aa5b4", showline=True, linecolor="#9aa5b4")
    return fig


def _eventos(fig: go.Figure, res: fisica.Resultado):
    if res.t_ini_fren is not None and res.t_ini_fren > 0:
        fig.add_vrect(x0=0, x1=res.t_ini_fren, fillcolor=AZUL, opacity=0.06, line_width=0)
    if res.t_ini_fren is not None:
        fin = res.t_detencion if res.resultado == fisica.RES_DETENIDO else (res.t_salida_fren or res.duracion)
        fig.add_vrect(x0=res.t_ini_fren, x1=fin, fillcolor=NARANJA, opacity=0.08, line_width=0)
        fig.add_vline(x=res.t_ini_fren, line=dict(color=NARANJA, dash="dash", width=1.5),
                      annotation_text="inicio del frenado", annotation_position="top right",
                      annotation_font=dict(size=10, color=NARANJA))
    if res.t_detencion is not None:
        fig.add_vline(x=res.t_detencion, line=dict(color=ROJO, dash="dash", width=1.5),
                      annotation_text=f"detención t = {res.t_detencion:.3f} s".replace(".", ","),
                      annotation_position="bottom right", annotation_font=dict(size=10, color=ROJO))
    if res.t_tope is not None:
        fig.add_vline(x=res.t_tope, line=dict(color=ROJO, dash="dot", width=1.5),
                      annotation_text="alcanza el tope", annotation_position="bottom left",
                      annotation_font=dict(size=10, color=ROJO))


def serie_temporal(res: fisica.Resultado, var: str) -> go.Figure:
    cfg = {
        "x": ("Gráfica 1 · Posición x(t)", "x (m)", AZUL, "linear"),
        "v": ("Gráfica 2 · Velocidad v(t)", "v (m/s)", VERDE, "linear"),
        "a": ("Gráfica 3 · Aceleración a(t)", "a (m/s²)", ROJO, "hv"),
    }[var]
    y = getattr(res, var)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.t, y=y, mode="lines", line=dict(color=cfg[2], width=2.6, shape=cfg[3]),
                             name=cfg[1], hovertemplate="t = %{x:.3f} s<br>%{y:.4f}<extra></extra>"))
    if var == "x":
        p = res.params
        fig.add_hline(y=p.x_ini_fren, line=dict(color=NARANJA, width=1, dash="dot"),
                      annotation_text="inicio zona de frenado", annotation_font=dict(size=10, color=NARANJA),
                      annotation_position="top left")
        fig.add_hline(y=p.x_fin_fren, line=dict(color=GRIS, width=1, dash="dot"),
                      annotation_text="fin zona de frenado", annotation_font=dict(size=10, color=GRIS),
                      annotation_position="bottom left")
    _eventos(fig, res)
    fig.update_layout(showlegend=False)
    return _base(fig, cfg[0], "t (s)", cfg[1], alto=320)


def _recortar(d: np.ndarray, techo: float) -> np.ndarray:
    d = d.astype(float).copy()
    d[~np.isfinite(d)] = np.nan
    d[d > techo] = np.nan
    return d


def distancia_vs_pendiente(p: fisica.Parametros, mus_extra: list[float] | None = None) -> go.Figure:
    th = np.round(np.arange(0.0, 20.0001, 0.1), 4)
    fig = go.Figure()
    techo = max(2.5 * p.L_fren, 1.0)
    curvas = [(p.mu_ef if p.modelo == "A" else None, "actual")]
    if p.modelo == "A":
        curvas += [(m, "extra") for m in (mus_extra or []) if abs(m - p.mu_ef) > 1e-9]
    for k, (mu, tipo) in enumerate(curvas):
        q = p if tipo == "actual" else p.copia(mu_ef=mu)
        b = fisica.barrido(q, "theta_deg", th)
        if p.modelo == "A":
            nombre = f"μef = {q.mu_ef:.2f}".replace(".", ",") + (" (actual)" if tipo == "actual" else "")
        else:
            nombre = f"Fb = {q.Fb:.2f} N (actual)".replace(".", ",")
        fig.add_trace(go.Scatter(
            x=th, y=_recortar(b["d_f"], techo), mode="lines", name=nombre,
            line=dict(color=PALETA[k % len(PALETA)], width=3 if tipo == "actual" else 2,
                      dash="solid" if tipo == "actual" else "dash"),
            hovertemplate="θ = %{x:.1f}°<br>d_f = %{y:.3f} m<extra>" + nombre + "</extra>"))
    # Ensayos de la maqueta 3°, 5°, 7°
    b37 = fisica.barrido(p, "theta_deg", [3, 5, 7])
    txt = [("no se detiene" if not np.isfinite(d) else f"{d:.3f} m".replace(".", ",")) for d in b37["d_f"]]
    fig.add_trace(go.Scatter(x=[3, 5, 7], y=_recortar(b37["d_f"], techo), mode="markers+text", name="ensayos 3°, 5°, 7°",
                             marker=dict(size=11, color="white", line=dict(color=NAVY, width=2.5)),
                             text=txt, textposition="top center", textfont=dict(size=11, color=NAVY)))
    # Punto actual
    ba = fisica.barrido(p, "theta_deg", [p.theta_deg])
    if np.isfinite(ba["d_f"][0]) and ba["d_f"][0] <= techo:
        fig.add_trace(go.Scatter(x=[p.theta_deg], y=ba["d_f"], mode="markers", name="configuración actual",
                                 marker=dict(size=14, color=ROJO, symbol="diamond", line=dict(color="white", width=1.5))))
    fig.add_hline(y=p.L_fren, line=dict(color=NARANJA, dash="dash", width=2),
                  annotation_text=f"longitud de la zona de frenado = {p.L_fren:.2f} m".replace(".", ","),
                  annotation_font=dict(color=NARANJA, size=11), annotation_position="top left")
    # Ángulo crítico
    if p.modelo == "A":
        thc = fisica.angulo_critico_deg(p.mu_ef)
        etiqueta = f"θc = arctan μef = {thc:.1f}°".replace(".", ",")
    else:
        r = p.Fb / (p.masa * p.g)
        thc = math.degrees(math.asin(r)) if r < 1 else None
        etiqueta = f"θc = arcsen(Fb/mg) = {thc:.1f}°".replace(".", ",") if thc is not None else ""
    if thc is not None and thc <= 20:
        fig.add_vrect(x0=thc, x1=20, fillcolor=ROJO, opacity=0.07, line_width=0)
        fig.add_vline(x=thc, line=dict(color=ROJO, width=1.5, dash="dot"), annotation_text=etiqueta + " · no se detiene",
                      annotation_font=dict(color=ROJO, size=11), annotation_position="top right")
    modo = "v0 constante" if p.modo == "v0" else f"suelta a {p.s_suelta:.2f} m (v de entrada varía con θ)".replace(".", ",")
    fig.update_yaxes(range=[0, techo * 1.02])
    return _base(fig, f"Gráfica 4 · Distancia de frenado contra pendiente ({modo})", "θ (°)", "d_f (m)", alto=420)


def distancia_vs_friccion(p: fisica.Parametros) -> go.Figure:
    techo = max(2.5 * p.L_fren, 1.0)
    fig = go.Figure()
    if p.modelo == "A":
        mus = np.round(np.arange(0.05, 0.9001, 0.005), 4)
        b = fisica.barrido(p, "mu_ef", mus)
        fig.add_trace(go.Scatter(x=mus, y=_recortar(b["d_f"], techo), mode="lines", name="d_f(μef)",
                                 line=dict(color=AZUL, width=3),
                                 hovertemplate="μef = %{x:.3f}<br>d_f = %{y:.3f} m<extra></extra>"))
        crit = math.tan(p.theta)
        fig.add_vrect(x0=0.05, x1=max(crit, 0.05), fillcolor=ROJO, opacity=0.07, line_width=0)
        fig.add_vline(x=crit, line=dict(color=ROJO, dash="dot", width=1.5),
                      annotation_text=f"μef = tan θ = {crit:.3f}".replace(".", ","), annotation_font=dict(color=ROJO, size=11))
        ba = fisica.barrido(p, "mu_ef", [p.mu_ef])
        xs, xt = p.mu_ef, "μef (coeficiente efectivo)"
        titulo = "Gráfica 5 · Distancia de frenado contra coeficiente de fricción (modelo A)"
    else:
        Fmax = max(4.0, 2.5 * p.Fb)
        Fs = np.round(np.linspace(0.0, Fmax, 400), 5)
        b = fisica.barrido(p, "Fb", Fs)
        fig.add_trace(go.Scatter(x=Fs, y=_recortar(b["d_f"], techo), mode="lines", name="d_f(Fb)",
                                 line=dict(color=AZUL, width=3),
                                 hovertemplate="Fb = %{x:.3f} N<br>d_f = %{y:.3f} m<extra></extra>"))
        crit = p.masa * p.g * math.sin(p.theta)
        fig.add_vrect(x0=0, x1=max(crit, 1e-6), fillcolor=ROJO, opacity=0.07, line_width=0)
        fig.add_vline(x=crit, line=dict(color=ROJO, dash="dot", width=1.5),
                      annotation_text=f"Fb = mg sen θ = {crit:.3f} N".replace(".", ","), annotation_font=dict(color=ROJO, size=11))
        ba = fisica.barrido(p, "Fb", [p.Fb])
        xs, xt = p.Fb, "Fb (N)"
        titulo = "Gráfica 5 · Distancia de frenado contra fuerza de frenado (modelo B)"
    if np.isfinite(ba["d_f"][0]) and ba["d_f"][0] <= techo:
        fig.add_trace(go.Scatter(x=[xs], y=ba["d_f"], mode="markers", name="configuración actual",
                                 marker=dict(size=14, color=ROJO, symbol="diamond", line=dict(color="white", width=1.5))))
    fig.add_hline(y=p.L_fren, line=dict(color=NARANJA, dash="dash", width=2),
                  annotation_text=f"zona de frenado = {p.L_fren:.2f} m".replace(".", ","),
                  annotation_font=dict(color=NARANJA, size=11), annotation_position="top right")
    fig.update_yaxes(range=[0, techo * 1.02])
    return _base(fig, titulo, xt, "d_f (m)", alto=420)


def distancia_vs_masa(p: fisica.Parametros, Fb_B: float) -> go.Figure:
    """Muestra que la masa se cancela en el modelo A pero no en el B."""
    ms = np.round(np.linspace(0.10, 2.00, 191), 4)
    pa = p.copia(modelo="A")
    pb = p.copia(modelo="B", Fb=Fb_B)
    ba = fisica.barrido(pa, "masa", ms)
    bb = fisica.barrido(pb, "masa", ms)
    techo = max(2.5 * p.L_fren, 1.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ms, y=_recortar(ba["d_f"], techo), name=f"Modelo A · μef = {pa.mu_ef:.2f}".replace(".", ","),
                             line=dict(color=AZUL, width=3)))
    fig.add_trace(go.Scatter(x=ms, y=_recortar(bb["d_f"], techo), name=f"Modelo B · Fb = {Fb_B:.2f} N".replace(".", ","),
                             line=dict(color=NARANJA, width=3, dash="dash")))
    fig.add_vline(x=p.masa, line=dict(color=GRIS, dash="dot"), annotation_text="masa actual",
                  annotation_font=dict(size=10, color=GRIS))
    fig.add_hline(y=p.L_fren, line=dict(color=NARANJA, dash="dot", width=1))
    fig.update_yaxes(range=[0, techo * 1.02])
    return _base(fig, "Efecto de la masa: el modelo A no depende de m; el modelo B sí", "m (kg)", "d_f (m)", alto=380)


def barras_escenarios(tabla, columna: str, titulo: str, yt: str) -> go.Figure:
    fig = go.Figure()
    vals = [v if v is not None and np.isfinite(v) else None for v in tabla[columna]]
    colores = [PALETA[i % len(PALETA)] for i in range(len(tabla))]
    txt = ["no se detiene" if v is None else f"{v:.3f}".replace(".", ",") for v in vals]
    fig.add_trace(go.Bar(x=tabla["Escenario"], y=[v if v is not None else 0 for v in vals], marker_color=colores,
                         text=txt, textposition="outside"))
    fig.update_layout(showlegend=False, hovermode="closest")
    return _base(fig, titulo, "", yt, alto=320)


def comparacion_experimental(res) -> go.Figure:
    fig = go.Figure()
    ok = res[np.isfinite(res["d_exp_m"].astype(float)) & np.isfinite(res["d_sim_m"].astype(float))]
    if not ok.empty:
        lim = float(max(ok["d_exp_m"].max(), ok["d_sim_m"].max())) * 1.15
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="coincidencia perfecta",
                                 line=dict(color=GRIS, dash="dash")))
        fig.add_trace(go.Scatter(x=ok["d_exp_m"], y=ok["d_sim_m"], mode="markers+text", name="ensayos",
                                 text=ok["ensayo"], textposition="top center",
                                 marker=dict(size=11, color=AZUL, line=dict(color="white", width=1)),
                                 hovertemplate="d_exp = %{x:.3f} m<br>d_sim = %{y:.3f} m<extra>%{text}</extra>"))
    fig.update_layout(hovermode="closest")
    return _base(fig, "Distancia de frenado: simulada contra experimental", "d_exp (m)", "d_sim (m)", alto=380)
