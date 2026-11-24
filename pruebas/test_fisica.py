"""
Pruebas automatizadas del modelo matemático (fisica.py) y de la comparación
experimental (experimental.py).

Ejecutar desde la carpeta del proyecto:
    python -m pytest pruebas -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experimental as exp  # noqa: E402
import fisica  # noqa: E402
from fisica import Parametros, simular  # noqa: E402

G = 9.81


# ---------------------------------------------------------------------------
# Caso de referencia del documento guía (sección 11)
# ---------------------------------------------------------------------------
def test_caso_referencia_documento():
    """v0 = 1,00 m/s, θ = 5°, μ = 0,30 → a ≈ −2,08 m/s², t_f ≈ 0,48 s, d_f ≈ 0,24 m."""
    r = simular(Parametros(theta_deg=5, mu_ef=0.30, v0=1.0, modo="v0"))
    assert r.resultado == fisica.RES_DETENIDO
    assert r.a_frenado == pytest.approx(-2.08, abs=0.005)
    assert r.t_f_teorico == pytest.approx(0.48, abs=0.005)
    assert r.d_f_teorico == pytest.approx(0.24, abs=0.005)
    # valores exactos
    a = G * (math.sin(math.radians(5)) - 0.30 * math.cos(math.radians(5)))
    assert r.a_frenado == pytest.approx(a, rel=1e-12)
    assert r.t_f_teorico == pytest.approx(1.0 / abs(a), rel=1e-12)
    assert r.d_f_teorico == pytest.approx(1.0 / (2 * abs(a)), rel=1e-12)
    # distancia recorrida en la animación = distancia teórica
    assert r.d_frenado_real == pytest.approx(r.d_f_teorico, rel=1e-12)


def test_ejemplo_numerico_carretera():
    """Ejemplo resuelto del documento: v0 = 16,67 m/s, θ = 5°, μ = 0,70 → a = −5,99; t = 2,78; d = 23,2."""
    p = fisica.prediccion_frenado(5, 1000, 16.67, "A", mu_ef=0.70)
    assert p["a"] == pytest.approx(-5.99, abs=0.01)
    assert p["t_f"] == pytest.approx(2.78, abs=0.01)
    assert p["d_f"] == pytest.approx(23.2, abs=0.05)


@pytest.mark.parametrize("theta,d_esperada", [(0, 0.20), (3, 0.26), (5, 0.31), (7, 0.40)])
def test_tabla_figura6_mu_025(theta, d_esperada):
    """Datos de la gráfica de barras del documento (v0 = 1 m/s, μef = 0,25)."""
    p = fisica.prediccion_frenado(theta, 0.35, 1.0, "A", mu_ef=0.25)
    assert p["d_f"] == pytest.approx(d_esperada, abs=0.006)


@pytest.mark.parametrize("theta,d_esperada", [(0, 0.13), (3, 0.15), (5, 0.16), (7, 0.19)])
def test_tabla_figura6_mu_040(theta, d_esperada):
    p = fisica.prediccion_frenado(theta, 0.35, 1.0, "A", mu_ef=0.40)
    assert p["d_f"] == pytest.approx(d_esperada, abs=0.006)


# ---------------------------------------------------------------------------
# Ángulos 0°, 3°, 7°
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta", [0.0, 3.0, 7.0])
def test_angulos_experimentales(theta):
    r = simular(Parametros(theta_deg=theta, mu_ef=0.30, v0=1.0))
    th = math.radians(theta)
    a = G * (math.sin(th) - 0.30 * math.cos(th))
    assert r.a_frenado == pytest.approx(a)
    assert r.resultado == fisica.RES_DETENIDO
    assert r.d_f_teorico == pytest.approx(1 / (2 * abs(a)))
    assert r.fuerzas["normal"] == pytest.approx(0.35 * G * math.cos(th))
    assert r.fuerzas["peso_paralelo"] == pytest.approx(0.35 * G * math.sin(th))


def test_mayor_pendiente_mayor_distancia():
    d = [fisica.prediccion_frenado(t, 0.35, 1.0, "A", mu_ef=0.3)["d_f"] for t in (0, 3, 5, 7)]
    assert all(d[i] < d[i + 1] for i in range(3))


def test_pendiente_porcentaje():
    assert fisica.pendiente_porcentaje(3) == pytest.approx(5.24, abs=0.01)
    assert fisica.pendiente_porcentaje(5) == pytest.approx(8.75, abs=0.01)
    assert fisica.pendiente_porcentaje(7) == pytest.approx(12.28, abs=0.01)
    assert fisica.angulo_desde_pendiente(10) == pytest.approx(5.71, abs=0.01)


def test_angulo_critico():
    assert fisica.angulo_critico_deg(0.30) == pytest.approx(16.7, abs=0.05)
    assert fisica.angulo_critico_deg(0.70) == pytest.approx(35.0, abs=0.05)


# ---------------------------------------------------------------------------
# Fricción insuficiente
# ---------------------------------------------------------------------------
def test_friccion_insuficiente_modelo_A():
    r = simular(Parametros(theta_deg=10, mu_ef=0.15, v0=1.0))
    assert r.a_frenado > 0
    assert r.resultado == fisica.RES_INSUFICIENTE
    assert math.isinf(r.t_f_teorico) and math.isinf(r.d_f_teorico)
    assert r.t_detencion is None
    assert r.t_tope is not None  # llega al tope; no hay tiempo de detención finito


def test_friccion_igual_a_tan_theta_no_desacelera():
    th = 5.0
    r = simular(Parametros(theta_deg=th, mu_ef=math.tan(math.radians(th)), v0=0.5))
    assert abs(r.a_frenado) < 1e-12
    assert r.resultado == fisica.RES_INSUFICIENTE
    assert math.isinf(r.d_f_teorico)


def test_friccion_insuficiente_modelo_B():
    p = Parametros(theta_deg=5, modelo="B", Fb=0.1, masa=0.35)  # mg sen θ = 0,299 N
    r = simular(p)
    assert r.resultado == fisica.RES_INSUFICIENTE


def test_frenado_ideal_sin_desaceleracion():
    assert fisica.frenado_ideal(1.0, 0.5) == (math.inf, math.inf)
    assert fisica.frenado_ideal(1.0, 0.0) == (math.inf, math.inf)
    assert fisica.frenado_ideal(0.0, -1.0) == (0.0, 0.0)
    with pytest.raises(ValueError):
        fisica.frenado_ideal(-1.0, -1.0)


# ---------------------------------------------------------------------------
# Velocidad inicial cero
# ---------------------------------------------------------------------------
def test_velocidad_inicial_cero_con_friccion_suficiente():
    r = simular(Parametros(v0=0.0, theta_deg=5, mu_ef=0.3))
    assert r.resultado == fisica.RES_DETENIDO
    assert r.d_frenado_real == 0.0
    assert r.t_frenado_real == 0.0
    assert np.all(r.v == 0) and np.all(r.x == r.x[0])
    assert r.reposo_estatico is True


def test_velocidad_inicial_cero_con_friccion_insuficiente_se_mueve():
    r = simular(Parametros(v0=0.0, theta_deg=10, mu_ef=0.1))
    assert r.resultado == fisica.RES_INSUFICIENTE
    assert r.x[-1] > r.x[0]


def test_reposo_sin_aceleracion_neta():
    """Modo reposo con θ = 0: no hay aceleración neta y el carrito no se mueve."""
    r = simular(Parametros(modo="reposo", theta_deg=0.0, s_suelta=0.5))
    assert r.resultado == fisica.RES_SIN_MOVIMIENTO
    assert np.all(r.x == r.x_inicio) and np.all(r.v == 0)
    r2 = simular(Parametros(modo="reposo", theta_deg=1.0, mu_r=0.05, s_suelta=0.5))
    assert r2.resultado == fisica.RES_SIN_MOVIMIENTO


# ---------------------------------------------------------------------------
# Masa variable en ambos modelos
# ---------------------------------------------------------------------------
def test_masa_no_cambia_aceleracion_modelo_A():
    a = [simular(Parametros(masa=m, modelo="A", mu_ef=0.3)).a_frenado for m in (0.1, 0.35, 1.0, 2.0)]
    assert max(a) - min(a) < 1e-12
    # pero sí cambian las fuerzas
    f1 = fisica.fuerzas(Parametros(masa=0.1))
    f2 = fisica.fuerzas(Parametros(masa=2.0))
    assert f2["peso"] == pytest.approx(20 * f1["peso"])
    assert f2["fuerza_frenado"] == pytest.approx(20 * f1["fuerza_frenado"])


def test_masa_cambia_aceleracion_modelo_B():
    r1 = simular(Parametros(masa=0.35, modelo="B", Fb=1.0))
    r2 = simular(Parametros(masa=1.0, modelo="B", Fb=1.0))
    assert r1.a_frenado == pytest.approx(G * math.sin(math.radians(5)) - 1.0 / 0.35)
    assert r2.a_frenado == pytest.approx(G * math.sin(math.radians(5)) - 1.0 / 1.0)
    assert r1.a_frenado < r2.a_frenado  # más masa → frena peor con la misma Fb
    assert r1.d_f_teorico < r2.d_f_teorico


def test_modelos_equivalentes_con_mu_equivalente():
    """Con Fb = μ·mg·cos θ, los modelos A y B dan la misma aceleración."""
    p = Parametros(theta_deg=7, masa=0.5, mu_ef=0.3)
    Fb = 0.3 * 0.5 * G * math.cos(math.radians(7))
    assert fisica.aceleracion_frenado(p) == pytest.approx(fisica.aceleracion_frenado(p.copia(modelo="B", Fb=Fb)))


# ---------------------------------------------------------------------------
# Distancia de frenado mayor que la zona
# ---------------------------------------------------------------------------
def test_distancia_mayor_que_zona():
    r = simular(Parametros(v0=2.0, mu_ef=0.25, theta_deg=5))
    assert r.d_f_teorico > 0.8
    assert r.resultado == fisica.RES_EXCEDE
    assert r.v_salida_fren > 0
    v_sal = math.sqrt(2.0 ** 2 + 2 * r.a_frenado * 0.8)
    assert r.v_salida_fren == pytest.approx(v_sal)
    # nunca atraviesa el tope
    assert r.x.max() <= r.params.x_max + 1e-12
    assert r.t_tope is not None
    # en la zona de seguridad no se aplica el patín: a = g sen θ (μr = 0)
    seg = r.fase == fisica.FASE_ZONA_SEGURIDAD
    assert np.allclose(r.a[seg], G * math.sin(math.radians(5)))


def test_tope_no_es_freno():
    """Al llegar al tope la simulación termina en el contacto con velocidad > 0."""
    r = simular(Parametros(v0=3.0, mu_ef=0.05, theta_deg=10))
    assert r.v_tope > 0
    assert r.x[-1] == pytest.approx(r.params.x_max)
    assert r.t_detencion is None


def test_longitud_de_frenado_variable():
    corto = simular(Parametros(v0=1.2, mu_ef=0.2, L_fren=0.4))
    largo = simular(Parametros(v0=1.2, mu_ef=0.2, L_fren=1.2))
    assert corto.resultado == fisica.RES_EXCEDE
    assert largo.resultado == fisica.RES_DETENIDO
    assert corto.params.L_acel == pytest.approx(1.2)
    assert largo.params.L_acel == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Detención exacta sin movimiento inverso
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("params", [
    Parametros(),
    Parametros(theta_deg=7, mu_ef=0.6, v0=2.5),
    Parametros(modo="reposo", s_suelta=0.7, theta_deg=7, mu_ef=0.5),
    Parametros(modelo="B", Fb=1.5),
])
def test_detencion_exacta_sin_retroceso(params):
    r = simular(params)
    assert r.resultado == fisica.RES_DETENIDO
    assert np.all(r.v >= 0)
    assert np.all(np.diff(r.x) >= -1e-12)
    # el instante exacto de detención está entre las muestras
    assert np.any(np.isclose(r.t, r.t_detencion, atol=1e-12))
    i = int(np.argmin(np.abs(r.t - r.t_detencion)))
    assert r.v[i] == pytest.approx(0, abs=1e-9)
    assert np.all(r.x[i:] == r.x_detencion)
    assert np.all(r.a[i:] == 0)


def test_tiempo_de_detencion_analitico():
    r = simular(Parametros(modo="reposo", s_suelta=0.5, theta_deg=5, mu_ef=0.3))
    a1 = G * math.sin(math.radians(5))
    t1 = math.sqrt(2 * 0.5 / a1)
    v_e = math.sqrt(2 * a1 * 0.5)
    assert r.t_ini_fren == pytest.approx(t1)
    assert r.v_entrada == pytest.approx(v_e)
    assert r.t_detencion == pytest.approx(t1 + v_e / abs(r.a_frenado))


# ---------------------------------------------------------------------------
# Modo reposo: aceleración y frenado son independientes
# ---------------------------------------------------------------------------
def test_fases_con_aceleraciones_distintas():
    r = simular(Parametros(modo="reposo", s_suelta=0.5, theta_deg=5, mu_ef=0.3))
    acel = r.a[r.fase == fisica.FASE_ACELERANDO]
    fren = r.a[r.fase == fisica.FASE_FRENANDO]
    assert np.allclose(acel, G * math.sin(math.radians(5)))  # sin fricción del patín
    assert np.allclose(fren, r.a_frenado)
    assert acel[0] > 0 > fren[0]


def test_rodadura_no_se_cuenta_dos_veces():
    """μr solo actúa fuera de la zona de frenado."""
    p = Parametros(mu_r=0.02, mu_ef=0.3)
    assert fisica.aceleracion_frenado(p) == pytest.approx(fisica.aceleracion_frenado(p.copia(mu_r=0.0)))
    assert fisica.aceleracion_libre(p) < fisica.aceleracion_libre(p.copia(mu_r=0.0))


def test_velocidad_documento_seccion6():
    """Con 0,8 m de aceleración y θ = 5°, v ideal ≈ 1,2 m/s."""
    assert fisica.velocidad_tras_recorrido(0, G * math.sin(math.radians(5)), 0.8) == pytest.approx(1.17, abs=0.01)


# ---------------------------------------------------------------------------
# Coherencia de muestras (gráficas, indicadores y posición 3D usan los mismos datos)
# ---------------------------------------------------------------------------
def test_coherencia_muestras():
    r = simular(Parametros(modo="reposo", s_suelta=0.6, theta_deg=7, mu_ef=0.4))
    # v ≈ dx/dt en el interior de cada tramo
    for tr in r.tramos:
        m = (r.t > tr.t0 + 1e-6) & (r.t < tr.t1 - 1e-6)
        if m.sum() > 3:
            t, x, v = r.t[m], r.x[m], r.v[m]
            dxdt = np.gradient(x, t)
            assert np.allclose(dxdt[1:-1], v[1:-1], atol=2e-3)
    assert len(r.t) == len(r.x) == len(r.v) == len(r.a) == len(r.fase) == len(r.F_res)
    assert np.all(np.diff(r.t) > 0)


def test_validacion_parametros():
    assert simular(Parametros(masa=0)).valido is False
    assert simular(Parametros(v0=-1)).valido is False
    assert simular(Parametros(modo="reposo", s_suelta=5.0)).valido is False
    assert simular(Parametros(L_fren=1.55)).valido is False


def test_barrido():
    b = fisica.barrido(Parametros(), "theta_deg", [0, 5, 20])
    assert b["d_f"][1] == pytest.approx(0.2408, abs=1e-4)
    assert math.isinf(b["d_f"][2])  # 20° > θc = 16,7°
    assert list(b["cabe"]) == [True, True, False]


def test_mu_calibrado_inverso():
    """La ecuación (15) recupera μef a partir de v0 y d."""
    p = fisica.prediccion_frenado(5, 0.35, 1.0, "A", mu_ef=0.3)
    assert fisica.mu_ef_calibrado(1.0, p["d_f"], 5) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Error porcentual y comparación experimental
# ---------------------------------------------------------------------------
def test_error_porcentual():
    assert fisica.error_porcentual(0.24, 0.26) == pytest.approx(7.69, abs=0.01)
    assert fisica.error_porcentual(0.0, 0.0) == 0.0
    assert math.isnan(fisica.error_porcentual(0.1, 0.0))
    assert math.isnan(fisica.error_porcentual(math.inf, 0.2))
    assert math.isnan(fisica.error_porcentual(None, 0.2))


def test_lectura_csv_ejemplo():
    ruta = Path(__file__).resolve().parents[1] / "datos" / "ejemplo.csv"
    df = exp.leer_csv(ruta.read_bytes())
    assert len(df) == 4
    assert df["theta_deg"].iloc[0] == pytest.approx(5.0)
    assert df["v0_m_s"].iloc[0] == pytest.approx(0.95)
    res, avisos = exp.comparar(df)
    assert {"d_sim_m", "error_d_pct", "t_sim_s", "error_t_pct"} <= set(res.columns)
    assert res["d_sim_m"].notna().all()
    agg = exp.resumen_por_ensayo(res)
    assert int(agg.loc[agg["ensayo"] == "EJEMPLO-A2", "n"].iloc[0]) == 3


def test_csv_coma_y_punto():
    txt_coma = "ensayo,theta,masa,v0,modelo,mu_ef,d_exp,t_exp\nX1,5,0.35,1.0,A,0.3,0.25,0.5\n"
    df = exp.leer_csv(txt_coma)
    res, _ = exp.comparar(df)
    assert res["d_sim_m"].iloc[0] == pytest.approx(0.2408, abs=1e-4)
    assert res["error_d_pct"].iloc[0] == pytest.approx(abs(0.2408 - 0.25) / 0.25 * 100, abs=0.05)


def test_comparacion_denominador_cero_y_faltantes():
    df = pd.DataFrame([
        {"ensayo": "Z", "theta_deg": 5, "masa_kg": 0.35, "v0_m_s": 1.0, "modelo": "A", "mu_ef": 0.3, "d_exp_m": 0.0, "t_exp_s": 0.0},
        {"ensayo": "F", "theta_deg": 5, "masa_kg": 0.35, "v0_m_s": None, "modelo": "A", "mu_ef": 0.3},
    ])
    res, avisos = exp.comparar(df)
    assert math.isnan(res["error_d_pct"].iloc[0])
    assert res["dif_d_m"].iloc[0] == pytest.approx(0.2408, abs=1e-4)
    assert any("faltan datos" in a for a in avisos)
