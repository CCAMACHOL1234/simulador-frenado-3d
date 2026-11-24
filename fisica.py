"""
fisica.py — Modelo matemático del frenado de un vehículo en una pista con pendiente.

Proyecto final de Dinámica · Ingeniería Civil · Universidad Nacional de Colombia.

Este módulo es la ÚNICA fuente de verdad de los cálculos físicos del simulador.
La escena 3D y las gráficas solo reproducen las muestras que se generan aquí.

Convenciones (iguales a las del documento guía, sección 9)
-----------------------------------------------------------
* Unidades del Sistema Internacional: m, kg, s, N.
* Eje x paralelo a la pista, POSITIVO en el sentido del descenso.
* x = 0 en el extremo alto de la pista (donde está el soporte ajustable).
* x es la posición del punto de referencia del carrito (su centro, donde está
  el patín). Como d_f es una diferencia de posiciones, d_f no depende de qué
  punto del carrito se use como referencia, siempre que sea el mismo.
* θ ≥ 0 es el ángulo de la pista (descenso).

Hipótesis (documento guía, tabla H1–H6)
---------------------------------------
H1 Movimiento rectilíneo.  H2 Partícula.  H3 Fuerza de frenado constante.
H4 Pendiente constante.    H5 Sin resistencia del aire.
H6 La resistencia de rodadura está INCLUIDA en μef (modelo A) o en Fb
   (modelo B) durante el frenado. Fuera de la zona de frenado el patín no
   toca la pista; allí solo actúa la rodadura μr·N (por defecto μr = 0,
   es decir, despreciada). Así la rodadura nunca se cuenta dos veces.

Ecuaciones
----------
N   = m g cos θ                          (3)
Fg  = m g sen θ                          (1)
Modelo A:  Fb = μef N ;  a = g (sen θ − μef cos θ)      (4), (7)
Modelo B:  Fb = cte   ;  a = g sen θ − Fb / m           (5)
Tramo sin patín:        a = g (sen θ − μr cos θ)
v = v0 + a t ;  x = x0 + v0 t + ½ a t² ;  v² = v0² + 2 a Δx      (8)–(10)
Si a < 0:  t_f = v0/|a| ;  d_f = v0² / (2|a|)                    (11)
Modelo A en descenso: se detiene solo si μef > tan θ              (12)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

G = 9.81  # m/s²
EPS_V = 1e-9  # m/s: por debajo se considera velocidad nula
EPS_A = 1e-9   # m/s²: |a| menor se considera aceleración nula

# Códigos de fase para cada muestra de la trayectoria
FASE_REPOSO_INICIAL = 0   # nunca se movió (o aún no se suelta)
FASE_ACELERANDO = 1       # zona de aceleración (sin patín)
FASE_FRENANDO = 2         # zona de frenado (patín en contacto)
FASE_ZONA_SEGURIDAD = 3   # salió de la zona de frenado y sigue moviéndose
FASE_DETENIDO = 4         # se detuvo (v = 0) y permanece en reposo
FASE_TOPE = 5             # alcanzó el tope final (el choque NO se modela)

NOMBRE_FASE = {
    FASE_REPOSO_INICIAL: "Preparado",
    FASE_ACELERANDO: "Acelerando",
    FASE_FRENANDO: "Frenando",
    FASE_ZONA_SEGURIDAD: "Excede longitud disponible",
    FASE_DETENIDO: "Detenido",
    FASE_TOPE: "Alcanza el tope final",
}

# Resultados globales del experimento
RES_DETENIDO = "DETENIDO"
RES_INSUFICIENTE = "FRENADO_INSUFICIENTE"
RES_EXCEDE = "EXCEDE_LONGITUD"
RES_SIN_MOVIMIENTO = "SIN_MOVIMIENTO"

TEXTO_RESULTADO = {
    RES_DETENIDO: "Detenido dentro de la zona de frenado",
    RES_INSUFICIENTE: "Frenado insuficiente: no hay desaceleración neta",
    RES_EXCEDE: "Excede la longitud disponible de frenado",
    RES_SIN_MOVIMIENTO: "Sin movimiento: no hay aceleración neta para salir del reposo",
}


# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
@dataclass
class Parametros:
    """Datos de entrada de un ensayo virtual."""

    theta_deg: float = 5.0          # ángulo de la pista (°)
    masa: float = 0.35              # kg
    modelo: str = "A"               # "A": Fb = μef·N ; "B": Fb constante
    mu_ef: float = 0.30             # coeficiente efectivo de resistencia (modelo A)
    Fb: float = 1.00                # fuerza de frenado constante (N) (modelo B)
    mu_r: float = 0.0               # rodadura fuera de la zona de frenado
    modo: str = "v0"                # "v0": velocidad dada al iniciar el frenado
                                    # "reposo": se suelta desde el reposo
    v0: float = 1.00                # m/s (solo modo "v0")
    s_suelta: float = 0.50          # m antes de la zona de frenado (solo modo "reposo")
    t_reaccion: float = 0.0         # s (opcional, solo para d_total)
    # Geometría de la maqueta
    L_total: float = 1.80           # longitud de la pista (m)
    L_fren: float = 0.80            # longitud de la zona de frenado (m)
    L_seg: float = 0.20             # zona final de seguridad (m)
    largo_carro: float = 0.18       # m
    espesor_tope: float = 0.03      # m (bloque acolchado al final)
    g: float = G

    # ----- geometría derivada -----
    @property
    def L_acel(self) -> float:
        return self.L_total - self.L_fren - self.L_seg

    @property
    def x_ini_fren(self) -> float:
        """Posición (desde el extremo alto) donde empieza la zona de frenado."""
        return self.L_acel

    @property
    def x_fin_fren(self) -> float:
        return self.L_acel + self.L_fren

    @property
    def x_max(self) -> float:
        """Posición máxima del centro del carrito: su frente toca el tope."""
        return self.L_total - self.espesor_tope - self.largo_carro / 2.0

    @property
    def s_suelta_max(self) -> float:
        """Distancia de suelta máxima: la cola del carrito en el extremo alto."""
        return self.L_acel - self.largo_carro / 2.0

    @property
    def theta(self) -> float:
        return math.radians(self.theta_deg)

    def validar(self) -> list[str]:
        """Devuelve la lista de errores de entrada (vacía si todo es válido)."""
        e = []
        if not (0.0 <= self.theta_deg < 90.0):
            e.append("El ángulo debe estar entre 0° y 90°.")
        if self.masa <= 0:
            e.append("La masa debe ser positiva.")
        if self.modelo not in ("A", "B"):
            e.append("El modelo de frenado debe ser 'A' o 'B'.")
        if self.mu_ef < 0 or self.mu_r < 0:
            e.append("Los coeficientes de resistencia no pueden ser negativos.")
        if self.Fb < 0:
            e.append("La fuerza de frenado no puede ser negativa.")
        if self.modo not in ("v0", "reposo"):
            e.append("El modo inicial debe ser 'v0' o 'reposo'.")
        if self.v0 < 0:
            e.append("La velocidad inicial no puede ser negativa.")
        if self.L_fren <= 0 or self.L_acel <= self.largo_carro / 2.0:
            e.append("La longitud de frenado no deja espacio para la zona de aceleración.")
        if self.x_max <= self.x_fin_fren - 1e-12:
            e.append("La zona de seguridad es demasiado corta para el carrito.")
        if self.modo == "reposo" and not (0.0 < self.s_suelta <= self.s_suelta_max + 1e-9):
            e.append(
                f"La posición de suelta debe estar entre 0 y {self.s_suelta_max:.2f} m "
                "antes de la zona de frenado."
            )
        return e

    def copia(self, **cambios) -> "Parametros":
        d = asdict(self)
        d.update(cambios)
        return Parametros(**d)


# ---------------------------------------------------------------------------
# Fórmulas básicas
# ---------------------------------------------------------------------------
def pendiente_porcentaje(theta_deg: float) -> float:
    """p(%) = 100·tan θ   (13)"""
    return 100.0 * math.tan(math.radians(theta_deg))


def angulo_desde_pendiente(p: float) -> float:
    """θ = arctan(p/100) en grados (13)."""
    return math.degrees(math.atan(p / 100.0))


def angulo_critico_deg(mu_ef: float) -> float:
    """θc = arctan μef: por encima de este ángulo el modelo A no se detiene."""
    return math.degrees(math.atan(mu_ef))


def fuerzas(p: Parametros) -> dict:
    """Fuerzas del diagrama de cuerpo libre (magnitudes en N)."""
    W = p.masa * p.g
    N = W * math.cos(p.theta)                     # (3)
    W_par = W * math.sin(p.theta)                 # (1)
    W_perp = W * math.cos(p.theta)                # (2)
    Fb = p.mu_ef * N if p.modelo == "A" else p.Fb  # (4) / (5)
    F_rod = p.mu_r * N
    return {
        "peso": W,
        "normal": N,
        "peso_paralelo": W_par,
        "peso_perpendicular": W_perp,
        "fuerza_frenado": Fb,
        "fuerza_rodadura": F_rod,
        "mu_equivalente": Fb / N if N > 0 else float("nan"),
    }


def aceleracion_frenado(p: Parametros) -> float:
    """Aceleración en la zona de frenado (positiva = gana velocidad bajando)."""
    if p.modelo == "A":
        return p.g * (math.sin(p.theta) - p.mu_ef * math.cos(p.theta))   # (7)
    return p.g * math.sin(p.theta) - p.Fb / p.masa                        # modelo B


def aceleracion_libre(p: Parametros) -> float:
    """Aceleración fuera de la zona de frenado (patín sin contacto)."""
    return p.g * (math.sin(p.theta) - p.mu_r * math.cos(p.theta))


def frenado_ideal(v0: float, a: float) -> tuple[float, float]:
    """
    Tiempo y distancia de frenado con aceleración constante a (11),
    suponiendo una zona de frenado ilimitada.

    Devuelve (t_f, d_f). Si el vehículo no desacelera, devuelve (inf, inf):
    no se reporta un tiempo finito de detención.
    """
    if v0 < 0:
        raise ValueError("v0 debe ser ≥ 0")
    if v0 <= EPS_V:
        # Ya está en reposo: se queda quieto si la fuerza neta no lo empuja.
        return (0.0, 0.0) if a <= EPS_A else (math.inf, math.inf)
    if a > -EPS_A:
        return math.inf, math.inf
    return v0 / abs(a), v0 ** 2 / (2.0 * abs(a))


def velocidad_tras_recorrido(v0: float, a: float, d: float) -> float:
    """v = sqrt(v0² + 2 a d)  (10). Devuelve 0 si el carrito se detiene antes."""
    v2 = v0 ** 2 + 2.0 * a * d
    return math.sqrt(v2) if v2 > 0 else 0.0


def puede_mantenerse_en_reposo(p: Parametros) -> bool:
    """
    Criterio de reposo estático sobre la tira de frenado.

    Se requiere que la resistencia estática máxima del patín supere la componente
    del peso paralela a la pista: Fs,max ≥ m g sen θ.
    Simplificación: se toma μs = μef (modelo A) o Fs,max = Fb (modelo B). Como en
    la realidad μs ≥ μk, el criterio es CONSERVADOR: si se cumple con el valor
    cinético, también se cumple con el estático.
    """
    W_par = p.masa * p.g * math.sin(p.theta)
    f = fuerzas(p)
    return f["fuerza_frenado"] >= W_par - 1e-12


def mu_ef_calibrado(v0: float, d_exp: float, theta_deg: float, g: float = G) -> float:
    """
    Método 1 de calibración (15):  μef = (|a| + g sen θ) / (g cos θ),
    con |a| = v0² / (2 d_exp).
    """
    if d_exp <= 0 or v0 <= 0:
        return float("nan")
    th = math.radians(theta_deg)
    a_abs = v0 ** 2 / (2.0 * d_exp)
    return (a_abs + g * math.sin(th)) / (g * math.cos(th))


def error_porcentual(simulado: float, experimental: float) -> float:
    """
    Error (%) = |sim − exp| / |exp| × 100   (16)

    Tratamiento del denominador nulo: si exp = 0 y sim = 0 el error es 0 %;
    si exp = 0 y sim ≠ 0 el error relativo no está definido y se devuelve NaN
    (en ese caso debe usarse la diferencia absoluta). Valores no finitos → NaN.
    """
    try:
        s = float(simulado)
        e = float(experimental)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(s) and math.isfinite(e)):
        return float("nan")
    if abs(e) < 1e-12:
        return 0.0 if abs(s) < 1e-12 else float("nan")
    return abs(s - e) / abs(e) * 100.0


def prediccion_frenado(theta_deg: float, masa: float, v0: float, modelo: str,
                       mu_ef: float = 0.0, Fb: float = 0.0) -> dict:
    """Predicción de la etapa de frenado (v0 = velocidad al entrar a la zona)."""
    p = Parametros(theta_deg=theta_deg, masa=masa, modelo=modelo, mu_ef=mu_ef,
                   Fb=Fb, v0=v0, modo="v0")
    a = aceleracion_frenado(p)
    t_f, d_f = frenado_ideal(v0, a)
    return {"a": a, "t_f": t_f, "d_f": d_f, "se_detiene": math.isfinite(d_f)}


# ---------------------------------------------------------------------------
# Simulación por tramos de aceleración constante
# ---------------------------------------------------------------------------
@dataclass
class Tramo:
    t0: float
    t1: float
    x0: float
    v0: float
    a: float
    fase: int


def _avanzar(x0: float, v0: float, a: float, x_lim: float):
    """
    Avanza con aceleración constante desde x0 hasta x_lim o hasta detenerse.

    Devuelve (dt, x_final, v_final, motivo) con motivo en
    {"limite", "detenido", "reposo"}. La velocidad nunca se vuelve negativa:
    si el carrito se detiene se calcula el instante EXACTO de detención y no se
    extrapola la ecuación más allá de ese punto.
    """
    if v0 <= EPS_V and a <= 0:
        return 0.0, x0, 0.0, "reposo"
    dist = x_lim - x0
    if dist <= 0:
        return 0.0, x0, v0, "limite"
    if a < 0:
        d_stop = v0 ** 2 / (2.0 * -a)
        if d_stop <= dist:
            return v0 / -a, x0 + d_stop, 0.0, "detenido"
    disc = v0 ** 2 + 2.0 * a * dist
    disc = max(disc, 0.0)
    raiz = math.sqrt(disc)
    # Forma numéricamente estable de t = (−v0 + √disc)/a, válida también si a = 0
    t = 2.0 * dist / (v0 + raiz)
    return t, x_lim, raiz, "limite"


@dataclass
class Resultado:
    params: Parametros
    valido: bool
    errores: list[str]
    resultado: str = ""                  # RES_*
    mensaje: str = ""
    advertencias: list[str] = field(default_factory=list)
    # Aceleraciones
    a_libre: float = 0.0
    a_frenado: float = 0.0
    # Eventos
    x_inicio: float = 0.0
    t_ini_fren: Optional[float] = None   # instante de entrada a la zona de frenado
    v_entrada: float = 0.0               # velocidad al iniciar el frenado
    t_detencion: Optional[float] = None  # instante absoluto de detención
    x_detencion: Optional[float] = None
    t_salida_fren: Optional[float] = None
    v_salida_fren: Optional[float] = None
    t_tope: Optional[float] = None
    v_tope: Optional[float] = None
    # Frenado teórico (ec. 11, zona ilimitada)
    t_f_teorico: float = math.inf
    d_f_teorico: float = math.inf
    # Frenado real en la maqueta
    d_frenado_real: Optional[float] = None  # distancia recorrida sobre la tira
    t_frenado_real: Optional[float] = None
    cabe_en_zona: bool = False
    reposo_estatico: Optional[bool] = None
    d_total_reaccion: float = math.inf
    fuerzas: dict = field(default_factory=dict)
    tramos: list[Tramo] = field(default_factory=list)
    duracion: float = 0.0
    x_final: float = 0.0
    # Muestras
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fase: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    F_res: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def se_detiene(self) -> bool:
        return self.resultado == RES_DETENIDO


def simular(p: Parametros, n_muestras: int = 700, t_espera: float = 0.6) -> Resultado:
    """
    Simula el ensayo completo por tramos de aceleración constante:

    1) Zona de aceleración (solo modo "reposo"): a = g(sen θ − μr cos θ).
    2) Zona de frenado: a = a_frenado (modelo A o B).
    3) Zona de seguridad (si sale de la zona de frenado): patín sin contacto,
       a = g(sen θ − μr cos θ), hasta detenerse o hasta que el frente del
       carrito alcance el tope. El tope NO es un sistema de frenado: el choque
       no se modela y la simulación termina en ese instante.

    Devuelve un `Resultado` con resumen y muestras t, x, v, a, fase, F_res.
    """
    errores = p.validar()
    res = Resultado(params=p, valido=not errores, errores=errores)
    if errores:
        res.mensaje = "Parámetros inválidos: " + " ".join(errores)
        return res

    f = fuerzas(p)
    res.fuerzas = f
    a_lib = aceleracion_libre(p)
    a_fre = aceleracion_frenado(p)
    res.a_libre, res.a_frenado = a_lib, a_fre

    zonas = [
        (p.x_ini_fren, a_lib, FASE_ACELERANDO),
        (p.x_fin_fren, a_fre, FASE_FRENANDO),
        (p.x_max, a_lib, FASE_ZONA_SEGURIDAD),
    ]

    if p.modo == "v0":
        x, v = p.x_ini_fren, p.v0
    else:
        x, v = p.x_ini_fren - p.s_suelta, 0.0
    res.x_inicio = x
    t = 0.0
    tramos: list[Tramo] = []
    fin = None  # motivo de finalización

    for x_lim, a_z, fase in zonas:
        if x >= x_lim - 1e-12:
            continue  # el carrito ya está más allá de esta zona
        if fase == FASE_FRENANDO and res.t_ini_fren is None:
            res.t_ini_fren = t
            res.v_entrada = v
        dt, x1, v1, motivo = _avanzar(x, v, a_z, x_lim)
        if dt > 0:
            tramos.append(Tramo(t, t + dt, x, v, a_z, fase))
        t += dt
        x, v = x1, v1
        if motivo in ("detenido", "reposo"):
            fin = (motivo, fase)
            break
        if fase == FASE_FRENANDO:
            res.t_salida_fren = t
            res.v_salida_fren = v
    else:
        fin = ("tope", FASE_ZONA_SEGURIDAD)

    res.tramos = tramos
    res.duracion = t
    res.x_final = x
    motivo, fase_fin = fin

    # ----- Resumen del frenado (ecuación 11 con la velocidad de entrada) -----
    if res.t_ini_fren is not None:
        res.t_f_teorico, res.d_f_teorico = frenado_ideal(res.v_entrada, a_fre)

    if motivo == "reposo" and fase_fin == FASE_ACELERANDO:
        res.resultado = RES_SIN_MOVIMIENTO
        res.mensaje = (
            "El carrito no sale del reposo: la componente del peso "
            f"(g·sen θ = {p.g*math.sin(p.theta):.3f} m/s²) no supera la rodadura "
            f"(μr·g·cos θ = {p.g*p.mu_r*math.cos(p.theta):.3f} m/s²)."
        )
    elif fase_fin == FASE_FRENANDO and motivo in ("detenido", "reposo"):
        res.resultado = RES_DETENIDO
        res.t_detencion = t
        res.x_detencion = x
        res.d_frenado_real = x - p.x_ini_fren
        res.t_frenado_real = t - res.t_ini_fren
        res.cabe_en_zona = True
        res.reposo_estatico = puede_mantenerse_en_reposo(p)
        res.mensaje = (
            f"Se detiene a {res.d_frenado_real:.3f} m del inicio de la zona de frenado "
            f"(longitud disponible {p.L_fren:.2f} m)."
        )
    else:
        # Sale de la zona de frenado (o nunca la frena)
        if a_fre > -EPS_A and res.v_entrada > EPS_V or (res.v_entrada <= EPS_V and a_fre > 0):
            res.resultado = RES_INSUFICIENTE
            if p.modelo == "A":
                res.mensaje = (
                    f"μef = {p.mu_ef:.3f} ≤ tan θ = {math.tan(p.theta):.3f}: la fricción no "
                    "supera la componente del peso y el carrito no desacelera "
                    f"(a = {a_fre:+.3f} m/s²). Ángulo crítico θc = {angulo_critico_deg(p.mu_ef):.1f}°."
                )
            else:
                res.mensaje = (
                    f"Fb = {p.Fb:.3f} N ≤ m g sen θ = {f['peso_paralelo']:.3f} N: el patín "
                    f"no supera la componente del peso (a = {a_fre:+.3f} m/s²)."
                )
        else:
            res.resultado = RES_EXCEDE
            res.mensaje = (
                f"Desacelera (a = {a_fre:.3f} m/s²) pero necesitaría d_f = "
                f"{res.d_f_teorico:.3f} m y la zona de frenado mide {p.L_fren:.2f} m. "
                f"Sale de la zona a {res.v_salida_fren:.3f} m/s."
            )
        if motivo == "tope":
            res.t_tope = t
            res.v_tope = v
            res.advertencias.append(
                f"El frente del carrito alcanza el tope final a {v:.3f} m/s en t = {t:.3f} s. "
                "El choque con el tope no se modela: la simulación termina allí."
            )
        elif motivo == "detenido":
            res.t_detencion = t
            res.x_detencion = x
            res.advertencias.append(
                f"Se detiene en la zona de seguridad (x = {x:.3f} m) solo por rodadura; "
                "no es un frenado del patín."
            )

    if p.modelo == "A" and p.mu_ef <= math.tan(p.theta) and res.resultado != RES_SIN_MOVIMIENTO:
        res.advertencias.append(
            f"Condición (12) no cumplida: μef = {p.mu_ef:.3f} ≤ tan θ = {math.tan(p.theta):.3f}."
        )
    if res.resultado == RES_DETENIDO and res.d_frenado_real is not None and res.d_frenado_real > 0.75 * p.L_fren:
        res.advertencias.append(
            "La distancia de frenado usa más del 75 % de la zona: poco margen "
            "(el documento recomienda d_f ≤ 0,6 m para 0,8 m de zona)."
        )
    if math.isfinite(res.d_f_teorico):
        res.d_total_reaccion = res.v_entrada * p.t_reaccion + res.d_f_teorico

    _muestrear(res, n_muestras, t_espera)
    return res


def _estado_en(tr: Tramo, tau: float):
    x = tr.x0 + tr.v0 * tau + 0.5 * tr.a * tau ** 2
    v = tr.v0 + tr.a * tau
    return x, max(v, 0.0)


def _muestrear(res: Resultado, n: int, t_espera: float) -> None:
    p = res.params
    f = res.fuerzas
    T = res.duracion
    t_total = T + t_espera
    base = np.linspace(0.0, t_total, max(n, 2))
    eventos = [tr.t0 for tr in res.tramos] + [tr.t1 for tr in res.tramos] + [T]
    ts = np.unique(np.concatenate([base, np.array(eventos, dtype=float)]))
    ts = ts[(ts >= 0) & (ts <= t_total + 1e-12)]

    xs = np.empty_like(ts)
    vs = np.empty_like(ts)
    as_ = np.empty_like(ts)
    fs = np.empty(ts.shape, dtype=int)
    Fr = np.empty_like(ts)

    W_par = f["peso_paralelo"]
    x_fin = res.x_final

    if res.resultado == RES_DETENIDO:
        fase_final = FASE_DETENIDO
    elif res.resultado == RES_SIN_MOVIMIENTO:
        fase_final = FASE_REPOSO_INICIAL
    elif res.t_tope is not None:
        fase_final = FASE_TOPE
    else:
        fase_final = FASE_DETENIDO  # detenido en zona de seguridad por rodadura

    # Fuerza resistente según la zona
    F_por_fase = {
        FASE_ACELERANDO: f["fuerza_rodadura"],
        FASE_FRENANDO: f["fuerza_frenado"],
        FASE_ZONA_SEGURIDAD: f["fuerza_rodadura"],
    }

    k = 0
    for i, ti in enumerate(ts):
        # tramo activo: el último con t0 <= ti < t1
        while k < len(res.tramos) - 1 and ti >= res.tramos[k].t1:
            k += 1
        if res.tramos and res.tramos[k].t0 <= ti < res.tramos[k].t1:
            tr = res.tramos[k]
            xs[i], vs[i] = _estado_en(tr, ti - tr.t0)
            as_[i] = tr.a
            fs[i] = tr.fase
            Fr[i] = F_por_fase[tr.fase]
        else:
            # Estado final (en reposo o en el tope)
            xs[i] = x_fin
            if fase_final == FASE_TOPE:
                vs[i] = res.v_tope or 0.0
                # instante del contacto en adelante: la simulación se congela
                as_[i] = 0.0
                Fr[i] = 0.0
            else:
                vs[i] = 0.0
                as_[i] = 0.0
                # En reposo la fricción estática equilibra m g sen θ
                Fr[i] = W_par
            fs[i] = fase_final
            if fase_final == FASE_REPOSO_INICIAL:
                Fr[i] = W_par

    # Seguridad geométrica: el carrito nunca atraviesa el tope ni retrocede
    xs = np.clip(xs, 0.0, p.x_max)
    xs = np.maximum.accumulate(xs)
    vs = np.maximum(vs, 0.0)

    res.t, res.x, res.v, res.a, res.fase, res.F_res = ts, xs, vs, as_, fs, Fr


# ---------------------------------------------------------------------------
# Barridos paramétricos (gráficas 4 y 5)
# ---------------------------------------------------------------------------
def barrido(p: Parametros, variable: str, valores) -> dict:
    """
    Calcula la etapa de frenado al variar `variable` ('theta_deg', 'mu_ef', 'Fb',
    'masa' o 'v0') manteniendo constantes las demás.

    En modo "reposo" la velocidad de entrada se recalcula con cada valor (depende
    de θ); en modo "v0" se usa v0 directamente.
    """
    valores = np.asarray(valores, dtype=float)
    a_l, v_e, a_f, t_f, d_f = [], [], [], [], []
    for val in valores:
        q = p.copia(**{variable: float(val)})
        al = aceleracion_libre(q)
        if q.modo == "reposo":
            ve = velocidad_tras_recorrido(0.0, al, q.s_suelta) if al > 0 else 0.0
        else:
            ve = q.v0
        af = aceleracion_frenado(q)
        tf, df = frenado_ideal(ve, af)
        a_l.append(al)
        v_e.append(ve)
        a_f.append(af)
        t_f.append(tf)
        d_f.append(df)
    d = np.array(d_f)
    return {
        "valores": valores,
        "a_libre": np.array(a_l),
        "v_entrada": np.array(v_e),
        "a_frenado": np.array(a_f),
        "t_f": np.array(t_f),
        "d_f": d,
        "cabe": np.isfinite(d) & (d <= p.L_fren),
    }


def resumen_dict(res: Resultado) -> dict:
    """Resumen serializable (para escenarios y para la escena 3D)."""
    def fin(v):
        return None if v is None or not math.isfinite(v) else float(v)

    p = res.params
    return {
        "theta_deg": p.theta_deg,
        "pendiente_pct": pendiente_porcentaje(p.theta_deg),
        "masa": p.masa,
        "modelo": p.modelo,
        "mu_ef": p.mu_ef,
        "Fb": res.fuerzas.get("fuerza_frenado") if res.fuerzas else None,
        "modo": p.modo,
        "v0": p.v0,
        "s_suelta": p.s_suelta,
        "L_fren": p.L_fren,
        "v_entrada": res.v_entrada,
        "a_libre": res.a_libre,
        "a_frenado": res.a_frenado,
        "t_f": fin(res.t_f_teorico),
        "d_f": fin(res.d_f_teorico),
        "d_real": fin(res.d_frenado_real),
        "resultado": res.resultado,
        "se_detiene": res.se_detiene,
    }
