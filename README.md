# Simulador 3D · Frenado de un vehículo en carreteras con pendiente

Proyecto final de **Dinámica · Ingeniería Civil · Universidad Nacional de Colombia**.

Laboratorio virtual que reproduce la maqueta del proyecto (pista inclinable de 1,80 m, carrito y patín de
fricción). Python calcula el movimiento con la segunda ley de Newton y la escena 3D (Three.js) reproduce
esas muestras: **no existe una física independiente en la animación**.

---

## 0. Uso en línea (sin instalar nada) — Streamlit Community Cloud

El proyecto está listo para publicarse gratis en https://share.streamlit.io desde un repositorio de GitHub:

* Archivo principal: `app.py` · Python 3.12 (en *Advanced settings*).
* Debe subirse **también la carpeta `.streamlit`** (contiene `config.toml`, que activa
  `enableStaticServing` para servir `static/vendor/three.module.min.js`). Si faltara, la escena intenta
  cargar Three.js desde jsDelivr/unpkg.
* La ruta de Three.js se resuelve respecto a la página de la app, por lo que funciona tanto en
  `localhost:8501` como en `https://<nombre>.streamlit.app` (que sirve la app bajo `/~/+/`).

---

## 1. Instalación en Windows (paso a paso)

1. **Instale Python** (una sola vez): https://www.python.org/downloads/ → versión 3.10 o superior.
   Durante la instalación **marque la casilla "Add python.exe to PATH"**.
2. **Descomprima** el ZIP `SIMULADOR_FRENADO_3D.zip` (clic derecho → *Extraer todo…*).
   No ejecute los archivos desde dentro del ZIP.
3. Abra la carpeta `SIMULADOR_FRENADO_3D` y haga **doble clic en `instalar.bat`**.
   Crea un entorno virtual `.venv`, instala las dependencias (necesita internet) y ejecuta las pruebas.
   Al final debe ver algo como `45 passed`.
4. Haga **doble clic en `iniciar_simulador.bat`**. Se abre el navegador en `http://localhost:8501`.
   Si no se abre solo, copie esa dirección en el navegador (Chrome, Edge o Firefox).
5. Para cerrar el simulador, cierre la ventana negra (consola).

> Si Windows muestra "Windows protegió su PC", pulse *Más información → Ejecutar de todas formas*.

### Con Visual Studio Code

1. *Archivo → Abrir carpeta…* → seleccione `SIMULADOR_FRENADO_3D`.
2. *Terminal → Nuevo terminal* y ejecute (si los .bat fallan):

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Si `py` no existe use `python` en su lugar. Si PowerShell no permite activar el entorno, use directamente
`.venv\Scripts\python.exe -m streamlit run app.py`.

Pruebas automáticas del modelo matemático:

```bat
.venv\Scripts\python.exe -m pytest pruebas -v
```

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Dependencias externas

* Paquetes de Python: `streamlit`, `plotly`, `numpy`, `pandas`, `pytest` (ver `requirements.txt`).
* **Three.js r160** viene incluido en `static/vendor/three.module.min.js` (licencia MIT), por lo que la
  escena 3D funciona **sin internet** una vez instalado. Solo si ese archivo faltara, la escena intenta
  cargarlo de `cdn.jsdelivr.net`.
* No se necesita Node.js.

---

## 2. Cómo usar el simulador

| Zona | Contenido |
|---|---|
| **Panel izquierdo** | Ángulo (0–20°, botones 3°/5°/7° de la maqueta), modelo de frenado A (μef) o B (Fb con niveles B1/B2/B3 configurables), masa, longitud de la zona de frenado, condición inicial (v₀ directa **o** suelta desde el reposo), opciones avanzadas (rodadura μr, tiempo de reacción). |
| **Centro (3D)** | Maqueta: base, pista de MDF, soporte de varillas roscadas, bisagra, guías, cinta métrica, zonas coloreadas, tira de fricción, tope acolchado, marcas de inicio / inicio de frenado / detención calculada e indicador de ángulo. Botones Iniciar, Pausar/Reanudar, Reiniciar, Simular nuevamente, barra temporal, velocidad de reproducción (0,1× a 2×), cámaras Lateral/Superior/Libre/Seguir, Mostrar/Ocultar fuerzas y etiquetas. Debajo, mini‑gráficas x(t), v(t), a(t) con marcador sincronizado (clic sobre ellas para ir a un instante). |
| **Derecha** | Estado del experimento, aceleración, v₀, t_f, d_f, uso de la zona, fuerzas (mg, N, mg sen θ, Fb), μ equivalente, ángulo crítico y botón **Guardar escenario**. |
| **Pestañas** | Gráficas 1–3 (Plotly), Gráficas 4–5 (d_f vs θ y vs μ/Fb) + efecto de la masa, Escenarios (tabla, % de variación, barras, exportar), Comparación experimental (tabla editable, CSV, errores, promedios, μef calibrado, exportar), Modelo y ecuaciones (desarrollo numérico del caso actual), Verificación. |

Controles de la cámara: arrastrar = girar · rueda = zoom · clic derecho o Shift + arrastrar = desplazar ·
barra espaciadora = pausar/reanudar.

Al cambiar cualquier parámetro, Python recalcula todo y la escena se recarga con el escenario nuevo (si
estaba reproduciendo, vuelve a empezar desde t = 0); nunca se mezclan muestras de dos escenarios. Si cambia
θ, la pista gira suavemente desde el ángulo anterior al nuevo.

---

## 3. Modelo físico (fisica.py)

Eje x paralelo a la pista, positivo en el descenso; g = 9,81 m/s².

* N = mg cos θ · mg∥ = mg sen θ
* Modelo A: Fb = μef·N → a = g (sen θ − μef cos θ) (la masa se cancela)
* Modelo B: Fb constante → a = g sen θ − Fb/m (la masa sí influye)
* Zonas sin patín (aceleración y seguridad): a = g (sen θ − μr cos θ), μr = 0 por defecto
* v = v₀ + at, x = x₀ + v₀t + ½at², v² = v₀² + 2aΔx
* Si a < 0: t_f = v₀/|a|, d_f = v₀²/(2|a|). Modelo A en descenso: se detiene solo si μef > tan θ.

Comprobación (documento guía): v₀ = 1,00 m/s, θ = 5°, μ = 0,30 → a = −2,077 m/s², t_f = 0,482 s,
d_f = 0,241 m.

### Hipótesis y simplificaciones

* H1–H5 del documento: movimiento rectilíneo, partícula, Fb constante, pendiente constante, sin aire.
  La rotación de las ruedas en 3D es **solo visual** (ángulo = x / r).
* H6: la rodadura está incluida en μef/Fb dentro de la zona de frenado; fuera de ella solo actúa μr.
  No se cuenta dos veces. La fricción del patín no se aplica en la zona de aceleración.
* μef se define respecto a la **normal total** N = mg cos θ; la precarga del patín (Np) solo determina Fb.
* Detención: se calcula el instante exacto; después v = 0 y a = 0 (sin extrapolar ni retroceder).
* Reposo estático: se toma μs = μef (o Fs,max = Fb). Es conservador porque en la realidad μs ≥ μk.
* Tope final: no frena. Si el carrito lo alcanza, la simulación termina en el contacto y se informa la
  velocidad; el choque no se modela.
* x es la posición del centro del carrito (donde está el patín). d_f es una diferencia de posiciones y no
  cambia si se mide con el frente del carrito.
* Pista total 1,80 m y zona de seguridad 0,20 m fijas; al cambiar la zona de frenado, la de aceleración
  se ajusta: L_acel = 1,60 m − L_fren. El carrito modelo mide 0,18 m.

Los valores de μef "bajo/medio/alto" (0,15 / 0,30 / 0,60) y de B1/B2/B3 (0,60 / 1,00 / 1,50 N) son
**parámetros de simulación provisionales**; deben reemplazarse por los calibrados en la maqueta.

---

## 4. Datos experimentales (CSV)

Formato (`datos/ejemplo.csv`), separador `;` o `,`, coma o punto decimal:

```
ensayo;repeticion;theta_deg;masa_kg;v0_m_s;modelo;mu_ef;Fb_N;d_exp_m;t_exp_s;observaciones
A2;1;5,0;0,350;0,95;A;0,30;;0,23;0,47;
```

* `v0_m_s`: velocidad medida al **inicio** de la zona de frenado.
* `modelo` A → use `mu_ef`; modelo B → use `Fb_N`.
* Error (%) = |sim − exp| / |exp| × 100. Si exp = 0 el error relativo no está definido: se muestra vacío
  y se reporta la diferencia absoluta.
* Las filas con el mismo `ensayo` se promedian (media ± desviación estándar).
* La columna `mu_ef_calibrado_ec15` calcula μef con la ecuación (15) a partir de v₀ y d_exp.

**Las filas de `datos/ejemplo.csv` son FICTICIAS** (marcadas así): solo muestran el formato.

---

## 5. Estructura

```
SIMULADOR_FRENADO_3D/
├── app.py                  Interfaz Streamlit (controles, resultados, pestañas)
├── fisica.py               Modelo matemático (fuente de verdad)
├── graficas.py             Figuras Plotly
├── experimental.py         Lectura de CSV y comparación con ensayos
├── escena3d.html           Maqueta 3D (Three.js) + reproducción y mini-gráficas
├── static/vendor/          three.module.min.js (r160, MIT)
├── .streamlit/config.toml  Tema y servicio de archivos estáticos
├── datos/ejemplo.csv       Plantilla de formato (valores ficticios)
├── pruebas/test_fisica.py  45 pruebas automáticas
├── requirements.txt
├── instalar.bat
└── iniciar_simulador.bat
```

## 6. Problemas frecuentes

* **"python no se reconoce"**: reinstale Python marcando *Add python.exe to PATH*.
* **La escena 3D sale en blanco**: actualice el navegador (Chrome/Edge) y verifique que WebGL esté activo
  (aceleración por hardware). Recargue la página con F5.
* **Puerto ocupado**: cierre otras ventanas del simulador o cambie `--server.port 8501` por `8502` en
  `iniciar_simulador.bat`.
* **El proxy/antivirus bloquea la instalación**: ejecute los comandos de la sección 1 manualmente.
