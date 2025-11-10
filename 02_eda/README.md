# EDA

Explora y audita el dataset ya limpio para entender su distribución, sesgos, truncado y duplicados. Produce **tablas (CSV)**, **gráficas (PNG)** y **metadatos** organizados por carpetas. Usa caché para no repetir trabajo.  

## Entradas y salidas clave

### Entradas

El módulo `02_eda` toma como punto de partida los **resultados generados por `01_data_cleaning`**, ubicados principalmente en las carpetas `reports/clean/` y `data/clean/`.
Entre los archivos más importantes que utiliza se incluyen:

- `class_balance_detail.csv`: contiene el balance de clases por conjunto (*train* y *valid*).
- `truncation_impact_train_valid.csv` y `truncation_impact_train_valid_real.csv`: muestran el porcentaje de truncado teórico y real, respectivamente.
- `ab_similarity_train.csv` y `ab_similarity_valid.csv`: resumen las similitudes entre respuestas A y B después de la limpieza.
- `near_duplicates_ab.csv`: lista de pares casi duplicados.
- `length_by_class_train.csv` y `length_by_class_valid.csv`: estadísticas de longitudes por clase.
- `train_strat.csv` y `valid_strat.csv`: datasets finales de entrenamiento y validación.
- `split_meta.json`: metadatos del proceso de división de datos (split).

Estos archivos son leídos mediante funciones seguras que verifican su existencia y formato antes de procesarlos.
Si algún archivo está ausente o tiene columnas inesperadas, el sistema lo reporta en el log de análisis para facilitar la depuración.

### Salidas

El módulo genera tres tipos principales de resultados, que constituyen la evidencia del análisis exploratorio y comparativo:

1. **Archivos CSV**
   Se guardan en la carpeta `reports/eda/` e incluyen tablas numéricas y tableros de resumen, como `before_after_board.csv` o las tablas calculadas en `eda_orchestrator.py`.
   Estas tablas permiten cuantificar cambios en el dataset, comparar etapas del flujo y servir de entrada para reportes más amplios.

2. **Gráficas e imágenes**
   Se almacenan en la carpeta `images/eda/`.
   Representan visualmente las principales métricas y comparaciones, tales como la distribución de etiquetas, longitudes, correlaciones, similitud entre respuestas, truncado y fuga de prompts.
   Son útiles para interpretar tendencias y validar visualmente los efectos de la limpieza y el split.

3. **Metadatos y registros**
   En la carpeta `reports/eda/` también se generan archivos informativos como `eda_meta.json` y `00_analisis.log`.
   El primero documenta la configuración, fecha y origen de los datos analizados, mientras que el segundo resume las conclusiones y hallazgos detectados durante la ejecución del análisis.
   Estos elementos permiten mantener trazabilidad y reproducibilidad del proceso de EDA.

## Orquestadores

### `eda_orchestrator.py` — Exploración general del dataset limpio

Este orquestador realiza el **análisis exploratorio descriptivo** del dataset ya limpio, generando tablas estadísticas, visualizaciones y un resumen general de su estructura.
Su propósito es **comprender la composición del dataset**, detectar desbalances, evaluar la distribución de longitudes y examinar correlaciones entre variables antes de iniciar cualquier modelado.

#### Tablas generadas (archivos CSV en `reports/eda/`)

El módulo produce varias tablas que condensan métricas relevantes sobre los datos:

- **`eda_summary.csv`**
  Resume la estructura general del dataset, incluyendo su tamaño total, número de registros válidos y valores nulos por columna.
  Es la primera verificación de integridad y sirve para confirmar que los datos fueron cargados correctamente.

- **`class_balance.csv`**
  Indica la cantidad de ejemplos por cada etiqueta (`label`).
  Permite identificar si existe un desbalance entre las clases (por ejemplo, una respuesta A ganando con más frecuencia que B).

- **`feature_stats.csv`**
  Presenta estadísticas descriptivas de las variables numéricas, como las longitudes de prompts y respuestas.
  Facilita la detección de valores atípicos y la comprensión de los rangos típicos de cada variable, ayudando a definir estrategias de truncado o normalización.

- **`correlations.csv`**
  Contiene la matriz de correlaciones entre las variables numéricas del dataset.
  Su objetivo es descubrir relaciones lineales fuertes o redundancias que puedan simplificar el conjunto de características o revelar sesgos.

- **`model_wins.csv`**
  Registra cuántas veces cada modelo (si existe esta información) generó una respuesta ganadora.
  Es útil para analizar la influencia de diferentes modelos o fuentes de generación dentro del conjunto de datos.

#### Gráficas generadas (archivos PNG en `images/eda/`)

Las gráficas creadas por este orquestador ayudan a visualizar patrones de distribución y posibles anomalías:

- **`correlation_heatmap.png`**
  Muestra un mapa de calor de las correlaciones numéricas, lo que permite identificar variables que se comportan de forma similar o redundante.

- **`distribution_label.png`**
  Representa en barras la distribución de las clases (`label`).
  Ayuda a verificar si los datos están equilibrados o si una categoría domina el conjunto.

- **`distribution_prompt_len.png`, `distribution_respA_len.png`, `distribution_respB_len.png`**
  Son histogramas que muestran la distribución de las longitudes de prompts y respuestas.
  Permiten detectar si existen textos excesivamente largos o cortos y estimar el impacto potencial del truncado.

- **`results_distribution.png`**
  Muestra la proporción de resultados A, B o empates, si las columnas de resultado lo permiten.
  Sirve para identificar sesgos globales en la calidad o preferencia de respuestas.

- **`top_models_wins.png`**
  Presenta un gráfico de barras con los modelos o fuentes que más respuestas ganadoras produjeron.
  Permite observar si hay una fuente dominante que pueda sesgar el entrenamiento posterior.

#### Metadatos generados

El archivo **`eda_meta.json`** se guarda en `reports/eda/` e incluye:

- La fecha y hora de ejecución del análisis.
- El archivo de origen (`data/clean/data_clean.csv`).
- Las rutas de salida utilizadas para tablas y gráficos.
- El listado completo de tablas generadas.

Su función es **mantener la trazabilidad** del proceso de análisis, garantizando que los resultados sean reproducibles y fácilmente auditables.

---

### `eda_analysis_orchestrator.py` — Análisis comparativo “antes vs después”

Este componente complementa el análisis general, enfocándose en una **evaluación comparativa del dataset antes y después del proceso de limpieza y división**.
Analiza cómo cambian las métricas de similitud, truncado, duplicados y fuga de prompts, con el fin de validar la efectividad de la etapa de depuración y asegurar que el split sea limpio y balanceado.

#### Registro narrativo de resultados

El archivo **`00_analisis.log`** contiene un resumen textual de todo el análisis, detallando:

- Balance de clases por conjunto (*train* y *valid*).
- Impacto del truncado esperado y real.
- Similitud entre respuestas A y B después de la limpieza.
- Conteo y proporción de near-duplicates detectados.
- Estadísticas de longitudes por clase.
- Verificación de fuga de prompts entre splits.
  Este log funciona como un **informe de auditoría**, ofreciendo una interpretación directa de las métricas sin necesidad de revisar cada archivo individualmente.

#### Tablas generadas (archivos CSV en `reports/eda/`)

- **`before_after_board.csv`**
  Es una tabla resumen que compara métricas clave del dataset **antes** y **después** de la limpieza.
  Incluye indicadores como porcentaje de pares con alta similitud (`%cos>=0.98` o `%cos>=0.995`), número de duplicados, y porcentajes de truncado en prompts y respuestas.
  Su objetivo es demostrar de forma cuantitativa si las transformaciones aplicadas realmente redujeron la redundancia y mejoraron la calidad del corpus.

#### Gráficas generadas (archivos PNG en `images/eda/`)

Las visualizaciones comparan métricas entre la versión original del dataset y la posterior al procesamiento:

- **`before_after_cosine_hist.png`**
  Histograma de los valores de similitud por coseno antes y después del filtrado.
  Se espera observar una disminución en la frecuencia de similitudes altas, indicando mayor diversidad entre respuestas.

- **`before_after_jaccard_hist.png`**
  Histograma equivalente para la métrica de similitud Jaccard.
  Permite confirmar que los pares A/B se volvieron menos redundantes.

- **`before_after_truncation_bars.png`**
  Gráfico de barras que muestra el porcentaje de truncado en prompts y respuestas.
  Facilita la evaluación del equilibrio entre limpieza y preservación de información.

- **`len_diff_by_label_train.png` y `len_diff_by_label_valid.png`**
  Comparan la longitud media de respuestas A y B en cada clase dentro de *train* y *valid*.
  Detectan posibles sesgos de longitud que podrían afectar las preferencias del modelo.

- **`tails_compare_train.png` y `tails_compare_valid.png`**
  Ilustran el porcentaje de ejemplos situados en la cola superior de longitudes (por encima de 512 tokens).
  Ayudan a identificar el riesgo de pérdida de contenido relevante por truncado.

- **`prompt_leakage_bars.png`**
  Grafica la cantidad de prompts únicos en *train* y *valid* y su intersección.
  Confirma que no exista fuga de información entre conjuntos.

- **`scatter_cosine_before.png` y `scatter_cosine_after.png`**
  Muestran la dispersión de los valores de similitud por coseno antes y después de la limpieza.
  Permiten visualizar de forma clara la reducción de pares excesivamente similares.

En conjunto, este orquestador proporciona una visión analítica completa de cómo la limpieza y división del dataset **mejoraron su calidad estructural y semántica**, ofreciendo evidencia visual y numérica de esos cambios.

## Cómo interpretar los hallazgos (guía rápida)

- **Balance de clases** (`distribution_label.png`, `class_balance.csv`): si una clase domina, ajusta muestreo/ponderaciones.  
- **Similitud A/B** (cosine/jaccard): una caída “después” indica éxito al remover redundancias y near-duplicates.
- **Truncado**: barras/colas altas sugieren subir límites, mejorar limpieza o reducir ruido.
- **Fuga de prompts**: intersección ≈ 0 valida un split sin contaminación entre *train* y *valid*.
- **Modelos** (`top_models_wins.png`, `model_wins.csv`): detecta fuentes desbalanceadas que puedan sesgar el entrenamiento.  
