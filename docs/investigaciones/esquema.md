# Esquema

## Definiciones útiles

### Principios

- **Doblar los datos invirtiendo A/B (para neutralizar el sesgo de posición).**
  La misma conversación se presenta dos veces: una con "Respuesta A vs B" y otra con "B vs A". Así el modelo aprende a fijarse en la **calidad** y no en quién va primero. En inferencia se procesan ambos órdenes y se **promedian** las probabilidades por `id`, estabilizando el resultado.

- **Respetar un "presupuesto de tokens" justo (truncado equilibrado).**
  Para que quepan prompt y dos respuestas, se recortan longitudes (ej. \~256 para prompt, \~700 por respuesta) y, si sobra texto, se recorta **un poco más** la respuesta más larga hasta que ambas quepan. Esto evita favorecer sistemáticamente a la respuesta corta o larga y reduce errores por cortes bruscos.

- **Validación robusta (k-fold estratificado y, cuando aplique, agrupar por prompt).**
  Dividir en 5 folds **manteniendo la proporción** de clases (A/B/empate) y evitando que prompts casi iguales caigan en train y valid a la vez, da una métrica offline confiable y cercana a leaderboard; así detectas sobreajuste por "memorizar" conversaciones.

- **Ensamblar salidas (combinar "logits/probabilidades" de varios modelos).**
  Distintos modelos suelen acertar en **casos distintos**; combinarlos reduce errores sistemáticos. Un script típico une salidas (p. ej., de Gemma, Qwen, Llama) por un identificador común y guarda el conjunto para la predicción final.

- **Entrenar y servir de forma eficiente (LoRA + cuantización).**
  Para no "reentrenar todo" un LLM grande ni requerir GPUs enormes, se ajustan **pocas capas** (LoRA) y se usa **bajo número de bits** al cargar pesos (4-bit/8-bit). Mantiene rendimiento competitivo con menos memoria y costo, facilitando más experimentos y mejor ciencia.

- **Medir lo que importa: probabilidades bien calibradas (log loss).**
  No basta con "acertar"; hay que **asignar probabilidades realistas** a A/B/empate. Durante validación se calculan probabilidades con softmax y se evalúa **log loss**, que castiga dar baja probabilidad a la clase correcta; esto alinea el trabajo offline con la métrica oficial.

### Conceptos

- **Modelo de preferencia / reward model (RLHF):** Aprende a puntuar qué respuesta prefieren los humanos en comparaciones A/B. *Ej.: Dado un prompt y dos respuestas, asigna mayor "preferencia" a la más elegida.*
- **Sesgo de posición:** Tendencia a elegir la primera respuesta por aparecer primero. *Ej.: Muchos votan "A" solo por verla antes que "B".*
- **Sesgo de verbosidad:** Favorecer respuestas largas aunque no sean mejores. *Ej.: Elegir un texto extenso con relleno frente a uno breve y preciso.*
- **Sesgo de auto-promoción:** Preferir respuestas que se autopromocionan sin sustento. *Ej.: "Soy el modelo más avanzado, por eso tengo razón".*
- **Log loss multiclase (calibración):** Penaliza fuerte dar baja probabilidad a la clase verdadera; requiere probabilidades calibradas. *Ej.: Verdad=B, pero P(B)=0.1 y P(A)=0.9 -> alto castigo.*
- **Clases objetivo (winner_model_a, winner_model_b, winner_tie):** Etiquetas mutuamente excluyentes; se predicen probabilidades que suman 1. *Ej.: (A: 0.60, B: 0.25, Tie: 0.15).*

## Análisis previo

- **Variables de entrada (train/test):**

  - `prompt` (texto del usuario).
  - `response_a`, `response_b` (respuestas de dos LLMs).
  - `model_a`, `model_b` **solo en `train`** (identidad de los modelos; no está en `test`).
  - `id` (identificador de la interacción).
  - Nota: en `test` no hay columnas de ganador; en `train` sí están como targets.

- **Variable(s) objetivo y codificación:**

  - Objetivo multiclase: `winner_model_a`, `winner_model_b`, `winner_tie`.
  - En `train`, aparecen como **binarias mutuamente excluyentes** (one-hot de la clase ganadora A/B/Empate).
  - En predicción (submission), se requieren **probabilidades** para cada clase por `id`.

- **Suposiciones y riesgos de fuga de información:**

  - **Uso de `model_[a/b]`:** disponible en `train` pero **no** en `test`; modelar directamente sobre estas columnas puede no generalizar y provocar **leakage** si se infiere identidad indirectamente.
  - **Patrones de longitud/posición:** sesgos de verbosidad o de orden (A vs. B) pueden correlacionarse con el ganador; si se explotan sin control, el modelo "aprende el sesgo" más que la preferencia real.
  - **Duplicados/near-duplicates:** si prompts o respuestas repetidas aparecen entre `train` y `test`, existe riesgo de **contaminación** inadvertida.
  - **Metadatos implícitos:** no usar heurísticas basadas en `id`, formatos de archivo o artefactos del muestreo que no representan el mundo real.
  - **Distribuciones distintas:** `test` carece de `model_[a/b]` y puede tener mezcla de modelos diferente a `train`; asumir la misma mezcla puede inducir error.

- **Preguntas EDA clave (3–6):**

  - ¿Cómo está el **balance de clases** (A/B/Empate) y varía por tipo de prompt?
  - ¿Distribuciones de **longitud** de `response_a` y `response_b` (tokens/caracteres) y su relación con la clase ganadora (sesgo de verbosidad)?
  - ¿Evidencia de **sesgo de posición**? (tasa de victoria de A vs. B a igualdad de condiciones).
  - ¿Idiomas/temas predominantes en `prompt` y su relación con el ganador? (diversidad y posibles sesgos).
  - ¿Duplicados o **alta similitud** entre prompts/respuestas dentro de `train` y entre `train`–`test`?
  - ¿Tasa de **valores atípicos** o respuestas anómalas (muy cortas/largas, vacías, "ambas malas") y cómo afectan la clase "empate"?

## Formulación del problema

**Problema (ML):** Clasificación probabilística multiclase con **3 clases** (A, B, tie).
**Entrada:** textos `prompt`, `response_a`, `response_b`.
**Salida:** probabilidades **P(A)**, **P(B)**, **P(tie)** que sumen 1 para cada `id`.

**Restricciones:**

- Debe **generalizar a test** sin usar `model_a/model_b` (no disponibles).
- Minimizar **sesgo de posición** (no favorecer sistemáticamente a A o B por orden).
- **Manejar empates** asignando probabilidad no trivial a "tie" cuando ambas respuestas sean comparables en calidad.
- Cumplir con evaluación por **log loss multiclase** (enfoque en calibración de probabilidades).

**Hipótesis (texto -> preferencia):**

1. **Calidad y cobertura**: respuestas que atienden todas las partes del prompt, con precisión y coherencia, aumentan su probabilidad de ser preferidas.
2. **Estilo y utilidad percibida**: claridad, estructura (listas, pasos), tono adecuado y ausencia de relleno incrementan preferencia frente a textos vagos o redundantes.
3. **Corrección factual y especificidad**: afirmaciones verificables, ejemplos concretos y ausencia de errores factuales o contradicciones elevan la probabilidad de victoria; cuando ambas cumplen de manera similar, crece P(tie).

## Métricas

- **Métrica principal: log loss multiclase.** Evalúa la **calidad probabilística** de las predicciones (penaliza asignar baja probabilidad a la clase verdadera) y coincide con la métrica de la leaderboard, por lo que es coherente para comparar resultados offline/online.
- **Esquema de validación:** **K-fold estratificado por clase** (A/B/tie) para mantener el balance de clases en cada fold y estimar varianza de desempeño.
- **Prevención de fuga entre folds:** si existen prompts repetidos o muy similares, aplicar **Group K-Fold por `prompt`** (o hash/cluster de prompt) para que instancias relacionadas no queden en train y valid simultáneamente.
- **Desbalance/rareza de "tie":** asegurar estratificación adecuada; si la clase "tie" es escasa, verificar que cada fold conserve su proporción mínima.
- **Calibración de probabilidades:** **verificar calibración** (p. ej., diagramas de confiabilidad/Brier) porque la leaderboard premia probabilidades bien calibradas; una buena exactitud sin calibración puede seguir dando **log loss** alto.
- **Reporte por fold:** para cada fold, informar **log loss**, distribución de clases y, opcionalmente, métricas auxiliares (accuracy, ECE) solo como referencia.
- **Reporte agregado:** media y desviación estándar del **log loss** en K folds; incluir intervalo de confianza simple para comunicar la **incertidumbre** del desempeño.

## Riesgos

| Riesgo                              | Señal temprana                                  | Mitigación en 1 línea                                                                                 |
| ----------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Contenido ofensivo en textos        | EDA muestra términos tóxicos o prompts marcados | Filtrar/mascarar con listas y reglas; excluir casos extremos del entrenamiento/evaluación.            |
| Desbalance de clases (empates)      | Distribución A/B/tie muy desigual por fold      | Estratificar por clase; usar métricas probabilísticas; ajustar splits para preservar proporciones.    |
| Sesgo de posición                   | Tasa de victoria de A > B constante sin razón   | Aleatorizar orden en validación y análisis; monitorear diferencia A vs. B por prompt.                 |
| Sesgo de verbosidad                 | Respuestas largas ganan sistemáticamente        | Controlar por longitud en EDA/reportes; normalizar expectativas y analizar por cuartiles de longitud. |
| Data leakage (model_\[a/b])        | Validación offline muy alta pero cae en test    | No usar `model_[a/b]`; revisar features derivadas que revelen identidad indirecta.                    |
| Overfitting a public LB             | Mejora en LB público sin mejora en CV           | Confiar en CV sólido (K-fold estratificado/agrupado); congelar hiperparámetros y validar estabilidad. |
| Reproducibilidad (seeds, versiones) | Resultados varían entre ejecuciones             | Fijar semillas, versiones y hashes; registrar config y artefactos; documentar entorno.                |
| Deriva entre train y test           | Cambio en longitudes/temas/idiomas entre splits | Monitorear *drift* con perfiles de datos; ajustar validación agrupada y reportar sensibilidad.        |
| Distribuciones por prompt repetido  | Leaks por duplicados/near-duplicates            | Deduplicar y agrupar por `prompt` (o hash/cluster) antes de dividir; auditar similitud.               |

<!-- ## Preguntas

1. ¿Se prioriza más la interpretabilidad de las predicciones o maximizar el desempeño en log loss multiclase?
2. ¿Qué nivel de tolerancia se espera hacia la clase "empate": tratarla como clase minoritaria inevitable o enfatizar su predicción precisa?
3. ¿El entregable debe incluir únicamente un notebook ejecutable o también un reporte escrito estructurado?
4. ¿La reproducibilidad requiere fijar semillas y versiones exactas, o basta con describir el entorno usado?
5. ¿Se evaluará la documentación del código (comentarios, claridad) como parte de la nota, o únicamente los resultados numéricos?
6. ¿Se espera análisis de sesgos y limitaciones en el reporte final, además de las métricas principales?
7. ¿Qué formato de presentación final se prefiere: resultados comparativos entre modelos o un único mejor modelo justificado? -->

## Restricciones

### Formato del archivo de envío (CSV)

- **Columnas obligatorias:**
  `id, winner_model_a, winner_model_b, winner_tie`
- **Valores:**

  - Probabilidades en el rango **\[0,1]**.
  - Cada fila debe sumar **exactamente 1** entre las tres probabilidades.
- **Encabezado:** debe estar incluido en la primera fila.

**Ejemplo válido (3 filas):**

```csv
id,winner_model_a,winner_model_b,winner_tie
1001,0.70,0.20,0.10
1002,0.25,0.50,0.25
1003,0.33,0.33,0.34
```

### Checks automáticos antes de enviar

1. **Suma $\approx$ 1 por fila:** verificar que `winner_model_a + winner_model_b + winner_tie` sea 1 (con tolerancia numérica mínima).
2. **Sin valores faltantes o inválidos:** no debe haber `NaN`, vacíos ni valores fuera de \[0,1].
3. **Orden de ids correcto:** la columna `id` debe estar ordenada y cubrir exactamente los ids del archivo de test.
