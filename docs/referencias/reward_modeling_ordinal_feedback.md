
# [Reward Modeling with Ordinal Feedback: Wisdom of the Crowd](https://arxiv.org/pdf/2411.12843)

## Resumen general

Este trabajo propone un marco para construir modelos de recompensa (Reward Models, RMs) usando **retroalimentación ordinal** (no solo “mejor/peor”, sino niveles intermedios y empates). Argumenta que esto aprovecha más información humana útil en comparación con el esquema binario típico.

Los autores formalizan una suposición llamada *marginal unbiasedness* (sabiduría de la multitud), que generaliza la suposición del modelo Bradley–Terry al caso ordinal. Bajo esa hipótesis, derivan un modelo probabilístico para las preferencias ordinales y muestran teóricamente que la retroalimentación más fina reduce la complejidad estadística (Rademacher complexity) frente al caso binario. También extienden la teoría a pérdidas tipo *hinge* y optimización directa de política (DPO).

En experimentos, muestran que usar retroalimentación de múltiples niveles (por ejemplo, 3 o 5 niveles) mejora el aprendizaje del modelo de recompensa, incluso cuando se mezclan muestras con etiquetas “empate”.

## Estructura detallada

### 1. Planteamiento del problema & motivación

* Tradicionalmente, los modelos de preferencia usan feedback **binario** (“A es mejor que B” o viceversa), basado en Bradley–Terry.
* El feedback ordinal permite etiquetas como “mejor”, “ligeramente mejor”, “igual”, “ligeramente peor”, etc.
* Muchos proyectos recolectan feedback más fino (por ejemplo, equipo de Llama solicita 4 niveles), pero luego reducen todo a binario, perdiendo información.

### 2. Suposición *marginal unbiasedness* (sabiduría de la multitud)

* Se define el feedback ordinal ( $Z \in {z_1, ..., z_m}$ ), con valores numéricos en ($[0,1]$).
* La suposición es:
  $$
  \mathbb{E}[Z \mid (x, y_1, y_2)] = P(y_1 \succ y_2 \mid x)
  $$
  Esto extiende la suposición implícita del modelo binario al caso ordinal.
* Bajo esta suposición, construyen un modelo probabilístico para las preferencias ordinales compatible con la regla de expectativa.

### 3. Objetivo de aprendizaje / función de pérdida

* Proponen una función de pérdida generalizada para entrenamiento:

  $$
  -Z \log(\sigma(r_\theta(x, y_1) - r_\theta(x, y_2))) - (1 - Z)\log(\sigma(r_\theta(x, y_2) - r_\theta(x, y_1)))
  $$

  donde ($\sigma$) es la sigmoide. Cuando ($Z \in {0,1}$), esto se reduce al caso binario clásico.
* Esta fórmula permite incorporar feedback suave (por ejemplo, ($Z = 0.5$) para empate) sin descartar datos o colapsarlos en binario.
* También extienden la teoría a pérdidas tipo *hinge* y a DPO (optimización directa de política).

### 4. Beneficios estadísticos del feedback ordinal

* Demuestran que usar feedback con más granularidad reduce la **complejidad de Rademacher** del modelo (una medida de capacidad de generalización), lo cual mejora la generalización cuando los datos son limitados.
* Presentan resultados que muestran que cualquier sistema ordinal (que cumpla la suposición) está "entre" la modalidad binaria y la modalidad ideal del feedback real del oráculo.
* Proponen la noción de *hierarchical expectation* para comparar diferentes diseños de feedback ordinal.

### 5. Experimentos numéricos

* Usan datasets como *Skywork-Reward-Preference-80K* y *RewardBench* para comparar sistemas de feedback: binario, 3 niveles, 5 niveles, y “feedback oráculo”.
* Observaciones clave:

  1. A mayor granularidad de retroalimentación (más niveles), mejor desempeño in-distribution (ID) y out-of-distribution (OOD).
  2. Incluir muestras de “empate” (tied) ayuda al entrenamiento si no se descartan, y suaviza la superficie de pérdida durante el entrenamiento.
  3. Hay un punto extremo donde si todas las muestras son empate, el modelo colapsa (no aprende distinción).

### 6. Discusión y comparaciones

* Comparan su enfoque con modelos de empate como **Bradley–Terry con Ties (BTT)**. Pero argumentan que su enfoque ordinal es más natural, no introduce hiperparámetros adicionales (como el λ del BTT) y escala mejor a más niveles.
* También conectan los resultados con *knowledge distillation* y *soft labeling*, explicando cómo el feedback ordinal puede verse como una forma de suavizar etiquetas y reducir varianza en el aprendizaje.
