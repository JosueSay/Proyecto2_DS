
# [Generative Reward Modeling via Synthetic Criteria Preference Learning](https://aclanthology.org/2025.acl-long.1297.pdf)

## Resumen

Este artículo propone **SyncPL (Synthetic Criteria Preference Learning)**, un método para construir modelos de recompensa generativos basados en criterios sintéticos, con el fin de reducir la dependencia de datos humanos etiquetados directamente.

SyncPL introduce una estructura de árbol de preferencias basada en criterios: cada camino en el árbol representa una trayectoria de razonamiento (**Chain-of-Thought**, CoT) con criterios, rúbricas y juicios intermedios. Estas trayectorias son optimizadas usando señales de recompensa a nivel de proceso, no solo el resultado final, permitiendo ajustar subrecompensas (sub-rewards) derivadas del razonamiento intermedio.

## Estructura del método

### 1. Motivación y antecedentes

* Los modelos discriminativos tradicionales (e.g. Bradley–Terry) modelan preferencias entre A y B de forma implícita, lo que puede generar *overoptimization* (aprender correlaciones espurias).
* Los modelos generativos de recompensa (GenRM) permiten razonamiento con CoT, pero tienen el riesgo de que las trayectorias generadas no sean confiables o coherentes.
* SyncPL pretende combinar lo mejor: generar trayectorias razonadas (criterios sintéticos) y supervisarlas parcialmente para evitar errores acumulativos.

### 2. Composición del árbol de preferencias por criterios

* Se define un conjunto de criterios generados sintéticamente (por el modelo) para juzgar respuestas A vs. B.
* Cada trayectoria (camino del árbol) consta de: criterio → rúbrica → juicio (evaluaciones de A y B).
* Se usan dos reglas para priorizar trayectorias:

  * **Ranking Rule:** da prioridad a criterios con mayor margen de diferencia entre A y B.
  * **Consistency Rule:** descarta trayectorias con juicios inconsistentes dentro del subárbol (votación mayoritaria).

### 3. Optimización del modelo (aprendizaje)

SyncPL combina varias técnicas para entrenar el modelo generativo de recompensa:

* **RSFT (Rejection Sampling Fine-Tuning):** usa trayectorias aceptadas como ejemplos positivos.
* **DPO (Direct Preference Optimization):** aplica pérdida tipo pairwise entre trayectorias elegidas / rechazadas.
* Además, el artículo explora el uso de formato de **CoT largo** (long CoT) para integrar múltiples criterios en una sola inferencia (“o1-like”).

### 4. Experimentos y resultados

* Evalúan en múltiples benchmarks de preferencias humanas: *RewardBench*, *Auto-J*, *MT-Bench*.
* SyncPL-DPO y SyncPL-o1 superan modelos base generativos y discriminativos en consistencia con decisiones humanas.
* En tareas fuera de distribución (OOD) también muestran robustez frente a otros métodos.
* La incorporación de múltiples criterios mejora notablemente la precisión de juicio frente a usar un solo criterio.

### 5. Limitaciones reconocidas

* El espacio de criterios sintéticos es limitado y depende de la capacidad del modelo para generar criterios adecuados.
* El costo de inferencia puede aumentar al usar múltiples criterios o CoT largos.
* Existe riesgo de propagación de errores en las trayectorias de razonamiento si no se supervisan bien (compounding errors).
