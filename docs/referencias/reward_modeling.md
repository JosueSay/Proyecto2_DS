# [Reward Modeling for Human Preferences](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-82.pdf)

## Resumen

Este documento analiza cómo construir modelos de recompensa (**reward models**) para alinear modelos de lenguaje con preferencias humanas en esquemas de *Reinforcement Learning from Human Feedback* (RLHF). Propone un nuevo conjunto de evaluación llamado **Preference Proxy Evaluations (PPE)** para medir qué tan bien un modelo de recompensa predice resultados humanos después de aplicar RLHF. También explora el uso de regresión **heterocedástica** para modelar la variabilidad en preferencias humanas.

## Estructura y contenidos principales

### 1. Introducción y motivación

* Los modelos de recompensa se usan como sustitutos escalables del juicio humano durante el entrenamiento de LLMs con RLHF.
* El documento plantea que la evaluación tradicional de modelos de recompensa (basada en tareas estáticas) no se correlaciona bien con el desempeño posterior de los LLMs entrenados con esos modelos.
* La contribución principal: un benchmark PPE que está explícitamente alineado con resultados reales posteriores a RLHF, y mejores técnicas para robustecer modelos de recompensa.

### 2. Preference Proxy Evaluations (PPE)

* PPE es un conjunto de pruebas que combina datos de preferencias humanas reales con tareas donde existe una solución verificable (correcta) para medir precisión de recompensa.
* Incluye métricas como precisión en comparaciones pareadas, correlaciones de ranking (Spearman, Kendall), y separación/confianza.
* PPE permite cuantificar cuán bien un modelo de recompensa sirve como proxy antes de hacer todo el proceso costoso de RLHF.

### 3. Validación del benchmark (PPE) frente a RLHF real

* Realizan experimentos donde entrenan LLMs usando distintos modelos de recompensa y luego miden la preferencia humana sobre sus salidas. Comparan esos resultados con las métricas obtenidas en PPE.
* Encuentran correlaciones significativas entre métricas de PPE y desempeño real posterior, lo que valida la utilidad del benchmark.
* Se discuten limitaciones del enfoque.

### 4. Hacia modelos de recompensa robustos

* Proponen usar **regresión heterocedástica**: además de predecir un puntaje medio, el modelo estima una varianza local, para capturar incertidumbre y ruido humano.
* La varianza estimada permite generar una recompensa “cuantílica pesimista” (usar puntaje medio – factor × desviación estándar) como señal de entrenamiento más robusta ante muestras atípicas o fuera de distribución.
* Comparan variantes: varianza fija, sin varianza y varianza aprendida, hallando que la versión con varianza aprendida mejora desempeño.

### 5. Conclusiones y direcciones futuras

* Los modelos de recompensa que estiman varianza son más resistentes a ruido y mejoran la correlación con preferencias humanas reales.
* PPE proporciona una manera eficiente de evaluar modelos de recompensa sin necesidad de desplegar RLHF completo.
* Abren camino a investigar mejores formas de manejar incertidumbre, sesgos y escalabilidad en modelos de recompensa.
