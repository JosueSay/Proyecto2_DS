# [LLM-BLENDER: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion](https://aclanthology.org/2023.acl-long.792.pdf)

## Resumen general

Este trabajo propone **LLM-BLENDER**, un marco para combinar (ensemble) múltiples modelos de lenguaje (LLMs) aprovechando sus fortalezas individuales. Usa dos módulos centrales:

* **PAIRRANKER**, para comparar candidatos por pares con un encoder cruzado (cross-attention) y seleccionar los mejores.
* **GENFUSER**, para fusionar los mejores candidatos generados en una respuesta final mejorada.
  Demuestran mejoras sustanciales frente a modelos individuales en su nuevo benchmark **MixInstruct**.

## Contenidos clave

### 1. Motivación y planteamiento

* No existe un LLM que siempre sea el mejor para todo tipo de entrada, ya que cada modelo tiene puntos fuertes y débiles distintos.
* Por ello, en lugar de elegir un modelo fijo, proponen seleccionar dinámicamente entre las salidas de varios LLMs y fusionar las mejores opciones.
* **PAIRRANKER** permite discernir diferencias sutiles entre las salidas de modelos que son muy similares.

### 2. PAIRRANKER: ranking por pares

* Dado un input ( $x$ ) y un conjunto de candidatos ($y_i$), construyen pares ($y_i$, $y_j$) y procesan ($x; y_i; y_j$) con un transformer de *cross-attention*.
* El modelo produce probabilidades de que ($y_i$) sea mejor que ($y_j$) (y viceversa).
* Para decidir el orden final entre los candidatos, usan funciones de agregación como **MaxLogits** (suma de probabilidades) o **MaxWins** (conteo de victorias en comparaciones).
* Para eficiencia en inferencia, también proponen usar un algoritmo tipo *bubble sort* basado en comparaciones en vez de calcular todas las posibles ($O(N^2)$) comparaciones.

### 3. GENFUSER: fusión generativa

* Después de que PAIRRANKER selecciona los top-K candidatos, GENFUSER los concatena con el input y los pasa a un modelo *seq2seq* (como Flan-T5) para generar una salida combinada optimizada.
* La idea es extraer los puntos fuertes de cada candidato y mitigar sus debilidades.

### 4. Dataset MixInstruct y evaluación

* **MixInstruct**: un nuevo benchmark con 110 000 ejemplos divididos en train/dev/test; genera candidatos con N = 11 LLMs y obtiene comparaciones “oracle” con ChatGPT.
* Usan métricas automáticas (BERTScore, BLEURT, BARTScore) y una métrica de ranking basada en GPT (*GPT-Rank*) para evaluar qué tan bien coincide su ranking con el juicio humano.
* Resultados: PAIRRANKER supera métodos tradicionales (MLM-Scoring, SimCLS, SummaReranker). LLM-BLENDER (PAIRRANKER + GENFUSER) obtiene las mejores métricas.

### 5. Arquitectura y detalles técnicos

* Para PAIRRANKER utilizan **DeBERTa (400M)** como *backbone*.
* En el entrenamiento: muestreo de pares (no comparar todas las combinaciones) para reducir costos.
* Durante inferencia: agregación de resultados de comparaciones en una matriz y aplicación de agregadores (MaxLogits es el preferido) para ranking final.
* En GENFUSER emplean Flan-T5-XL (~3B parámetros) para fusionar candidatos en respuestas mejores.

### 6. Limitaciones y futuras direcciones

* La eficiencia es un reto: PAIRRANKER puede requerir muchas comparaciones. Se propone paralelismo y métodos de ordenamiento más ligeros.
* Evaluación humana es limitada actualmente; usar ChatGPT como juez es un reemplazo práctico, pero no perfecto.
* Futuras mejoras podrían incluir nuevos módulos de ranking/fusión, extensión a otros tipos de modelos o modalidades (imágenes, audio), y uso de aprendizaje activo para adaptarse mejor.
