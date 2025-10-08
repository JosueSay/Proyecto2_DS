# [Brief Review — DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://sh-tsang.medium.com/brief-review-deberta-decoding-enhanced-bert-with-disentangled-attention-f5cdb9a8bf0b)

## Descripción general

Este artículo es una reseña de **DeBERTa**, una variante de BERT / RoBERTa propuesta por Microsoft con dos mejoras clave: atención “disentangled” y posicionamiento absoluto en la capa de decodificación. Se enfoca en explicar los principios del modelo, su entrenamiento y los resultados experimentales.

## Principales componentes del modelo DeBERTa

### 1. Mecanismo de atención disentangled (desentrelazada)

* En lugar de representar cada token con una sola vector, DeBERTa lo representa con dos vectores: uno para contenido y otro para posición relativa.
* La atención entre tokens i y j se descompone en múltiples términos (contenido ↔ contenido, contenido ↔ posición, posición ↔ contenido, posición ↔ posición). Esto permite que la posición relativa afecte la atención de forma más rica.
* Se implementa de forma eficiente para no incrementar demasiado la memoria requerida.

### 2. Incorporación de posición absoluta en la capa de decodificación

* DeBERTa añade información de posición absoluta justo antes de la capa softmax en el decodificador de máscaras (mask decoder).
* Esto mejora cómo el modelo distingue tokens con la misma apariencia pero en posiciones diferentes (por ejemplo en tareas de *masked language modeling*).

### 3. Fine-tuning con regularización: SiFT (Scale-invariant Fine-Tuning)

* Introducen una versión de entrenamiento adversarial (virtual adversarial training) como método de regularización para mejorar la generalización.
* Dado que embeddings de palabras pueden tener diferentes magnitudes, DeBERTa normaliza estas representaciones antes de aplicar perturbaciones (esto estabiliza el entrenamiento).

## Resultados y estudios ablation

* En tareas de **GLUE**, **SuperGLUE** y otras benchmarks estándar, DeBERTa supera consistentemente a BERT, RoBERTa y otras arquitecturas competidoras.
* En estudios de ablación, eliminar cualquiera de los componentes clave (atención disentangled, posición absoluta, etc.) causa una caída significativa en desempeño.
* Una versión grande (1.5 mil millones de parámetros) de DeBERTa logró superar el rendimiento humano en SuperGLUE.

## Conexión con tu problema de ranking / preferencia

* DeBERTa proporciona una arquitectura poderosa como encoder (modelo de representación) gracias a su mecanismo de atención más expresivo.
* Puedes usar DeBERTa como **modelo base** para tareas de *preference ranking*, concatenando pares (respuesta A / respuesta B) y luego proyectando a puntuaciones de preferencia usando capas adicionales.
* Su superioridad en tareas de comprensión del lenguaje lo convierte en buen candidato como backbone para tu modelo de preferencia con *pairwise loss*.
