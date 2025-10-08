# [BERT, RoBERTa or DeBERTa? Comparing Performance Across Transformers Models in Political Science Text](https://svallejovera.github.io/files/bert_roberta_jop.pdf)

## Resumen

Este artículo compara el desempeño de **BERT, RoBERTa y DeBERTa** para tareas de clasificación de texto en ciencias políticas (incluyendo contextos multilíngües). Muestra que RoBERTa y DeBERTa superan consistentemente a BERT, especialmente cuando se hace un entrenamiento adicional (“further training”) en textos especializados. En escenarios multilingües, XLM-RoBERTa destaca frente a mBERT y mDeBERTa.

## Metodología y diseño experimental

* Se evalúan los modelos sobre **tres aplicaciones distintas** de texto político:

  1. Clasificación de tweets como “civil” o “incivil” (inglés).
  2. Clasificación multilingüe de discursos usando 29 idiomas (Global Populism Database).
  3. Clasificación de noticias COVID (verdaderas vs falsas) después de hacer *further training* con textos recientes.

* Se comparan también con modelos basales clásicos: **SVM con TF-IDF** y **Bi-LSTM con embeddings GloVe**.

* Para cada modelo transformer, se realiza *fine-tuning* estándar (ajuste sobre datos etiquetados) y, además, *further training* con textos no etiquetados del dominio específico.

* Uso de **validación cruzada repetida** (10-fold CV repetido múltiples veces) para estimar robustamente el desempeño.

* Se reportan métricas como **F1** macro, precisión, recall, y desviaciones estándar.

## Resultados principales

* **RoBERTa-large** obtuvo el mejor desempeño balanceado: mejor F1 macro en la mayoría de las tareas.
* **DeBERTa-v3-large** rindió muy cerca de RoBERTa, pero con mayor costo computacional.
* **BERT** quedó rezagado especialmente en tareas más exigentes o cuando el texto es complejo (por ejemplo, clasificación incivil).
* En la tarea de multilíngüe, **XLM-RoBERTa** superó a mBERT y mDeBERTa en muchos idiomas.
* El *further training* (pre-entrenar con textos del dominio) mejora sustancialmente los resultados sobre modelos base.

## Ventajas y limitaciones identificadas

**Ventajas:**

* Permite comparar directamente qué modelo transformer generaliza mejor en textos políticos.
* Demuestra que ajustar más allá del fine-tuning puede ayudar en dominios especializados.
* Usa comparación con modelos clásicos para mostrar la ganancia real del enfoque de transformers.

**Limitaciones:**

* El costo computacional de DeBERTa es más alto, lo que puede limitar su uso práctico.
* Los modelos transformers están restringidos al máximo de tokens (e.g. 512 en muchos casos), lo que puede limitar su capacidad con textos largos.
* El dominio específico del artículo es ciencias políticas; los resultados podrían variar para otros dominios o tipos de texto.
