# Guía de configuración YAML

Este documento describe **todas las claves del archivo de configuración YAML**, sus valores válidos, equivalencias y propósito dentro del sistema de entrenamiento de modelos de preferencias.

## 1. `model`

Define el modelo base y opciones de ejecución.

| Clave                | Tipo    | Posibles valores         | Descripción                    |
| -------------------- | ------- | ------------------------ | ------------------------------ |
| `name`               | `str`   | libre (informativo)      | Nombre corto del backbone.     |
| `pretrained_name`    | `str`   | identificador HF válido  | Modelo en Hugging Face.        |
| `dropout`            | `float` | 0.0–1.0                  | Probabilidad de dropout.       |
| `grad_checkpointing` | `bool`  | `true` \| `false`        | Activa gradient checkpointing. |
| `compile`            | `bool`  | `true` \| `false`        | Usa `torch.compile()`.         |

## 2. `lengths`

Controla la longitud máxima de entrada.

| Clave            | Tipo  | Posibles valores | Descripción                |
| ---------------- | ----- | ---------------- | -------------------------- |
| `max_len_prompt` | `int` | >0               | Tokens máx. del prompt.    |
| `max_len_resp`   | `int` | >0               | Tokens máx. por respuesta. |

## 3. `train`

Parámetros de entrenamiento del modelo.

| Clave          | Tipo    | Posibles valores            | Descripción                    |
| -------------- | ------- | --------------------------- | ------------------------------ |
| `epochs`       | `int`   | >0                          | Épocas totales.                |
| `batch_size`   | `int`   | >0                          | Tamaño de batch.               |
| `grad_accum`   | `int`   | ≥1                          | Acumulación de gradiente.      |
| `lr`           | `float` | 1e-6–1e-3                   | Tasa de aprendizaje.           |
| `weight_decay` | `float` | 0.0–0.1                     | Decaimiento de pesos.          |
| `warmup_ratio` | `float` | 0.0–1.0                     | Proporción de warmup.          |
| `scheduler`    | `str`   | `cosine` \| `linear`        | Tipo de scheduler.             |
| `clip_norm`    | `float` | >0                          | Límite de norma del gradiente. |
| `seed`         | `int`   | cualquier entero            | Semilla.                       |
| `amp`          | `str`   | `bf16` \| `fp16` \| `false` | Precisión mixta.               |
| `num_workers`  | `int`   | ≥0                          | Workers del DataLoader.        |

## 4. `data`

Configura los archivos de datos y columnas.

| Clave            | Tipo   | Posibles valores   | Descripción              |
| ---------------- | ------ | ------------------ | ------------------------ |
| `train_csv`      | `str`  | ruta válida        | CSV de entrenamiento.    |
| `valid_csv`      | `str`  | ruta válida        | CSV de validación.       |
| `use_label`      | `bool` | `true` \| `false`  | Usa columna de etiqueta. |
| `use_clean_cols` | `bool` | `true` \| `false`  | Usa columnas “clean”.    |
| `shuffle`        | `bool` | `true` \| `false`  | Barajar entrenamiento.   |
| `val_batch_size` | `int`  | >0                 | Batch de validación.     |
| `pin_memory`     | `bool` | `true` \| `false`  | Pinned memory.           |
| `prompt_col`     | `str`  | nombre de columna  | Columna del prompt.      |
| `respA_col`      | `str`  | nombre de columna  | Columna respuesta A.     |
| `respB_col`      | `str`  | nombre de columna  | Columna respuesta B.     |

> `test_csv` no forma parte del esquema.

## 5. `dataloader`

| Clave                   | Tipo   | Posibles valores  | Descripción           |
| ----------------------- | ------ | ----------------- | --------------------- |
| `prefetch_factor_train` | `int`  | ≥1                | Prefetch en train.    |
| `prefetch_factor_val`   | `int`  | ≥1                | Prefetch en valid.    |
| `persistent_workers`    | `bool` | `true` \| `false` | Workers persistentes. |

## 6. `loss`

Define el tipo de pérdida y sus pesos.

| Clave             | Tipo          | Posibles valores                   | Descripción                |
| ----------------- | ------------- | ---------------------------------- | -------------------------- |
| `type`            | `str`         | `cross_entropy` \| `bradley_terry` | Función de pérdida.        |
| `label_smoothing` | `float`       | 0.0–1.0                            | Solo para `cross_entropy`. |
| `class_weights`   | `list[float]` | lista de 3 positivos               | Pesos `[A, B, TIE]`.       |

## 7. `eval`

Controla parámetros de inferencia o métricas.

| Clave       | Tipo    | Posibles valores | Descripción                |
| ----------- | ------- | ---------------- | -------------------------- |
| `bt_temp`   | `float` | 0.1–2.0          | Temperatura sigmoide (BT). |
| `tie_tau`   | `float` | 0.0–1.0          | Escala de empate.          |
| `tie_alpha` | `float` | 0.0–1.0          | Máximo de prob. de empate. |

## 8. `logging`

Controla checkpoints y detección de fallos.

| Clave                    | Tipo  | Posibles valores   | Descripción                    |
| ------------------------ | ----- | ------------------ | ------------------------------ |
| `reports_dir`            | `str` | ruta válida        | Carpeta de reportes.           |
| `runs_dir`               | `str` | ruta válida        | Carpeta de checkpoints.        |
| `step_csv`               | `str` | nombre archivo     | Registro por pasos.            |
| `epoch_csv`              | `str` | nombre archivo     | Registro por épocas.           |
| `alerts_csv`             | `str` | nombre archivo     | Alertas.                       |
| `confusion_csv`          | `str` | nombre archivo     | Matriz de confusión.           |
| `class_report_csv`       | `str` | nombre archivo     | Reporte de clases.             |
| `preds_sample_csv`       | `str` | nombre archivo     | Muestras de predicciones.      |
| `pred_distributions_csv` | `str` | nombre archivo     | Distribución de predicciones.  |
| `val_pred_tpl`           | `str` | plantilla con `{}` | Archivo de pred. por época.    |
| `token_budget_tpl`       | `str` | plantilla con `{}` | Archivo de tokens por época.   |
| `run_config_used`        | `str` | nombre archivo     | Dump de la config usada.       |
| `step_interval`          | `int` | ≥1                 | Frecuencia de logging (pasos). |

## 9. `monitor`

| Clave             | Tipo   | Posibles valores                       | Descripción                    |
| ----------------- | ------ | -------------------------------------- | ------------------------------ |
| `detect_collapse` | `bool` | `true` \| `false`                      | Detecta colapso de salida.     |
| `save_best_by`    | `str`  | `macro_f1` \| `accuracy` \| `val_loss` | Métrica para mejor checkpoint. |
| `save_last`       | `bool` | `true` \| `false`                      | Guarda último checkpoint.      |
| `verbose`         | `bool` | `true` \| `false`                      | Verbosidad.                    |

## 10. `early_stopping`

| Clave      | Tipo  | Posibles valores                       | Descripción               |
| ---------- | ----- | -------------------------------------- | ------------------------- |
| `metric`   | `str` | `macro_f1` \| `val_loss` \| `accuracy` | Métrica monitoreada.      |
| `mode`     | `str` | `max` \| `min`                         | Dirección de comparación. |
| `patience` | `int` | ≥0                                     | Épocas sin mejora.        |

## 11. `env`

Variables de entorno y rendimiento.

| Clave                     | Tipo   | Posibles valores             | Descripción                  |
| ------------------------- | ------ | ---------------------------- | ---------------------------- |
| `tokenizers_parallelism`  | `bool` | `true` \| `false`            | Paralelismo del tokenizador. |
| `cuda_launch_blocking`    | `int`  | `0` \| `1`                   | Sincronización CUDA.         |
| `pytorch_cuda_alloc_conf` | `str`  | `"max_split_size_mb:<int>"`  | Partición de memoria CUDA.   |
| `hf_home`                 | `str`  | ruta válida                  | Caché local de HF.           |
| `use_slow_tokenizer`      | `bool` | `true` \| `false`            | Tokenizador “slow”.          |
