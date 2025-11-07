# Módulo 01_data_cleaning

Este módulo contiene el flujo de limpieza, preparación y resumen inicial de los datos utilizados en el proyecto.
Su propósito es convertir los archivos originales (`train.csv`, `test.csv`) en versiones limpias, balanceadas y listas para entrenamiento o inferencia, asegurando trazabilidad y consistencia entre ejecuciones.

## 1. Ejecución del entorno

Antes de utilizar cualquier script, es necesario configurar y activar el entorno virtual del proyecto:

```bash
./scripts/00_setup-environment.sh
```

Este comando crea (si no existe) el entorno `venv/` y instala todas las dependencias definidas en `requirements.txt`, unificado para todo el repositorio.

## 2. Flujo de limpieza de datos

El proceso de limpieza se ejecuta con:

```bash
./scripts/01_cleaning-data.sh
```

Este comando invoca internamente el archivo
`01_data_cleaning/clean_data.py`, el cual realiza las siguientes acciones:

1. **Carga del dataset original**

   - Lee `data/train.csv` y `data/test.csv`.

2. **Normalización de texto**

   - Convierte textos codificados como listas (`["text"]`) en cadenas planas.
   - Elimina URLs, menciones, hashtags y caracteres no alfabéticos.
   - Estandariza mayúsculas y elimina stopwords del idioma inglés.
   - Limpia errores de codificación y espacios redundantes.

3. **Gestión de valores faltantes**

   - Se imputa el único valor nulo detectado en `response_a_clean` con `"no response"`.
   - Se eliminan filas vacías en las columnas de texto principales.

4. **Generación de etiquetas (`label`)**

   - Se crean tres clases:

     - `0`: gana modelo A
     - `1`: gana modelo B
     - `2`: empate

5. **Prevención de sesgo posicional (swap A↔B)**

   - Se duplican las filas invirtiendo `response_a_clean` ↔ `response_b_clean`
     y `model_a` ↔ `model_b`, ajustando las etiquetas.
   - El dataset aumentado queda en `data_clean_aug.csv`.

6. **Análisis de longitudes**

   - Se calculan métricas descriptivas (`count`, `mean`, `std`, `max`, etc.)
     de la longitud de `prompt_clean`, `response_a_clean` y `response_b_clean`.
   - El resultado se guarda en `length_stats.csv` para definir
     los parámetros `max_len_prompt` y `max_len_response` de los modelos.

7. **División estratificada (train / valid)**

   - Se generan subconjuntos con `train_test_split` (90/10) estratificando por `label`.
   - Los resultados se guardan como:

     - `train_strat.csv`
     - `valid_strat.csv`

8. **Procesamiento de test.csv**

   - Se limpia con el mismo pipeline y se guarda en `test_clean.csv`.

9. **Registro y cacheo**

   - Se guarda un resumen de proceso (`data_process_info.csv`) con:
     pasos aplicados, fechas, número de filas, imputaciones y división.
   - Se crea un archivo de caché (`cache/cleaning_done.txt`) para evitar repeticiones.

## 3. Resumen estadístico

Después de la limpieza, se puede generar un informe descriptivo con:

```bash
./scripts/02_summary-data.sh
```

Este comando ejecuta `01_data_cleaning/data_summary.py`, que produce múltiples reportes
y los guarda en la carpeta `reports/clean/`.

Los archivos generados permiten auditar y validar la calidad de los datos.

## 4. Archivos generados

| Archivo                                                                                    | Ubicación        | Descripción                                                                 |
| ------------------------------------------------------------------------------------------ | ---------------- | --------------------------------------------------------------------------- |
| **data_clean.csv**                                                                         | `data/clean/`    | Dataset base limpio sin duplicaciones ni filas nulas.                       |
| **data_clean_aug.csv**                                                                     | `data/clean/`    | Versión aumentada con inversión A↔B para evitar sesgo de posición.          |
| **train_strat.csv**, **valid_strat.csv**                                                   | `data/clean/`    | Subconjuntos estratificados (90/10) usados para entrenamiento y validación. |
| **test_clean.csv**                                                                         | `data/clean/`    | Conjunto de prueba limpio con las mismas transformaciones.                  |
| **length_stats.csv**                                                                       | `data/clean/`    | Estadísticas de longitud promedio y máxima de los textos.                   |
| **data_process_info.csv**                                                                  | `data/clean/`    | Registro de los pasos de limpieza, imputaciones y resultados de split.      |
| **describe.csv**, **describe_numeric.csv**                                                 | `reports/clean/` | Estadísticas descriptivas globales (numéricas y categóricas).               |
| **na_overview.csv**                                                                        | `reports/clean/` | Conteo y porcentaje de valores nulos por columna.                           |
| **dtypes.csv**                                                                             | `reports/clean/` | Tipos de datos detectados en cada columna.                                  |
| **head.csv**, **tail.csv**                                                                 | `reports/clean/` | Primeras y últimas filas del dataset limpio.                                |
| **empty_prompt_clean.csv**, **empty_response_a_clean.csv**, **empty_response_b_clean.csv** | `reports/clean/` | Registros vacíos detectados por columna (solo si existen).                  |

## 5. Interpretación de los principales reportes

### `describe.csv`

Contiene métricas estadísticas o de frecuencia para cada columna:

| Tipo de dato       | Métricas                                                  | Descripción                                                                |
| ------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------- |
| Numérico           | `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max` | resumen estadístico de valores numéricos                                   |
| Categórico / texto | `count`, `unique`, `top`, `freq`                          | cantidad de registros, valores únicos, valor más frecuente y su frecuencia |
| Binario            | `mean` ≈ proporción de valores 1                          |                                                                            |

Ejemplo:

- `winner_model_a`, `winner_model_b`, `winner_tie` tienen medias ~0.34, 0.34 y 0.31, lo que muestra un balance casi perfecto.
- `prompt_len`, `respA_len`, `respB_len` reflejan longitudes promedio (≈35, 133 y 134 palabras).

### `na_overview.csv`

Muestra cuántos valores nulos tiene cada columna y su porcentaje relativo.
Sirve para verificar la integridad y calidad del dataset.

### `length_stats.csv`

Ayuda a decidir los límites de truncado:

- `max_len_prompt` ≈ 128 tokens
- `max_len_response` ≈ 512–640 tokens

## 6. Conclusión

El módulo **`01_data_cleaning`** automatiza la preparación completa de los datos:
limpieza, imputación, normalización, balanceo y resumen estadístico.

Después de ejecutar:

```bash
./scripts/00_setup-environment.sh
./scripts/01_cleaning-data.sh
./scripts/02_summary-data.sh
```

se obtiene un conjunto de datos estandarizado y trazable, listo para los módulos de entrenamiento y evaluación posteriores.
