# Comparación y visualización de métricas de modelos

Este módulo genera **gráficas comparativas** entre los distintos modelos entrenados con **m_pair-ranker**, utilizando los resultados almacenados en `reports/<modelo>/run_*/`.
El objetivo es tener una **evaluación visual unificada** del rendimiento de cada arquitectura (`DeBERTa`, `RoBERTa`, `Electra`, `XLNet`) sin alterar el pipeline principal.

## Estructura

```bash
03_metrics/
├── __init__.py
├── compare_models.py      # orquestador principal (punto de entrada)
├── loaders.py             # carga y validación de los CSV de cada modelo
├── paths.py               # utilidades de rutas para detectar los últimos runs
├── plots.py               # generación de todas las gráficas
└── styles.py              # paleta y estilo visual coherente con 02_eda
```

## Descripción general

El módulo lee los CSV generados por la validación de entrenamiento:

| Archivo origen           | Descripción principal                                                     |
| ------------------------ | ------------------------------------------------------------------------- |
| `epochs.csv`             | Métricas por época (accuracy, macro-F1, entropía, distribución de clases) |
| `class_report.csv`       | Reporte de precisión, recall y F1 por clase (`A`, `B`, `TIE`)             |
| `confusion.csv`          | Matriz de confusión final (predicciones vs etiquetas)                     |
| `pred_distributions.csv` | Distribución de probabilidades promedio por clase (solo informativo)      |

Con esta información se producen visualizaciones centralizadas guardadas en `images/results/`.

## Gráficas generadas

1. **Accuracy por época**
   Muestra la evolución de `val_acc` durante el entrenamiento de cada modelo.

2. **Macro-F1 por época**
   Curvas de `macro_f1` por época para comparar estabilidad y convergencia.

3. **Entropía por época**
   Nivel de confianza promedio en las predicciones de validación (menor es mejor).

4. **F1 por clase (barras)**
   Muestra el desempeño final por clase (`A`, `B`, `TIE`) en la última época.

5. **Distribución de predicciones promedio (barras)**
   Porcentaje de cada tipo de predicción en la última época, útil para detectar sesgos de clase.

6. **Matriz de confusión por modelo**
   Representa visualmente el desempeño final, con conteos reales y predichos.

## Estilo visual

El módulo aplica la paleta definida en `styles.py`, coherente con la utilizada en `02_eda`:

| Rol            | Color     | Uso                                |
| -------------- | --------- | ---------------------------------- |
| `dominante`    | `#1B3B5F` | Títulos, ejes, texto               |
| `secundario`   | `#2E5984` | Series o barras del modelo RoBERTa |
| `mediacion`    | `#5C7EA3` | Series del modelo Electra          |
| `acento`       | `#F28C38` | Serie XLNet o barras de equilibrio |
| `confirmacion` | `#3D8361` | Serie DeBERTa                      |
| `advertencia`  | `#C14953` | Elementos de alerta                |
| `neutro`       | `#B0BEC5` | Grilla y bordes                    |

Todos los gráficos se exportan con fondo blanco y calidad de 300 dpi (`EXPORT_KW` en `styles.py`).

## Ejecución

El módulo no forma parte del pipeline estándar (`make all`).
Se ejecuta de forma independiente con:

```bash
make metrics
```

o manualmente:

```bash
python -m 03_metrics.compare_models
```

## Salidas

Las imágenes se guardan en:

```bash
images/results/
├── results_acc_vs_epoch.png
├── results_f1_macro_vs_epoch.png
├── results_entropy_vs_epoch.png
├── results_class_f1_by_model.png
├── results_pred_dist_by_model.png
├── results_confmat_deberta.png
├── results_confmat_roberta.png
├── results_confmat_electra.png
└── results_confmat_xlnet.png
```
