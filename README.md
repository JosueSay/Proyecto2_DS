# Procesamiento del Lenguaje Natural - ChatBot Arena

## 📁 Estructura del proyecto

```text
Proyecto2_DS
├── 00_cache-manager       # Gestión de cache
├── 00_data_cleaning       # Limpieza y resumen de datos
├── 00_logs-manager        # Configuración y logs
├── 01_pair-ranker-deberta # Modelo DeBERTa para ranking de pares
├── cache                  # Archivos temporales de cache
├── data                   # Datos crudos y procesados
├── docs                   # Documentación, investigaciones y referencias
├── images                 # Visualizaciones y diagramas
├── logs                   # Archivos de logs
├── results                # Resultados y métricas de modelos
├── runs                   # Configuraciones y checkpoints de entrenamiento
└── scripts                # Scripts para limpieza, entrenamiento e inferencia
```

## 🖥️ Entorno

* **SO**: Ubuntu 22.04.5 LTS
* **Python**: 3.10

Instala dependencias principales:

```bash
pip install -r requirements.txt
```

Para el modelo DeBERTa:

```bash
pip install -r 01_pair-ranker-deberta/requirements_pairranker.txt
```

## ⚡ Scripts principales

Los scripts `.sh` permiten ejecutar distintas partes del flujo de trabajo:

| Script                   | Descripción                                      | Comando                               |
| ------------------------ | ------------------------------------------------ | ------------------------------------- |
| `00_cleaning-data.sh`    | Limpieza de la data y almacenamiento global      | `bash scripts/00_cleaning-data.sh`    |
| `00_summary-data.sh`     | Genera resumen de los datos procesados           | `bash scripts/00_summary-data.sh`     |
| `01_setup-pairranker.sh` | Configuración del entorno para el modelo DeBERTa | `bash scripts/01_setup-pairranker.sh` |
| `02_train-pairranker.sh` | Entrena el modelo DeBERTa                        | `bash scripts/02_train-pairranker.sh` |
| `03_infer-pairranker.sh` | Realiza inferencia con el modelo DeBERTa         | `bash scripts/03_infer-pairranker.sh` |

> 💡 **Nota:** Primero se deben ejecutar los scripts de limpieza y resumen para tener los datos procesados.

## 📥 Datos

Si no cuentas con los datos, puedes descargarlos aquí:
[Enlace a Google Drive con los datos crudos y resultados de DeBERTa](https://drive.google.com/drive/folders/1oxm4w52mPMGAd0iex9FNXftMEYMTlhav?usp=sharing)

* `train.csv` y `test.csv`: Datos originales.
* `train_clean.csv` y `validation_clean.csv`: Datos limpios listos para entrenamiento.

## 📝 Uso rápido

1. Limpiar y procesar los datos:

    ```bash
    bash scripts/00_cleaning-data.sh
    bash scripts/00_summary-data.sh
    ```

2. Configurar el modelo DeBERTa:

    ```bash
    bash scripts/01_setup-pairranker.sh
    ```

3. Entrenar el modelo:

    ```bash
    bash scripts/02_train-pairranker.sh
    ```

4. Realizar inferencia:

    ```bash
    bash scripts/03_infer-pairranker.sh
    ```

## 📊 Resultados

Los resultados y métricas se encuentran en:

* `results/deberta/metrics_deberta.csv`
* `results/xlnet/`

Visualizaciones en la carpeta `images`.

## 📚 Referencias

* Documentación técnica y estudios previos en `docs/referencias`.
* Investigaciones y análisis exploratorio en `docs/investigaciones`.
