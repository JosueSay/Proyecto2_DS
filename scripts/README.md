# Scripts de automatización

Este directorio contiene los scripts de shell para preparar el entorno, entrenar modelos y administrar la aplicación mediante Docker.

## Preparación de entorno

Antes de ejecutar los scripts, asegúrate de convertirlos al formato UNIX y darles permisos de ejecución:

```bash
sudo apt install dos2unix -y
dos2unix scripts/*.sh
chmod +x scripts/*.sh
```

## Ejecución base

### 1. Preparar entorno

Configura las dependencias y variables necesarias para el proyecto.

```bash
./scripts/00_setup-environment.sh
```

### 2. Entrenamiento de modelos

Ejecuta los entrenamientos del modelo **PairRanker** con las configuraciones deseadas:

```bash
./scripts/01_train-pairranker.sh m_pair-ranker/configs/xlnet.yaml
./scripts/01_train-pairranker.sh m_pair-ranker/configs/deberta.yaml
./scripts/01_train-pairranker.sh m_pair-ranker/configs/roberta.yaml
./scripts/01_train-pairranker.sh m_pair-ranker/configs/electra.yaml
```

Los resultados y reportes se almacenan automáticamente en las carpetas `results/` y `reports/`.

## Administración con Docker

Los siguientes scripts controlan el ciclo de vida de la aplicación Streamlit en contenedor.

| Script       | Descripción                                                         |
| ------------ | ------------------------------------------------------------------- |
| `build.sh`   | Construye la imagen Docker con Python 3.10.12 y el entorno virtual. |
| `rebuild.sh` | Reconstruye completamente la imagen y limpia contenedores previos.  |
| `start.sh`   | Inicia la aplicación Streamlit en segundo plano (puerto 8501).      |
| `stop.sh`    | Detiene los contenedores activos sin eliminar volúmenes.            |
| `restart.sh` | Reinicia la aplicación limpiamente sin reconstruir la imagen.       |
| `clean.sh`   | Elimina contenedores, volúmenes, imágenes y caché de Docker.        |

### Flujo recomendado para la aplicación

```bash
./scripts/build.sh
./scripts/start.sh
```

Para detener o limpiar:

```bash
./scripts/stop.sh
# o
./scripts/clean.sh
```

La aplicación estará disponible en:

```bash
http://localhost:8501
```
