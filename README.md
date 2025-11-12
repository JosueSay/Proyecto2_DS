# Chatbot Arena — Modelos de Preferencia Humana

Los **modelos de lenguaje a gran escala (LLMs)** son sistemas de inteligencia artificial entrenados para comprender y generar texto de manera similar a un humano, y se utilizan en asistentes virtuales, buscadores y herramientas de productividad.
Sin embargo, aún existe el reto de que sus respuestas realmente coincidan con lo que los usuarios consideran más útiles o correctas.

En este marco surge **Chatbot Arena**, una plataforma en línea donde los usuarios interactúan con **dos chatbots anónimos** (basados en diferentes LLMs) que responden a la misma instrucción o *prompt*. Después de leer ambas respuestas, el usuario selecciona la que prefiere o puede declarar un empate.
Este esquema de *batalla cara a cara* permite recopilar datos directos sobre las **preferencias humanas frente a distintos modelos de IA**.

Comprender y predecir estas elecciones es fundamental porque aporta información sobre cómo las personas valoran la calidad de las respuestas más allá de lo técnico. Esto resulta clave para construir sistemas conversacionales más útiles, confiables y aceptados en contextos reales, ya que la capacidad de un modelo para adaptarse a las expectativas humanas determina su éxito en aplicaciones prácticas y en la satisfacción del usuario final.

## 📦 Enlaces a recursos

- [Repositorio principal](https://github.com/JosueSay/Proyecto2_DS)
- [Competencia Kaggle](https://www.kaggle.com/competitions/lmsys-chatbot-arena)
- [Drive con data, reportes y resultados](https://drive.google.com/drive/folders/1oxm4w52mPMGAd0iex9FNXftMEYMTlhav?usp=drive_link)

## ⚙️ Instalación y configuración

Probado en **Ubuntu 22.04 / WSL2** con **Python 3.10+**.

### 1. Dependencias del sistema

Instala las utilidades necesarias para ejecutar el Makefile, los scripts y el entorno Python:

```bash
sudo apt-get update
sudo apt-get install -y make build-essential python3 python3-venv python3-pip dos2unix unzip git
dos2unix scripts/*.sh
chmod +x scripts/*.sh
```

### 2. Estructura esperada

Descarga los archivos comprimidos desde el enlace de Drive (ver sección siguiente) y descomprímelos en la raíz del proyecto para restaurar las carpetas:

```bash
data/
reports/
results/
```

Cada carpeta contiene los datos originales, los resultados de entrenamiento y los reportes de análisis y validación generados por el pipeline. Estos archivos son necesarios para consumirlos por el dashboard.

## 🚀 Ejecución

El proyecto puede ejecutarse **mediante Docker** o **de forma local**.

### Opción A — Docker

```bash
./scripts/build.sh
./scripts/start.sh
```

Esto iniciará automáticamente el contenedor, cargará el entorno y abrirá el **dashboard interactivo**.
Dentro de la pestaña *inference*, podrás hacer predicciones cargando tus datos o utilizando el archivo `test.csv` incluido en `data/`.

### Opción B — Ejecución local

Primero configura el entorno Python y dependencias:

```bash
./scripts/00_setup-environment.sh
```

Luego lanza el dashboard con Streamlit:

```bash
streamlit run ./app/streamlit_app.py
```

Esto abrirá la aplicación web en tu navegador, permitiendo explorar resultados, visualizar comparaciones de modelos y realizar inferencias con nuevos prompts.

>**Nota:** Es la `Opción A` es la recomendada pero al copiar la estructura a un contenedor puede ser más lenta, si ya tienes la versión de python correcta utiliza la `Opción B`.

## 🧩 Estructura general

| Carpeta             | Descripción                                                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `00_cache-manager/` | Control de caché para evitar reprocesamiento innecesario en las etapas del pipeline.                                                    |
| `01_data_cleaning/` | Limpieza y preprocesamiento de datos, manejo de duplicados, longitudes y truncado.                                                      |
| `02_eda/`           | Análisis exploratorio con gráficas de distribución, correlaciones, similitud y truncado.                                                |
| `03_metrics/`       | Comparación visual de modelos (DeBERTa, RoBERTa, XLNet, Electra). Incluye métricas de validación, f1 por clase y matrices de confusión. |
| `app/`              | Dashboard desarrollado con **Streamlit** para visualización e inferencia interactiva.                                                   |
| `m_pair-ranker/`    | Módulo de entrenamiento del modelo de ranking por pares. Implementa pérdidas Bradley-Terry y Cross Entropy.                             |
| `reports/`          | Resultados numéricos y métricas generadas automáticamente.                                                                              |
| `results/`          | Pesos y configuraciones de los modelos entrenados.                                                                                      |
| `images/`           | Visualizaciones generadas (EDA y resultados).                                                                                           |
| `scripts/`          | Automatización de tareas: build, start, clean y setup de entorno.                                                                       |

## 🧠 Modelos evaluados

Los modelos utilizados se basan en variantes **transformer preentrenadas**, adaptadas a la tarea de comparación de respuestas (*pairwise ranking*):

- **RoBERTa**
- **DeBERTa**
- **XLNet**
- **Electra**

El entrenamiento y validación se manejan con el módulo `m_pair-ranker`, que registra automáticamente métricas de desempeño, entropía, distribución de predicciones y matrices de confusión.

## 📈 Resultados

El sistema genera comparativas visuales automáticas mediante el comando:

```bash
make metrics
```

Estas gráficas se guardan en `images/resultados/` e incluyen:

- Accuracy y F1 por época.
- Entropía de predicción.
- Distribución de clases por modelo.
- F1 por clase.
- Matrices de confusión para cada arquitectura.

## 📺 Video demostrativo

- [Enlace YouTube](https://youtu.be/-JMSfvz8AOY)
