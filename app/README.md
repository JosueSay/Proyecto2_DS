# Dashboard

Este módulo contiene la aplicación principal desarrollada en **Streamlit**, diseñada para explorar datos, ejecutar inferencias, comparar modelos y visualizar resultados de entrenamiento.
La estructura está organizada en componentes, páginas y utilidades para facilitar la mantenibilidad y extensión del proyecto.

## Estructura general

```bash
app/
   ├── README.md
   ├── __init__.py
   ├── app_config.yaml          # Configuración general (rutas, modelos habilitados, opciones globales)
   ├── assets/                  # Recursos estáticos (íconos, CSS, etc.)
   ├── components/              # Elementos visuales reutilizables (gráficas, tarjetas de métricas, toggles)
   ├── pages/                   # Páginas principales de la aplicación Streamlit
   ├── streamlit_app.py         # Entrada principal de la app
   └── utils/                   # Funciones auxiliares (carga de datos, gráficos, limpieza, predicción, etc.)
```

## Descripción de las páginas (`pages/`)

- **model_dashboard.py** – Panel comparativo entre modelos. Muestra métricas globales, curvas de entrenamiento, matrices de confusión y distribuciones de predicciones.
- **compare_models.py** – Tabla y gráfico comparativo de métricas finales (F1, accuracy, pérdida) entre modelos.
- **data_explorer.py** – Explorador de archivos CSV generados en etapas de limpieza o análisis exploratorio. Sugiere gráficos según el tipo de datos.
- **inference.py** – Permite realizar inferencias individuales o por lotes (CSV) con los modelos entrenados, mostrando resultados y exportables.
- **settings.py** – Página de configuración y mantenimiento. Muestra rutas activas, modelos habilitados y permite limpiar cachés de datos o recursos.

## Archivo principal

- **streamlit_app.py** – Punto de inicio de la aplicación. Define pestañas para visión general, gestión de modelos, limpieza de datos y visualización de métricas del mejor modelo.

## Ejecución

Desde la raíz del proyecto, ejecutar:

```bash
streamlit run app/streamlit_app.py
```
