# Manejador de caché

Este módulo se encarga de manejar un sistema de caché para evitar repetir procesos o ejecutar comandos que ya fueron realizados previamente.

Su función principal es controlar y registrar resultados temporales en archivos, permitiendo que el sistema reconozca si una tarea ya fue ejecutada antes. Así, cuando se repite un comando, puede reutilizar la información guardada en lugar de volver a procesarla.

El módulo:

- Crea y organiza los archivos de caché según una configuración definida en un archivo YAML.
- Verifica si existe una caché asociada a una tarea específica.
- Permite crear, leer o eliminar archivos de caché según sea necesario.

En resumen, sirve como un controlador central que optimiza la ejecución evitando repeticiones innecesarias de procesos.
