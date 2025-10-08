# [Bradley–Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)

## ¿De qué trata?

El modelo **Bradley–Terry** es un modelo de probabilidad para **comparaciones por pares**: dados dos ítems *i* y *j*, estima la probabilidad de que *i* sea preferido sobre *j*.
Se usa para tareas como ranking de equipos deportivos, preferencias de productos o elección entre modelos de IA.

## Definición matemática

* Se le asigna a cada ítem *i* un puntaje positivo real ( $p_i$ ).
* La probabilidad de que *i* gane frente a *j* está dada por:
  $$
  \Pr(i > j) = \frac{p_i}{p_i + p_j}
  $$
* Otra formulación: expresar ( $p_i = e^{\beta_i}$ ), entonces:
  $$
  \Pr(i > j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}} = \frac{1}{1 + e^{\beta_j - \beta_i}}
  $$
* La diferencia ($\beta_i - \beta_j$) se asemeja a una regresión logística entre pares.

## Inferencia (estimación de parámetros)

* A partir de resultados observados (cuántas veces *i* venció a *j*), se puede estimar los valores ( $p_i$ ) mediante **máxima verosimilitud**.
* Si ( $w_{ij}$ ) es el número de veces que *i* vence a *j*, la verosimilitud del modelo es:
  $$
  \prod_{i,j} [\Pr(i > j)]^{w_{ij}}
  $$
* No hay solución cerrada simple, por lo que se usa iteración:
  $$
  p_i' = \frac{\sum_j w_{ij} \cdot \frac{p_j}{(p_i + p_j)}}{\sum_j w_{ji} \cdot \frac{1}{(p_i + p_j)}}
  $$
  Luego se normalizan para mantener escala.
* Se repite hasta converger al máximo de la verosimilitud.

## Aplicaciones y extensiones

* Se ha usado en ranking deportivo, elección de productos, comparación entre modelos, ranking de documentos en motores de búsqueda, etc.
* Una extensión es **Crowd-BT**, que adapta el modelo para escenarios con múltiples jueces y calidad variable del evaluador (filtrado de jueces malos).
* El modelo generaliza al **modelo Plackett–Luce** cuando hay más de dos ítems comparados (ranking completo).
