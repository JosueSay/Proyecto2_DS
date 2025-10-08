
# [Preference Transformer: Modeling Human Preferences using Transformers for RL](https://arxiv.org/abs/2303.00957)

## Resumen

Este trabajo propone un modelo llamado **Preference Transformer**, que utiliza arquitecturas transformer para modelar preferencias humanas (entre dos comportamientos o secuencias) en el contexto de aprendizaje por refuerzo con retroalimentación humana.

Se parte del problema de que los métodos clásicos de aprendizaje basado en preferencias asumen que las decisiones humanas dependen de recompensas *markovianas* con igual peso en todos los estados, lo cual puede fallar al capturar dependencias temporales o juicios que consideren eventos más críticos. El modelo trata de capturar esas dependencias temporales no markovianas para reflejar mejor la forma en que las personas juzgan secuencias de comportamiento.

## Arquitectura propuesta

* La arquitectura combina capas de **self-attention causal** (para modelar dependencia hacia adelante) y **auto-atención bidireccional**, permitiendo que el modelo atienda tanto pasado como futuro en la secuencia.
* Se define la preferencia humana como suma ponderada de recompensas no markovianas a lo largo de la trayectoria, en lugar de asumir que cada paso contribuye igual.
* De esta manera, el modelo puede asignar más peso a pasos clave de la secuencia (por ejemplo, eventos críticos), lo que imita cómo los humanos pueden valorar más algunos momentos que otros.

## Experimentos y resultados

* Se aplica el modelo en tareas de control con retroalimentación humana. En comparación, métodos anteriores que usaban recompensas markovianas o modelos simples no logran resolver ciertos entornos donde las decisiones humanas dependen de eventos específicos.
* Preference Transformer logra inducir una recompensa bien especificada y es capaz de “atender” eventos críticos (es decir, identificar qué partes de la secuencia son relevantes para la preferencia humana).
* Muestra que el modelo funciona incluso cuando los modelos clásicos fallan en capturar juicios humanos complejos.

## Ventajas y limitaciones

**Ventajas:**

* Captura dependencias temporales no markovianas en juicios humanos, lo que permite modelar preferencia más realistas.
* Arquitectura transformer flexible que puede asistir con atención en diferentes momentos de la secuencia.
* Buena capacidad para reflejar cómo los humanos valoran más ciertos eventos clave en la trayectoria.

**Limitaciones:**

* Requiere datos de retroalimentación humana suficientes para entrenar correctamente estas estructuras complejas.
* Potencialmente más costoso en cómputo por usar transformadores con atención compleja sobre secuencias largas.
* No está explícitamente diseñado para tareas de texto comparativo (como tu problema “respuesta A vs. respuesta B”), aunque la idea puede adaptarse.
