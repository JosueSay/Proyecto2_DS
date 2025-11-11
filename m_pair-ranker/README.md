# **PairRanker – Estrategia de Entrenamiento y Arquitectura General**

## **1. Propósito del sistema**

El proyecto **PairRanker** implementa un pipeline de entrenamiento completo para modelos **cross-encoder** de preferencia entre pares de respuestas a un mismo *prompt*.
El objetivo es que el modelo aprenda a decidir, entre dos respuestas posibles (`respA`, `respB`), cuál es mejor o si ambas son equivalentes (*TIE*).

Este enfoque es común en sistemas de evaluación de calidad de texto generativo, comparaciones de respuestas de modelos de lenguaje o preferencias humanas.

## **2. Filosofía de diseño**

La implementación se guía por tres principios:

1. **Configuración totalmente declarativa:**
   Todo el entrenamiento se controla desde un archivo YAML validado contra un esquema cerrado.
   Esto garantiza reproducibilidad, evita configuraciones inconsistentes y documenta explícitamente todos los hiperparámetros.

2. **Arquitectura modular y extensible:**
   Los componentes (datasets, modelos, pérdidas, validación) son intercambiables mediante interfaces consistentes.
   Cada módulo está diseñado para poder sustituirse o ampliarse sin romper dependencias.

3. **Entrenamiento trazable y auditado:**
   Cada ejecución genera reportes, métricas y configuraciones completas en un directorio de *run* único con timestamp.
   Esto permite análisis detallado posterior y comparaciones entre experimentos.

## **3. Estructura general del sistema**

```bash
m_pair-ranker
   ├── README.md
   ├── configs
   │   ├── README.md
   │   ├── deberta.yaml
   │   ├── electra.yaml
   │   ├── roberta.yaml
   │   └── xlnet.yaml
   └── src
       └── pairranker
           ├── __init__.py
           ├── config
           │   ├── __init__.py
           │   ├── loader.py
           │   └── schema.py
           ├── data_p
           │   ├── __init__.py
           │   ├── collate.py
           │   └── dataset.py
           ├── losses
           │   ├── __init__.py
           │   ├── bradley_terry.py
           │   ├── cross_entropy.py
           │   └── factory.py
           ├── models
           │   ├── __init__.py
           │   ├── backbones.py
           │   ├── cross_encoder.py
           │   └── heads.py
           ├── train
           │   ├── __init__.py
           │   ├── loop.py
           │   ├── utils.py
           │   └── validation.py
           └── utils
               ├── __init__.py
               ├── io.py
               └── text.py
```

## **4. Configuración y validación**

### **Archivos:**

* `config/loader.py`
* `config/schema.py`

El entrenamiento se define mediante un YAML que sigue estrictamente el esquema `REQUIRED_SCHEMA`.
El cargador (`loadYamlConfig`) valida:

* Que todas las claves requeridas estén presentes.
* Que no existan claves no reconocidas.
* Que los tipos y valores posibles sean los esperados.
* Que se generen directorios de ejecución (`run_dir`, `report_dir`) con timestamp único.

Además, guarda la configuración efectiva usada para asegurar trazabilidad.

**Razón:**
Garantiza reproducibilidad, previene errores silenciosos en configuración y permite auditoría completa de cada experimento.

## **5. Dataset y collation**

### **Archivos:**

* `data_p/dataset.py`
* `data_p/collate.py`

El dataset (`PairDataset`) usa un **enfoque de pares**:

* Cada fila representa un *prompt* con dos posibles respuestas (`respA`, `respB`) y una etiqueta (`label` ∈ {0: A, 1: B, 2: TIE}).
* Se tokenizan ambos pares por separado, con límites configurables (`max_len_prompt`, `max_len_resp`).
* Se construye un *input* combinando ambos textos, con truncado balanceado y padding fijo.

El *collate function* (`collateFn`) combina lotes, calcula estadísticas útiles (longitudes, truncamientos, presupuesto de tokens) y produce un diccionario de tensores homogéneo.

**Razón:**
Controlar el presupuesto de secuencia es crítico en modelos Transformer. El truncado balanceado mantiene la proporción entre prompt y respuesta, evitando sesgos por longitud.

## **6. Modelo Cross-Encoder**

### **Archivos:**

* `models/backbones.py`
* `models/heads.py`
* `models/cross_encoder.py`

#### **Backbone**

Usa `AutoModel` de Hugging Face, configurable por nombre (`pretrained_name`), dropout y *gradient checkpointing*.
El checkpointing se activa sólo si el modelo lo soporta, reduciendo memoria sin alterar resultados.

#### **Cabezas**

* **ScoreHead:** Produce un escalar por representación `pooled`, usado en pérdidas de tipo *Bradley-Terry* (ranking).
* **PairClassifier:** Combina `h_a`, `h_b`, su diferencia absoluta y su producto para producir logits de clase `[A, B, TIE]`.

#### **CrossEncoder**

Integra backbone + cabezas y ofrece tres salidas:

* `logits` para clasificación (*cross-entropy*),
* `s_a`, `s_b` para pérdidas basadas en ranking,
* `predictProbs()` para evaluación con `softmax`.

**Razón:**
Permite usar tanto entrenamiento directo de clasificación (CE) como comparativo (BT), con mínima duplicación de código.

## **7. Funciones de pérdida**

### **Archivos:**

* `losses/cross_entropy.py`
* `losses/bradley_terry.py`
* `losses/factory.py`

#### **Cross Entropy**

Usa `torch.nn.functional.cross_entropy` con:

* *label smoothing* opcional.
* *class weights* configurables (para manejar desbalance de clases A/B/TIE).

#### **Bradley-Terry**

Implementa un modelo probabilístico de preferencias:

* Calcula la probabilidad de que A gane a B según la diferencia de *scores*.
* Maneja empates con un componente `p_TIE` adicional.
* Compatible con la validación que reconstruye `p_A`, `p_B`, `p_TIE` a partir de los *scores*.

#### **Factory**

Permite seleccionar la pérdida desde el YAML (`cross_entropy` o `bradley_terry`).

**Razón:**
Facilita experimentación entre estrategias de aprendizaje:

* CE es más estable y directa.
* BT es más interpretativa y apropiada para comparaciones de pares.

## **8. Entrenamiento**

### **Archivo:** `train/loop.py`

`trainModel(cfg_path)` ejecuta el ciclo completo de entrenamiento:

1. Carga la configuración validada.
2. Inicializa dataset, dataloaders y modelo.
3. Configura optimizador (AdamW) y scheduler (`cosine` o `linear` warmup).
4. Soporta AMP (bf16 o fp16), acumulación de gradientes y clipping.
5. Registra métricas en CSV (`step.csv`, `epoch.csv`, `alerts.csv`).
6. Evalúa periódicamente en validación (`runValidation`).
7. Aplica early stopping y guarda el mejor modelo.

**Razón:**
Se prioriza estabilidad (grad clipping, manejo de NaN), eficiencia (AMP, gradient accumulation) y trazabilidad (reportes CSV).

## **9. Validación y reportes**

### **Archivo:** `train/validation.py`

Evalúa el modelo en el conjunto de validación:

* Calcula métricas: `val_loss`, `val_acc`, `macro_f1`, `entropy`, `dist`.
* Genera `confusion matrix`, `class report`, `pred distributions`.
* Exporta resultados por época en CSV.
* Detecta colapsos o distribuciones anómalas de predicciones (alertas de colapso).

**Razón:**
La validación está diseñada para análisis posterior. La métrica de entropía y la distribución promedio permiten detectar colapsos hacia predicciones uniformes o dominadas por una clase.

## **10. Utilidades**

### **Archivos:**

* `train/utils.py`: manejo de semillas, GPU y métricas de gradiente.
* `train/io.py`: operaciones de lectura/escritura con verificación de existencia.
* `utils/text.py`: limpieza robusta de texto (caracteres Unicode, NaN, listas serializadas).

**Razón:**
Estas funciones evitan errores comunes y garantizan consistencia de datos y reproducibilidad.
