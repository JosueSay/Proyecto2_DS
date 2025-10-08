# RoBERTa / DeBERTa ajustado con aprendizaje por pares (pairwise loss)

## Resumen

Se busca predecir preferencias humanas entre dos respuestas textuales usando **encoders transformer** (RoBERTa/DeBERTa) entrenados con **pérdidas de ranking por pares**.
Se fundamenta en:

1. La solidez empírica de RoBERTa/DeBERTa como representadores de texto.
2. La modelación probabilística tipo **Bradley–Terry** para comparaciones A/B.
3. Prácticas de *reward modeling* y retroalimentación ordinal para capturar empates e incertidumbre.

## Objetivo

Aprender un modelo que, dado un **prompt** y dos **respuestas** ($A$ y $B$), produzca **probabilidades calibradas** de preferencia: $P(A), P(B), P(\text{empate})$, minimizando *log loss* multiclase y respetando sesgos comunes (posición, verbosidad).

## Contexto y motivación

Los encoders **RoBERTa** y **DeBERTa** muestran ventajas consistentes sobre BERT en tareas de comprensión y clasificación de texto, por lo que son candidatos naturales como *backbones* para ranking de respuestas. En particular, **DeBERTa** incorpora atención “disentangled” y manejo de posiciones que fortalecen su capacidad representacional; estos rasgos han demostrado mejoras en benchmarks como GLUE/SuperGLUE.  

## Formulación del problema

Dada una tupla ($x, y_A, y_B$) (prompt y dos respuestas), buscamos un **score** $r_\theta(x, y)$ que represente la “calidad” relativa de cada respuesta y permita derivar una probabilidad de preferencia entre pares. El modelo debe además manejar el **empate** como caso válido cuando ambas respuestas son comparables, lo que se alinea con marcos de **retroalimentación ordinal**.

## Arquitectura propuesta

**Encoder cruzado (cross-encoder)** basado en **RoBERTa** o **DeBERTa**:

- Entrada: concatenación estructurada de *prompt* y cada respuesta, p. ej. $[CLS]\ x\ [SEP]\ y_A\ [SEP]$ y $[CLS]\ x\ [SEP]\ y_B\ [SEP]$.
- El encoder produce representaciones contextualizadas y un **cabezal de puntuación** $r_\theta(\cdot)$ (MLP sobre el *pooled output*).
- Para comparar A vs. B se computa $\Delta = r_\theta(x, y_A) - r_\theta(x, y_B)$, que alimenta la **pérdida por pares**.
  La elección de **DeBERTa** se justifica por su **atención disentangled** (contenido/posición) y su manejo de posiciones que han mostrado ganancias frente a RoBERTa/BERT; no obstante, RoBERTa suele rendir de forma muy competitiva con menor costo.  

## Función de pérdida: Bradley–Terry / RankNet

Para una etiqueta binaria “A preferido a B” ($y=1$ si A gana, $0$ si B gana), usamos la **formulación Bradley–Terry**:

$$
P(A \succ B \mid x) = \sigma\big(r_\theta(x, y_A) - r_\theta(x, y_B)\big),
$$

y la **cross-entropy** correspondiente:

$$
\mathcal{L}_{\text{pair}} = - y \log P(A\succ B) - (1-y)\log \big(1-P(A\succ B)\big).
$$

Esta formulación conecta con regresión logística entre pares y sustenta el aprendizaje directo de preferencias A/B.

## Manejo de empates y retroalimentación ordinal

Cuando existe **empate** o juicios de granularidad fina, se recomienda una extensión **ordinal** que suaviza etiquetas y mejora la generalización. Un esquema es usar una variable ordinal $Z\in[0,1]$ (p. ej., $Z=0.5$ para empate) en la pérdida:
$$
-Z \log \sigma(\Delta) - (1-Z)\log \sigma(-\Delta),
$$
lo que generaliza la pérdida binaria y aprovecha más información humana (reduce complejidad estadística y mejora robustez OOD).

## Entrenamiento y validación

- **Expansión A↔B**: duplicar ejemplos invirtiendo respuestas mitiga **sesgo de posición** y estabiliza el aprendizaje de diferencias verdaderamente semánticas.
- **Truncado balanceado**: respetar presupuesto de tokens recortando proporcionalmente para no sesgar por longitud.
- **K-fold estratificado**: validación robusta, idealmente agrupando por *prompt* para evitar *leakage* por duplicados.
- **Calibración**: verificar confiabilidad probabilística (p. ej., confiabilidad Brier/diagrama) dado que la métrica objetivo es *log loss*.
  Estas prácticas están alineadas con enfoques de **modelos de recompensa** que priorizan coherencia con preferencias humanas y robustez ante ruido/varianza.

## Extensiones

- **Reward modeling con incertidumbre**: estimar **varianza heterocedástica** junto al puntaje medio permite tratar ruido humano y usar señales “pesimistas” más robustas durante entrenamiento.
- **Ensamble/Blending con PAIRRANKER**: marcos como **LLM-BLENDER** usan un **PAIRRANKER** basado en DeBERTa para comparar respuestas y agregarlas; esto aporta evidencia empírica de que encoders tipo DeBERTa funcionan bien como jueces pareados.
- **Transformers de preferencia**: lineamientos de arquitectura que modelan preferencias humanas más allá de recompensas markovianas, útiles como base teórica cuando las decisiones dependen de eventos críticos.

## Limitaciones y consideraciones

- **Costo y longitud**: los encoders tienen límites de contexto; es necesario un diseño de truncado que no destruya señales clave.
- **Dominio de datos**: *further training* en dominio específico puede ser beneficioso, pero incrementa costo y riesgo de sobreajuste.
- **Empates y ruido**: sin retroalimentación ordinal o calibración adecuada, el modelo puede sobreconfiar en casos ambiguos.

## Protocolo de uso

1. **Backbone**: DeBERTa (o RoBERTa) como encoder cruzado con cabezal de puntuación.
2. **Pérdida**: Bradley–Terry/RankNet; extender a ordinal si existen empates o grados de preferencia.  
3. **Datos**: duplicación A↔B y truncado equilibrado; dividir con K-fold estratificado/agrupado.
4. **Calibración y robustez**: medir calibración; considerar varianza heterocedástica en *reward modeling*.
5. **Validación**: reportar *log loss* por *fold* y promedios; revisar estabilidad y sesgos residuales.

## Arquitectura

**Stack:** PyTorch + Hugging Face (transformers, datasets, accelerate).

Librerías: `torch`, `transformers`, `datasets`, `accelerate`, `scikit-learn`, `numpy`, `pandas`, `tqdm`, `pyyaml` y `wandb`.

### Estructura de carpetas

```bash
project/
├─ data/
│  ├─ train.csv
│  ├─ valid.csv
│  └─ test.csv
├─ configs/
│  └─ default.yaml
├─ src/
│  ├─ __init__.py
│  ├─ data.py            # Dataset pairwise, duplicación A↔B, truncado balanceado
│  ├─ tokenizer.py       # Carga de tokenizer y reglas de truncado
│  ├─ model.py           # Cross-encoder RoBERTa/DeBERTa + head escalar rθ
│  ├─ losses.py          # Bradley–Terry/RankNet; variante ordinal para empates
│  ├─ train.py           # Loop de entrenamiento (Accelerate), logging y checkpoints
│  ├─ evaluate.py        # Log loss multiclase, calibración
│  ├─ infer.py           # Predicción P(A), P(B), P(tie) con TTA A↔B y promedio por id
│  ├─ utils.py           # Semillas, métrica, lectura de config, guardado
│  └─ postprocess.py     # Normalizar probabilidades y generar submission.csv
├─ scripts/
│  ├─ run_train.sh
│  ├─ run_eval.sh
│  └─ run_infer.sh
├─ requirements.txt
└─ README.md
```

### requirements.txt

```bash
torch
transformers
datasets
accelerate
scikit-learn
numpy
pandas
tqdm
pyyaml
```

### configs/default.yaml

```yaml
model_name: microsoft/deberta-v3-base
max_len_prompt: 256
max_len_resp: 700
batch_size: 8
lr: 2e-5
num_epochs: 3
weight_decay: 0.01
warmup_ratio: 0.06
seed: 42
k_folds: 5
bal_truncate: true
use_ordinal_tie: true
```

### Descripción de módulos

- `data.py`:

  - Lee `data/train.csv`, crea pares (x, yA) y (x, yB); construye ejemplos A vs B y su duplicado invertido B vs A.
  - Devuelve tensores listos para el encoder cruzado.
- `tokenizer.py`:

  - Carga tokenizer del backbone y aplica truncado balanceado para `prompt`, `response_a`, `response_b`.
- `model.py`:

  - Carga RoBERTa/DeBERTa como encoder cruzado; agrega una capa MLP a partir del `[CLS]` (o `pooler`) para producir `rθ(x,y)`.
- `losses.py`:

  - Bradley–Terry/RankNet: `-y*log σ(Δ) - (1-y)*log σ(-Δ)`; variante ordinal para empates (Z∈[0,1]).
- `train.py`:

  - Entrenamiento con `accelerate` (CPU/GPU/MP); K-fold; guarda mejor checkpoint por fold.
- `evaluate.py`:

  - Calcula log loss multiclase por fold; opcionalmente confiabilidad/calibración.
- `infer.py`:

  - Para cada `id` en `data/test.csv`, puntúa A y B, obtiene `Δ`, `σ(Δ)`→P(A|B), trata empates (ordinal/regla) y aplica TTA A↔B promediando por `id`.
- `postprocess.py`:

  - Garantiza que `P(A)+P(B)+P(tie)=1` y genera `submission.csv`.

### Scripts

`scripts/run_train.sh`

```bash
accelerate launch -m src.train --config configs/default.yaml
```

`scripts/run_eval.sh`

```bash
python -m src.evaluate --config configs/default.yaml
```

`scripts/run_infer.sh`

```bash
python -m src.infer --config configs/default.yaml --test data/test.csv --out submission.csv
```

## Referencias

- [https://svallejovera.github.io/files/bert_roberta_jop.pdf](https://svallejovera.github.io/files/bert_roberta_jop.pdf)
- [https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
- [https://sh-tsang.medium.com/brief-review-deberta-decoding-enhanced-bert-with-disentangled-attention-f5cdb9a8bf0b](https://sh-tsang.medium.com/brief-review-deberta-decoding-enhanced-bert-with-disentangled-attention-f5cdb9a8bf0b)
- [https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-82.pdf](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-82.pdf)
- [https://arxiv.org/pdf/2411.12843](https://arxiv.org/pdf/2411.12843)
- [https://aclanthology.org/2023.acl-long.792.pdf](https://aclanthology.org/2023.acl-long.792.pdf)
- [https://arxiv.org/abs/2303.00957](https://arxiv.org/abs/2303.00957)
