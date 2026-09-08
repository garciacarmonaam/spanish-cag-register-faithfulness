# Spanish CAG Register Faithfulness

Reproducible experiments for evaluating **Faithfulness robustness in Cache-Augmented Generation (CAG)** when semantically equivalent queries are expressed in **formal and colloquial Spanish**.

This repository contains the experimental resources, code, and results associated with the study:

> **Does Linguistic Register Affect Faithfulness in Spanish Cache-Augmented Generation? An Exploratory Study with Open-Source LLMs**

**Author:** Ángel Manuel García-Carmona  
**ORCID:** 0009-0000-5959-4868

## Overview

Large Language Model (LLM) applications must process user queries expressed through heterogeneous linguistic formulations. This study investigates whether changes in linguistic register affect the extent to which CAG responses remain supported by a fixed preloaded knowledge context.

The experiment compares semantically equivalent **formal and colloquial Spanish queries** while keeping constant:

- semantic intent;
- external knowledge;
- instructional prefix;
- generation configuration;
- evaluation procedure.

The study focuses specifically on **Faithfulness**, understood as the degree to which claims contained in a generated response are supported by the supplied contextual knowledge.

## Experimental Design

The experiment comprises:

- **10 Spanish concepts** related to emotional states and feelings;
- **2 task types**:
  - definition generation;
  - concept identification;
- **20 semantic cases**;
- **2 linguistic registers**:
  - formal Spanish;
  - colloquial Spanish;
- **40 queries per model**;
- **3 open-source instruction-tuned LLMs**;
- **120 planned generations**.

Formal and colloquial versions of each semantic case preserve the same underlying information need while varying their linguistic realization.

## Models

The following models were evaluated:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `ministral/Ministral-3b-instruct`

Generation was deterministic:

```python
do_sample = False
max_new_tokens = 200
```

No model-specific prompt engineering was applied.

## Cache-Augmented Generation

The experiment uses a retrieval-free CAG pipeline.

The complete knowledge base is incorporated into a fixed textual context and processed before individual queries. Its key-value (KV) representations are computed once and retained as a reusable base cache.

For each query:

1. the base KV cache is copied;
2. the query-specific tokens are appended;
3. generation proceeds conditionally on the cached knowledge.

Formal and colloquial versions of the same semantic case are therefore processed against the **same preloaded knowledge state**.

No query-time document retrieval, ranking, or context selection is performed.

## Knowledge Base and Queries

The knowledge base contains definitions manually identified and transcribed from the **Diccionario de la lengua española (DLE)** of the **Real Academia Española (RAE)**.

Formal and colloquial query formulations were generated using **GPT-5 through the ChatGPT interface** and subsequently reviewed manually to verify preservation of semantic intent and requested information.

## Faithfulness Evaluation

Generated responses were evaluated using **RAGAS Faithfulness (v0.3.9)**.

The evaluation uses:

- **Judge:** `granite3.2:latest`
- **Runtime:** Ollama
- **Judge temperature:** `0`

RAGAS Faithfulness evaluates whether claims contained in a generated response are supported by the supplied context.

Of the **120 planned generations**, **113 received valid Faithfulness scores**. Seven evaluations affected by technical failures in the automated evaluation process were retained as missing values and were not imputed.

## Statistical Analysis

The primary analysis follows the paired experimental design.

For each semantic case:

```text
D = Faithfulness(formal) - Faithfulness(colloquial)
```

Positive values indicate higher Faithfulness for the formal formulation, whereas negative values indicate higher Faithfulness for the colloquial formulation.

The analysis includes:

- descriptive statistics;
- Shapiro-Wilk tests on paired differences;
- two-sided paired t-tests;
- 95% confidence intervals;
- two-sided Wilcoxon signed-rank tests;
- median-centered Levene test;
- Friedman test for repeated model-level comparisons.

A total of **53 complete formal-colloquial pairs** were available for paired analysis.

## Main Findings

No statistically significant formal-colloquial Faithfulness difference was detected for any of the three evaluated models.

Descriptively, however, different model-specific patterns emerged:

- **SmolLM2** tended toward higher Faithfulness for formal queries, particularly in definition generation.
- **Ministral** showed the opposite tendency, most prominently in concept identification.
- **Qwen2.5** exhibited the most stable descriptive profile, combining comparatively high Faithfulness, limited formal-colloquial separation, and relatively low within-condition variability.

These findings should not be interpreted as evidence of equivalence between formal and colloquial queries. The study is exploratory and does not establish that CAG is more robust to linguistic register than Retrieval-Augmented Generation (RAG).

## Repository Structure

```text
.
├── data/          # Experimental knowledge and query data
├── notebooks/     # Exploratory and statistical analyses
├── results/       # Experimental outputs and evaluation results
├── src/           # CAG generation and evaluation code
├── requirements.txt
└── README.md
```

Local Python virtual environments are intentionally excluded from version control.

## Reproducibility

Install the required Python dependencies using:

```bash
pip install -r requirements.txt
```

The repository is intended to preserve the experimental configuration used in the study and facilitate independent replication and extension.

Exact execution instructions may depend on local hardware, model availability, Hugging Face configuration, and the Ollama installation used for Faithfulness evaluation.

## Scope

This repository supports an **exploratory controlled experiment** rather than a general benchmark of Spanish linguistic robustness.

The findings are restricted to:

- the formal-colloquial Spanish contrast operationalized in the dataset;
- the three evaluated models;
- the two experimental task types;
- the selected knowledge domain;
- the CAG and Faithfulness evaluation configuration used in the study.

Future work may extend the framework to larger datasets, additional Spanish varieties and registers, broader model families, alternative evaluators, and controlled comparisons between CAG and RAG.

## Citation

A citation entry will be added upon publication of the associated preprint.

## License

License information will be added before the public release of the repository.
