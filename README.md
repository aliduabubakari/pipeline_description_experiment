# Prompt2Workflow: How Pipeline Description Quality Governs Multi-Orchestrator Workflow Generation

A controlled empirical study measuring how pipeline description quality affects LLM-generated workflow code across Apache Airflow, Prefect, and Dagster.

---

## Overview

**Prompt2Workflow** investigates a critical question in AI-assisted workflow engineering: *Which information gaps in pipeline descriptions most degrade the quality of generated workflow code?*

Large language models can synthesize workflow orchestration code from natural language, but real-world descriptions are often underspecified—missing scheduling details, unclear dependencies, omitted external system identifiers, and unstated runtime assumptions. This project provides a controlled experimental framework to measure the causal impact of specific description omissions on workflow generation quality across multiple orchestrators.

### Key Features

- **Controlled ablation benchmark**: 10 deterministic description variants (C0–C9) per pipeline
- **Multi-orchestrator support**: Generate workflows for Airflow, Prefect, and Dagster
- **Orchestrator-agnostic specification**: JSON-based `PipelineSpec` schema
- **Comprehensive evaluation suite**: Static analysis (SAT), platform compliance (PCT), and semantic fidelity
- **Reproducible experiments**: Structured JSON outputs, CSV exports, and pinned dependencies

---

## Quick Start

### Prerequisites

- Python 3.10+
- Apache Airflow 2.8.4 (pinned via constraints)
- Prefect, Dagster
- API keys for LLM providers (configured in `config_llm.json` and `config_reasoning_llm.json`)

### Installation

```bash
# Create virtual environment
python3.10 -m venv .venv-airflow284
source .venv-airflow284/bin/activate
pip install -U pip

# Install Apache Airflow 2.8.4 with constraints
pip install "apache-airflow==2.8.4" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.4/constraints-3.10.txt"

# Install Airflow providers
pip install apache-airflow-providers-docker \
  apache-airflow-providers-http \
  apache-airflow-providers-postgres \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.4/constraints-3.10.txt"

# Install other orchestrators
pip install prefect dagster

# Install evaluation dependencies
pip install pyyaml rouge-score bert-score torch transformers
pip install pylint radon bandit flake8
```

### Environment Check

```bash
# Verify environment before running experiments
python scripts/evaluators/env_preflight_check.py --strict --airflow-version 2.8.4

# Verify CodeBERT for semantic analysis
python -c "from bert_score import BERTScorer; BERTScorer(model_type='microsoft/codebert-base', num_layers=12); print('OK')"
```

---

## Project Structure

```
pipeline_description_experiment/
├── config_llm.json                    # Non-reasoning model configs
├── config_reasoning_llm.json          # Reasoning model configs
├── pipeline_specs/                    # Oracle PipelineSpec files (JSON)
│   ├── *.pipelinespec.json
│   └── MANIFEST.json
├── pipeline_variants/                 # Description variants (C0-C9)
│   ├── <pipeline_id>/
│   │   ├── C0_FULL_STRUCTURED.txt
│   │   ├── C1_NO_SCHEDULING.txt
│   │   ├── ...
│   │   ├── C9_PROSE_UNSTRUCTURED.txt
│   │   └── specs/                     # Optional variant specs
│   └── MANIFEST.json
├── selected_batch_dags/               # Input DAG metadata (JSON)
├── scripts/
│   ├── generate_pipelinespecs_phase1.py
│   ├── render_variants_phase2.py
│   ├── run_batch_experiments_4_pipeline_description_experiement.py
│   ├── direct_prompting_non_reasoning.py
│   ├── direct_prompting_reasoning.py
│   ├── direct_prompting_react_agent.py
│   ├── evaluators/
│   │   ├── enhanced_static_analyzer.py
│   │   ├── platform_compliance_tester.py
│   │   ├── semantic_analyzer.py
│   │   ├── unified_evaluator.py
│   │   └── export_unified_to_csv.py
│   └── utils/
└── README.md
```

---

## Workflow: From DAGs to Results

### Phase 1: Generate Oracle Specifications

Convert input DAGs into orchestrator-agnostic `PipelineSpec` JSON files:

```bash
python scripts/generate_pipelinespecs_phase1.py \
  --input selected_batch_dags \
  --output pipeline_specs
```

**Output**: `pipeline_specs/*.pipelinespec.json`

Each spec contains:
- Steps (id, objective, mechanism, parameters, env/infra)
- Control flow (edges, branches, gates)
- External systems and data artifacts
- Failure handling semantics

---

### Phase 2: Render Description Variants

Generate 10 controlled description classes (C0–C9) per pipeline:

```bash
python scripts/render_variants_phase2.py \
  --specs-dir pipeline_specs \
  --output pipeline_variants
```

**Output**: `pipeline_variants/<pipeline_id>/C*.txt`

#### Description Classes (C0–C9)

| Class | Description | What's Removed |
|-------|-------------|----------------|
| **C0** | `FULL_STRUCTURED` | Nothing (complete oracle) |
| **C1** | `NO_SCHEDULING` | Schedule, start date, catchup |
| **C2** | `NO_CONTROL_FLOW` | Dependencies, edges, branching, gates |
| **C3** | `NO_EXECUTION_MECHANISM` | Mechanism labels (container_run, sql_query, etc.) |
| **C4** | `NO_STEP_PARAMETERS` | Commands, images, SQL, endpoints, poke intervals |
| **C5** | `NO_EXTERNAL_SYSTEM_DETAILS` | DB names, API endpoints, connection IDs |
| **C6** | `NO_DATA_ARTIFACT_DETAILS` | Table names, file paths, buckets, topics |
| **C7** | `NO_ENV_INFRA_DETAILS` | Env vars, mounts, networks, runtime config |
| **C8** | `NO_FAILURE_HANDLING_DETAILS` | Retries, timeouts, alerts, idempotency |
| **C9** | `PROSE_UNSTRUCTURED` | Nothing removed, but unstructured prose format |

---

### Phase 3: Generate Workflow Code

Generate workflows across orchestrators, models, and description classes:

```bash
python scripts/run_batch_experiments_4_pipeline_description_experiement.py \
  --prompts-dir pipeline_variants \
  --pipeline-specs-dir pipeline_specs \
  --pipeline-variants-dir pipeline_variants \
  --config-nonreasoning config_llm.json \
  --config-reasoning config_reasoning_llm.json \
  --modes non_reasoning \
  --nonreasoning-models Qwen3-Coder,deepseek_ai \
  --orchestrators airflow,prefect,dagster \
  --classes C0,C1,C2,C3,C4,C5,C6,C7,C8,C9 \
  --repetitions 3 \
  --generation-root generated_runs \
  --eval-root eval_results/unified \
  --semantic \
  --semantic-device mps \
  --export-csv results/unified_summary.csv \
  --export-issues-csv results/unified_issues.csv
```

**Generation Modes**:
- `non_reasoning`: Direct one-shot generation
- `reasoning`: Reasoning models with code extraction
- `react`: Iterative plan → generate → validate → repair loop

**Output**:
- Generated code: `generated_runs/<run_id>/*.py`
- Metadata: `generated_runs/<run_id>/generation_metadata.json`

---

### Phase 4: Evaluation

Evaluation runs automatically during batch experiments, but can also be run standalone:

```bash
python scripts/evaluators/unified_evaluator.py \
  --code-file generated_runs/<run_id>/workflow.py \
  --orchestrator airflow \
  --pipeline-spec pipeline_specs/<pipeline_id>.pipelinespec.json \
  --variant-spec pipeline_variants/<pipeline_id>/specs/C1.json \
  --output eval_results/unified/<run_id>.unified.json \
  --semantic \
  --semantic-device mps
```

#### Evaluation Components

**1. SAT (Static Analysis Tester)**
- Overall score: 0–10 (unweighted mean of dimensions)
- Dimensions: correctness, code_quality, best_practices, maintainability, robustness
- Tools: `pylint`, `radon`, `bandit`, optional `flake8`
- **Penalty-free**: Issues logged but don't subtract from scores

**2. PCT (Platform Compliance Tester)**
- Overall score: 0–10 (dimension-based, penalty-free)
- Gate checks: syntax_valid, minimum_structure, critical_imports
- Dimensions: loadability, structure_validity, configuration_validity, task_validity, executability
- Issues exported for post-hoc penalty analysis

**3. Semantic Analyzer**
- ROUGE-1, ROUGE-L, BERTScore (CodeBERT with `num_layers=12`)
- Oracle fidelity: code vs oracle spec
- Variant fidelity: code vs prompt variant spec
- Scores: structural_fidelity (0–10), full_fidelity (0–10), combined (0–10)

#### Combined Paper Score

```
S_code = α × SAT + (1-α) × PCT
```

Subject to gating:
- If PCT gates fail → `S_code = 0`
- Optional YAML validity gate

**Semantic fidelity reported separately** (not included in `S_code`)

---

### Phase 5: Export and Analysis

Convert JSON results to analysis-ready CSV:

```bash
python scripts/evaluators/export_unified_to_csv.py \
  --input eval_results/unified \
  --out results/unified_summary.csv \
  --issues-out results/unified_issues.csv
```

**Output**:
- `unified_summary.csv`: One row per run (pipeline_id, class_id, orchestrator, model, SAT/PCT/semantic scores, tokens, etc.)
- `unified_issues.csv`: One row per issue (for penalty analysis)

---

## Research Questions

**RQ1**: How does pipeline description quality affect static correctness (SAT), platform compliance (PCT), and semantic fidelity?

**RQ2**: Which omitted information categories (C1–C9) cause the largest degradation in structural fidelity, integration fidelity, and operational reliability?

**RQ3**: Do Airflow, Prefect, and Dagster differ in robustness to description omissions?

**RQ4**: Do omissions impact non-linear workflows (fan-out/fan-in, branch/merge, sensor-gated) more than linear pipelines?

---

## Dataset

**40 pipelines** spanning diverse workflow patterns:

| Pattern | Count | Examples |
|---------|-------|----------|
| Linear | 1 | Simple sequential ETL |
| Fan-out only | 2 | Multi-region replication |
| Fan-out/fan-in | 4 | Aggregation pipelines |
| Branch/merge | 5 | Conditional routing |
| Sensor-gated | 4 | File arrival watchers |
| Staged ETL | 2 | Multi-stage data processing |
| Mixed | 22 | Real-world DAGs from GitHub |

---

## Configuration Files

### `config_llm.json` (Non-reasoning Models)
```json
{
  "providers": {
    "Qwen3-Coder": {
      "type": "azure",
      "azure_endpoint": "...",
      "deployment_name": "Qwen-2.5-Coder-32B-Instruct",
      "api_version": "2024-10-21",
      "api_key_env": "AZURE_OPENAI_API_KEY"
    }
  }
}
```

### `config_reasoning_llm.json` (Reasoning Models)
```json
{
  "providers": {
    "deepseek_reasoning": {
      "type": "openai",
      "base_url": "https://api.deepseek.com",
      "model_name": "deepseek-reasoner",
      "api_key_env": "DEEPSEEK_API_KEY"
    }
  }
}
```

---

## Reproducibility Notes

### Determinism
- Set `temperature=0.0` in generation configs for deterministic outputs
- Use repetitions to measure provider nondeterminism or stochastic variance

### Version Pinning
- **Airflow**: Locked to 2.8.4 via constraints
- **Providers**: Compatible versions from Airflow constraints
- **Python**: 3.10+ recommended

### Semantic Analysis
- **CodeBERT**: Explicitly set `num_layers=12` for BERTScore
- **Device**: Supports `cpu`, `cuda`, `mps` (Apple Silicon)

---

## Limitations

1. **Semantic analyzer is deterministic but imperfect**:
   - Dynamic code can hide identifiers from AST
   - Prefect/Dagster edge inference is best-effort

2. **PCT depends on installed orchestrator versions**:
   - Results may vary slightly across minor version updates

3. **Repetitions at temperature=0 may be identical**:
   - Use temperature > 0 to measure stochastic variance

4. **Signature extraction limitations**:
   - Complex dynamic workflows may not fully extract all semantic elements

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{prompt2workflow2025,
  title={Prompt2Workflow: How Pipeline Description Quality Governs Multi-Orchestrator Workflow Generation},
  year={2026}
}
```

---

## Contributing

This is a research project. For questions or collaboration:
- Open an issue in the repository
- Contact: [a.alidu@campus.unimib.it]

---

## License

Apache 2.0

---

## Acknowledgments

This work builds on real-world Airflow DAGs from GitHub and synthetic pipelines designed to stress-test workflow generation across diverse patterns and integration types.