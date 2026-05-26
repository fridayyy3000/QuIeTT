# QuIeTT: Query Independent Table Transformation For Robust Reasoning

**QuIeTT** is a three-stage pipeline that converts each raw table into a single SQL-ready canonical representation before any evaluation query is observed: (1) issue probing, where synthetic queries generated from the raw table surface structural deficiencies such as ambiguous schemas, heterogeneous formats, and implicit relational structure; (2) transformation planning, which generates an ordered sequence of operations to resolve detected issues; and (3) structured execution, which materializes the canonical table by applying deterministic operators and LLM-generated code, cleanly separating table preprocessing from downstream reasoning.

---

## Overview

![QuIeTT Overview](assets/Quiett_Pipeline.pdf)

---

## Repository Structure

```
Code/
├── pipeline_steps_1_2.py   # Steps 1 & 2 — issue detection + transformation plan generation
├── pipeline_step3.py        # Step 3  — deterministic plan execution
├── pipeline_step4.py        # Step 4  — CoT-guided SQL question answering
├── ops_runtime.py           # ~21 deterministic table operators (used by Step 3)
├── validate_plan.py         # JSON schema validator for Step 2 output plans
├── operation_schema.json    # Formal operator specification
├── run_pipeline.py          # Batch orchestrator for dataset-level evaluation
├── models.json.example      # Model configuration template
data_preprocessing/
├── flattening_tables.py     # Preprocessing for hierarchical tables (e.g. HiTab) → flat CSVs
evaluate.py              # Standalone Exact Match + token-F1 scorer
requirements.txt
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Authentication

QUIETT calls LLM APIs via either the Google GenAI SDK or an OpenAI-compatible REST endpoint.
Both require a valid access token from the Google Cloud CLI:

```bash
gcloud auth application-default login
```

---

## Model Configuration

Copy `models.json.example` to `models.json` and fill in your model details:

```json
{
  "my_model": {
    "model_id":    "gemini-2.5-flash",
    "api_type":    "sdk",
    "project":     "YOUR_GCP_PROJECT_ID",
    "location":    "us-central1",
    "description": "Gemini 2.5 Flash"
  }
}
```

`api_type` must be one of:
- `"sdk"` — Google GenAI Python SDK (recommended for Gemini-family models)
- `"openapi"` — OpenAI-compatible REST endpoint (`chat/completions`)

Alternatively, set environment variables directly (no `models.json` needed):

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `your-model-id` | Model identifier |
| `API_TYPE` | `sdk` | `sdk` or `openapi` |
| `CLOUD_PROJECT` | `YOUR_PROJECT_ID` | Google Cloud project ID |
| `CLOUD_LOCATION` | `us-central1` | Cloud region |
| `LLM_MAX_TOKENS` | `65536` | Maximum output tokens |

Per-stage token limits (matching paper §3.2):

| Variable | Default | Stage |
|---|---|---|
| `STEP1_MAX_TOKENS` | `8000` | Issue detection |
| `STEP2_MAX_TOKENS` | `6000` | Plan generation |
| `STEP4_MAX_TOKENS` | `4096` | CoT SQL QA |

---

## Dataset Format

Place one JSON file per table in a directory (set via `DATASET_DIR`).

**Format A** — questions list + paired markdown file:
```
dataset/
  my_table.json   # list of question objects
  my_table.md     # markdown table (same stem)
```
```json
[
  {"qid": "Q1", "text": "How many rows have value > 10?", "ground_truth": ["3"]}
]
```

**Format B** — self-contained object:
```json
{
  "table_id": "my_table",
  "markdown": "| Col A | Col B |\n|-------|-------|\n| ...",
  "questions": [
    {"qid": "Q1", "text": "...", "ground_truth": ["..."]}
  ]
}
```
You can also use `"csv_path"` or `"markdown_path"` instead of embedding the content directly.

**Column descriptions** (optional): provide a JSON file at `COL_DESC_PATH` keyed by `table_id`
to supply human-readable column context to the LLM.

---

## Data Preprocessing — Hierarchical Tables

QUIETT expects flat CSV tables as input. If your dataset uses a hierarchical table format
(e.g. **HiTab**, where row/column headers form a multi-level tree), you must flatten the
tables first using the provided preprocessing script.

### When to use this

Use `data_preprocessing/flattening_tables.py` whenever your tables have:
- Multi-level row headers (e.g. region → country → player)
- Multi-level column headers (e.g. year → competition → metric)
- A JSON structure with `top_root` / `left_root` hierarchy trees and a `data` cell matrix

This covers datasets such as HiTab and any other table corpus that follows the same
hierarchical JSON schema.

### Usage

```bash
python data_preprocessing/flattening_tables.py \
  --input  /path/to/hierarchical_json_tables \
  --output /path/to/flattened_csvs
```

Each `.json` file in `--input` is flattened into a corresponding `.csv` file in `--output`.
Left-hierarchy levels are stored as `row_level_0`, `row_level_1`, … columns; top-hierarchy
paths become the column headers.

Once flattening is complete, point `DATASET_DIR` at the output folder and run QUIETT as normal.

### Full workflow for hierarchical datasets

```bash
# Step 0 — flatten hierarchical tables
python data_preprocessing/flattening_tables.py \
  --input  /path/to/hitab/test_tables \
  --output /path/to/flattened_csvs

# Step 1–4 — run QUIETT on the flattened CSVs
export DATASET_DIR=/path/to/flattened_csvs
export OUTPUT_DIR=/path/to/output
python Code/run_pipeline.py --model my_model
```

---

## Running the Pipeline

### Batch evaluation

```bash
export DATASET_DIR=/path/to/dataset
export OUTPUT_DIR=/path/to/output

python run_pipeline.py --model my_model --limit 100
python run_pipeline.py --model my_model              # full dataset
python run_pipeline.py --all-models                  # all models in models.json
python run_pipeline.py --model my_model --force      # ignore cache, re-run all
```

Key flags:

| Flag | Description |
|---|---|
| `--model` | Model key from `models.json` (default: first key) |
| `--all-models` | Run all configured models sequentially |
| `--limit N` | Process only the first N tables (0 = all) |
| `--start N` / `--end N` | Slice the table list by index |
| `--workers N` | Parallel workers (default: 5) |
| `--force` | Force re-run even if results are cached |
| `--rerun-empty-sql` | Re-run tables where SQL generation failed |

### Running individual steps

**Steps 1 & 2** (issue detection + plan generation):
```bash
python pipeline_steps_1_2.py /path/to/table.csv /path/to/col_descriptions.json
```

**Step 3** (plan execution):
```bash
python pipeline_step3.py /path/to/table.csv --run-dir /path/to/run/ --output-file final.csv
```

**Step 4** (CoT SQL QA):
```bash
python pipeline_step4.py --csv /path/to/canonical.csv --question "Which team scored the most goals?"
```

---

## Evaluation

`evaluate.py` computes **Exact Match (EM)** and **token-level F1** over pipeline outputs.
EM is strictly string-based after normalisation (no threshold inflation). F1 is reported
separately as a soft overlap metric.

```bash
# Pipeline output already contains targets (qa_results.json):
python evaluate.py pipeline_output/my_model/my_table/qa_results.json

# Separate gold file:
python evaluate.py predictions.json --gold gold.json

# Save per-question results:
python evaluate.py pipeline_output/my_model/my_table/qa_results.json --output eval_results.json
```

Output:
```
============================================================
  Questions      : 500
  Exact Match    : 312 / 500  (62.4%)
  Avg token-F1   : 0.7031
============================================================
```

---

## Pipeline Details

### Step 1 — Issue Detection
The LLM receives the raw table and column descriptions and outputs a list of quality issues
(e.g., merged cells, inconsistent units, transposed headers, missing values).

### Step 2 — Transformation Plan Generation
Given the detected issues, the LLM produces a structured JSON plan consisting of ordered
operators from a fixed vocabulary (defined in `operation_schema.json`).

### Step 3 — Plan Execution
Each operator in the plan is executed deterministically by `ops_runtime.py`.
For `custom` operators the LLM generates a short pandas code snippet.
The output is a canonical CSV (`final.csv`).

### Step 4 — CoT SQL QA
A single chain-of-thought prompt presents the canonical table with column descriptions
and instructs the model to reason step-by-step, produce an `<sql_plan>`, then output a
SQL query. The query is executed against an in-memory SQLite database to produce the answer.

---

## Hyperparameters

All settings match the paper unless noted.

| Parameter | Value | Notes |
|---|---|---|
| Temperature (all stages) | 0.2 | — |
| Step 1 max tokens | 8 000 | Issue detection |
| Step 2 max tokens | 6 000 | Plan generation |
| Step 3 max tokens | 1 024 | Code generation |
| Step 4 max tokens | 4 096 | CoT SQL QA |
| Rate-limit retries | 3 | Exponential backoff |
| SQL retries (empty) | 3 | Per question |

---


