# QUIETT Challenge Set

This folder contains the released QUIETT evaluation dataset.

## Structure

- `tables/`        — one JSON file per table: `{table_id, table_title, markdown}`
- `questions.jsonl`— one line per question: `{table_id, qid, text, ground_truth}`

## Usage

```bash
python run_pipeline.py --model default --dataset_dir data/challenge_set
```

## Loading the dataset

```python
import json
from pathlib import Path

tables_dir = Path("data/challenge_set/tables")
tables = {
    int(f.stem): json.loads(f.read_text())
    for f in sorted(tables_dir.glob("*.json"), key=lambda f: int(f.stem))
}

questions = {}  # table_id -> [question, ...]
with open("data/challenge_set/questions.jsonl") as f:
    for line in f:
        q = json.loads(line)
        questions.setdefault(q["table_id"], []).append(q)
```

## Dataset Statistics

| Statistic | Values | Table Dimension |
|---|---|---|
| Total tables | 529 | Mean: (20, 11) |
| Total questions | 2,500 | Median: (8, 8) |
| Questions per table (mean / median / max) | 4 / 5 / 7 | Min: (1, 1) |
| | | Max: (380, 76) |

The set comprises 2,500 questions over 529 tables. Tuple-valued entries in the Table Dimension column report (rows, columns); the wide size range reflects the structural diversity the challenge set targets.

- Sources: WikiTQ and NQ-Tables
