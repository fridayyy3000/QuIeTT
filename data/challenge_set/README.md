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

- Tables   : 529
- Questions: 2,500
- Questions per table: mean 4.7 / median 5 / max 7
- Sources  : WikiTQ and NQ-Tables
