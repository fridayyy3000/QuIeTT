# QUIETT Challenge Set

This folder contains the released QUIETT evaluation dataset.

## File

- `dataset.jsonl` — one JSON object per line; each line is one table with its questions

## Usage

```bash
python run_pipeline.py --model default --dataset_dir data/challenge_set
```

## Loading the dataset

```python
import json

with open("data/challenge_set/dataset.jsonl") as f:
    dataset = [json.loads(line) for line in f]

# Each entry:
# {
#   "table_id":    "1",
#   "table_title": "Chesapeake and Delaware Canal",
#   "markdown":    "| Col | ... |",
#   "questions": [
#     {"qid": "1", "text": "...", "ground_truth": ["..."]}
#   ]
# }
```

## Dataset Statistics

- Tables   : 529
- Questions: 2,500
- Questions per table: mean 4.7 / median 5 / max 7
- Sources  : WikiTQ and NQ-Tables
