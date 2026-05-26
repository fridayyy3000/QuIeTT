#!/usr/bin/env python3
"""
QUIETT Pipeline Runner
======================
Runs the complete QUIETT pipeline over a dataset directory:
- Steps 1-2 (pipeline_steps_1_2.py): Issue detection + transformation plan generation
- Step 3   (pipeline_step3.py):       Execute transformations to produce canonical.csv
- Step 4   (pipeline_step4.py):       CoT-guided SQL QA on the canonical table

Dataset format
--------------
Place one JSON file per table in DATASET_DIR. Each file should be either:
  A) A list of question objects (the file name becomes the table id):
       [{"qid": "Q1", "text": "...", "ground_truth": ["answer"]}, ...]
     Pair with a same-name .md file for the table markdown.

  B) A single object with embedded data:
       {"table_id": "...", "markdown": "...",
        "questions": [{"qid": "Q1", "text": "...", "ground_truth": ["answer"]}]}
     Or use "csv_path" instead of "markdown" to point at an existing CSV.

Model configuration
-------------------
Define models in models.json (copy models.json.example as a starting point),
or set MODEL_ID / API_TYPE / CLOUD_PROJECT / CLOUD_LOCATION env vars for a
single-model setup.

Usage
-----
    python run_pipeline.py --model default --limit 10
    python run_pipeline.py --model my_model --limit 0
    python run_pipeline.py --all-models --limit 100
"""

import os
import sys
import json
import time
import subprocess
import csv
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# evaluate.py lives one level above this folder (repo root)
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from evaluate import score_prediction

# Global API rate limiter - controls calls across all workers
API_CALL_LOCK = Lock()
LAST_API_CALL_TIME = 0.0
MIN_API_DELAY = 0.2  # Minimum seconds between API calls across all workers

def rate_limited_delay():
    """Add delay to prevent rate limiting across parallel workers."""
    global LAST_API_CALL_TIME
    with API_CALL_LOCK:
        now = time.time()
        elapsed = now - LAST_API_CALL_TIME
        if elapsed < MIN_API_DELAY:
            time.sleep(MIN_API_DELAY - elapsed)
        LAST_API_CALL_TIME = time.time()

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ===================== CONFIGURATION =====================

SCRIPT_DIR = Path(__file__).parent.resolve()

# ---- Data paths ----
# DATASET_DIR: folder containing one JSON file per table.
# COL_DESC_PATH: optional column descriptions file (JSON or JSONL, keyed by table_id).
DATASET_DIR    = Path(os.environ.get("DATASET_DIR",    str(SCRIPT_DIR / "dataset")))
OUTPUT_BASE_DIR = Path(os.environ.get("OUTPUT_DIR",   str(SCRIPT_DIR / "pipeline_output")))
COL_DESC_PATH  = Path(os.environ.get("COL_DESC_PATH", str(SCRIPT_DIR / "col_descriptions.json")))

# ---- Model configuration ----
# Load from models.json if present; otherwise derive a single "default" config from env vars.
_MODELS_CONFIG_FILE = SCRIPT_DIR / "models.json"

def _load_model_configs() -> dict:
    """Return model config dict.  Keys are model aliases used with --model."""
    if _MODELS_CONFIG_FILE.exists():
        with open(_MODELS_CONFIG_FILE) as f:
            return json.load(f)
    # Fallback: single model from env vars
    return {
        "default": {
            "model_id":    os.environ.get("MODEL_ID",          "your-model-id"),
            "api_type":    os.environ.get("API_TYPE",          "sdk"),
            "project":     os.environ.get("CLOUD_PROJECT",     "YOUR_PROJECT_ID"),
            "location":    os.environ.get("CLOUD_LOCATION",    "us-central1"),
            "description": os.environ.get("MODEL_DESCRIPTION", "Default LLM"),
        }
    }

MODEL_CONFIGS = _load_model_configs()

# Pre-loaded column descriptions (populated lazily)
PRELOADED_COL_DESCS: Dict = {}

# Thread safety
print_lock = Lock()
stats_lock = Lock()
log_lock = Lock()

# Global stats
global_stats = {
    'completed': 0,
    'success': 0,
    'failed': 0,
    'cached': 0,
    'total_questions': 0,
    'total_correct': 0,
    'total_f1': 0.0
}

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)

def update_stats(key, value=1):
    with stats_lock:
        global_stats[key] += value


# ===================== SQL LOGGING =====================

def log_sql_query(log_file: Path, table_id: str, question: str, sql: str, answer: str, targets: list, correct: bool):
    """Log SQL query and answer to file."""
    with log_lock:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Table: {table_id}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Target(s): {', '.join(str(t) for t in targets)}\n")
            f.write(f"SQL Query:\n{sql}\n")
            f.write(f"Answer: {answer}\n")
            f.write(f"Correct: {correct}\n")
            f.write("=" * 80 + "\n\n")


# ===================== LOAD COLUMN DESCRIPTIONS =====================

def load_col_descriptions() -> Dict:
    """Load column descriptions from COL_DESC_PATH (JSON object or JSONL).

    Supported formats:
    - JSON object   : {"table_id": {"table_title": ..., "column_descriptions": {...}}}
    - JSON array    : [{"table_id": "...", "table_title": ..., ...}, ...]
    - JSONL         : one JSON object per line, must have a "table_id" or "context" key
    """
    global PRELOADED_COL_DESCS
    if PRELOADED_COL_DESCS:
        return PRELOADED_COL_DESCS
    if not COL_DESC_PATH.exists():
        return {}

    with open(COL_DESC_PATH, encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            PRELOADED_COL_DESCS = data
        elif isinstance(data, list):
            for item in data:
                key = item.get("table_id") or item.get("context", "")
                if key:
                    PRELOADED_COL_DESCS[key] = item
    except json.JSONDecodeError:
        # Try JSONL
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                key = item.get("table_id") or item.get("context", "")
                if key:
                    PRELOADED_COL_DESCS[key] = item
            except Exception:
                pass

    return PRELOADED_COL_DESCS


def get_table_metadata(table_id: str) -> Dict:
    """Return column descriptions for a given table_id (fuzzy-matched)."""
    descs = load_col_descriptions()
    if table_id in descs:
        return descs[table_id]
    # Fuzzy: accept partial key matches
    for key, val in descs.items():
        if key in table_id or table_id in key:
            return val
    return {"table_title": table_id, "column_descriptions": {}}


# ===================== LOAD DATASET =====================

def load_dataset() -> List[Dict]:
    """Scan DATASET_DIR for table JSON files and return a list of table info dicts.

    Supports two JSON formats per file — see module docstring for details.
    """
    if not DATASET_DIR.exists():
        safe_print(f"Dataset directory not found: {DATASET_DIR}")
        return []

    tables = []
    json_files = sorted(DATASET_DIR.glob("*.json"))

    for idx, json_path in enumerate(json_files):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # --- Format A: file IS the questions list ---
        if isinstance(data, list):
            table_id = json_path.stem
            questions = data
            md_path = json_path.with_suffix(".md")
            markdown_content = ""
            if md_path.exists():
                with open(md_path, encoding="utf-8") as f:
                    markdown_content = f.read()
            tables.append({
                "table_id": table_id,
                "index": idx,
                "questions": questions,
                "markdown_content": markdown_content,
                "csv_path": None,
            })

        # --- Format B: object with table_id + questions + markdown/csv_path ---
        elif isinstance(data, dict):
            table_id = data.get("table_id", json_path.stem)
            questions = data.get("questions", [])

            # Markdown: embedded string or external file reference
            markdown_content = data.get("markdown", "")
            if not markdown_content and data.get("markdown_path"):
                md_path = Path(data["markdown_path"])
                if not md_path.is_absolute():
                    md_path = DATASET_DIR / md_path
                if md_path.exists():
                    with open(md_path, encoding="utf-8") as f:
                        markdown_content = f.read()

            tables.append({
                "table_id": table_id,
                "index": idx,
                "questions": questions,
                "markdown_content": markdown_content,
                "csv_path": data.get("csv_path"),
            })

    return tables


# ===================== MARKDOWN TO CSV =====================

def deduplicate_column_names(headers: list) -> list:
    """Make column names unique by adding suffixes based on semantic meaning."""
    seen = {}
    result = []
    
    for i, header in enumerate(headers):
        if header not in seen:
            seen[header] = 0
            result.append(header)
        else:
            seen[header] += 1
            # Try to infer semantic suffix based on position/pattern
            # Common patterns: Name, Time/Value pairs
            suffix_map = {
                1: "_name" if "gold" in header.lower() or "silver" in header.lower() or "bronze" in header.lower() else "_value",
                2: "_time" if seen[header] == 1 else f"_{seen[header]}"
            }
            suffix = suffix_map.get(seen[header], f"_{seen[header]}")
            result.append(f"{header}{suffix}")
    
    return result


def markdown_to_csv(markdown_content: str, output_path: Path) -> bool:
    """Convert markdown table to CSV with intelligent column deduplication.
    
    Handles multi-line headers by joining lines that don't start with |.
    """
    lines = markdown_content.strip().split('\n')
    
    # First pass: merge multi-line cells (lines that don't start with | are continuations)
    merged_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|'):
            merged_lines.append(stripped)
        elif merged_lines and stripped:
            # This is a continuation of the previous cell - append with space
            # Replace newlines in cells with spaces
            merged_lines[-1] = merged_lines[-1].rstrip('|').rstrip() + ' ' + stripped
            if not merged_lines[-1].endswith('|'):
                merged_lines[-1] += ' |'
    
    table_lines = []
    for line in merged_lines:
        line = line.strip()
        if '|' in line:
            if re.match(r'^[\s|:-]+$', line):
                continue
            table_lines.append(line)
    
    if len(table_lines) < 2:
        return False
    
    rows = []
    for idx, line in enumerate(table_lines):
        line = line.strip('|')
        cells = [cell.strip() for cell in line.split('|')]
        
        # Deduplicate header row
        if idx == 0:
            cells = deduplicate_column_names(cells)
        
        rows.append(cells)
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
        return True
    except:
        return False


# ===================== TRANSFORMATION STEPS =====================

def run_transformation_steps(table_id: str, csv_path: Path, run_dir: Path, col_meta_path: Path) -> dict:
    """Run Steps 1-2-3 for table transformation."""
    run_dir.mkdir(parents=True, exist_ok=True)
    
    plan_json_path = run_dir / "plan.json"
    final_csv = run_dir / "final.csv"
    
    result = {'success': False, 'cached': False, 'final_csv': None}
    
    # Check if already done
    if final_csv.exists():
        result['success'] = True
        result['cached'] = True
        result['final_csv'] = final_csv
        return result
    
    input_info = {
        'table_id': table_id,
        'csv_path': str(csv_path),
        'csv_exists': csv_path.exists(),
        'csv_size_bytes': csv_path.stat().st_size if csv_path.exists() else 0,
        'processing_started': datetime.now().isoformat()
    }
    
    # Capture first few rows of input CSV
    if csv_path.exists():
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline() for _ in range(3)]
                input_info['csv_header'] = lines[0].strip() if lines else ""
                input_info['csv_sample_rows'] = [l.strip() for l in lines[1:] if l.strip()]
        except Exception as e:
            input_info['csv_sample_error'] = str(e)
    
    input_log = run_dir / "00_input_info.json"
    with open(input_log, 'w', encoding='utf-8') as f:
        json.dump(input_info, f, indent=2, ensure_ascii=False)
    
    env = os.environ.copy()
    env["PIPELINE_ROOT_OVERRIDE"] = str(run_dir.parent)
    
    # Step 1 & 2: Generate Plan
    if not plan_json_path.exists():
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / 'pipeline_steps_1_2.py'),
            str(csv_path),
            str(col_meta_path)
        ]
        
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            if proc.returncode != 0:
                error_log = run_dir / "test_py_error.log"
                error_log.write_text(f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}", encoding="utf-8")
                return result
        except subprocess.TimeoutExpired:
            return result
    
    if not plan_json_path.exists():
        # Use original CSV as final if no plan
        shutil.copy2(csv_path, final_csv)
        result['success'] = True
        result['final_csv'] = final_csv
        return result
    
    # Step 3: Execute transformations
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / 'pipeline_step3.py'),
        str(csv_path),
        '--run-dir', str(run_dir),
        '--output-file', 'final.csv'
    ]
    
    execution_info = {
        'step3_started': datetime.now().isoformat(),
        'step3_success': False,
        'step3_error': None,
        'plan_validation_retry': False,
    }
    
    def _run_step3():
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    
    try:
        proc = _run_step3()
        
        # ── Paper Algorithm 1: 1 retry on plan-validation failure ────────
        validation_errors_path = run_dir / "validation_errors.json"
        if validation_errors_path.exists():
            execution_info['plan_validation_retry'] = True
            # Discard failed plan + validation error report; regenerate plan once
            try:
                plan_json_path.unlink(missing_ok=True)
                validation_errors_path.unlink(missing_ok=True)
            except Exception:
                pass
            
            # Re-run Step 1 & 2 to produce a fresh plan
            retry_cmd = [
                sys.executable,
                str(SCRIPT_DIR / 'pipeline_steps_1_2.py'),
                str(csv_path),
                str(col_meta_path),
            ]
            try:
                subprocess.run(retry_cmd, capture_output=True, text=True, env=env, timeout=300)
            except subprocess.TimeoutExpired:
                pass
            
            # Re-execute Step 3 against the regenerated plan
            if plan_json_path.exists():
                proc = _run_step3()
        
        if final_csv.exists():
            result['success'] = True
            result['final_csv'] = final_csv
            execution_info['step3_success'] = True
        else:
            # Fallback to original
            shutil.copy2(csv_path, final_csv)
            result['success'] = True
            result['final_csv'] = final_csv
            execution_info['step3_success'] = True
            execution_info['step3_fallback'] = True
        
        # Save step3 output
        step3_log = run_dir / "step3_output.txt"
        step3_log.write_text(f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}", encoding="utf-8")
        
    except subprocess.TimeoutExpired:
        shutil.copy2(csv_path, final_csv)
        result['success'] = True
        result['final_csv'] = final_csv
        execution_info['step3_error'] = 'timeout'
    
    execution_info['step3_finished'] = datetime.now().isoformat()
    
    # Save execution report
    exec_report = run_dir / "execution_report.json"
    with open(exec_report, 'w', encoding='utf-8') as f:
        json.dump(execution_info, f, indent=2, ensure_ascii=False)
    
    return result


# ===================== Q&A WITH SQL =====================

def answer_question_with_sql(question_text: str, transformed_csv: Path, original_csv: Path) -> dict:
    """Answer a question using pipeline_step4 (two-phase CoT-SQL)."""
    sys.path.insert(0, str(SCRIPT_DIR))

    import importlib
    import pipeline_step4
    importlib.reload(pipeline_step4)

    try:
        qa_result = pipeline_step4.answer_question(
            question=question_text,
            transformed_csv_path=str(transformed_csv),
            original_csv_path=str(original_csv)
        )
        return qa_result
    except Exception as e:
        return {"answer": "", "sql": "", "error": str(e)[:200]}


# ===================== PROCESS TABLE =====================

def process_table(table_info: Dict, model_key: str, output_dir: Path, sql_log_file: Path) -> Dict:
    """Process a single table through the full pipeline."""
    
    table_id = table_info["table_id"]
    questions = table_info["questions"]
    markdown_content = table_info.get("markdown_content", "")
    prebuilt_csv   = table_info.get("csv_path")
    
    start = time.time()
    
    result = {
        'table_id': table_id,
        'model': model_key,
        'status': 'unknown',
        'time': 0,
        'cached': False,
        'questions': len(questions),
        'correct': 0,
        'total_f1': 0.0
    }
    
    run_dir = output_dir / table_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already complete (with valid SQL - not empty or SELECT 1)
    qa_results_path = run_dir / "qa_results.json"
    rerun_empty_sql = table_info.get("rerun_empty_sql", False)
    
    if qa_results_path.exists():
        try:
            with open(qa_results_path, 'r') as f:
                qa_results = json.load(f)
            
            # Count empty/fallback SQL
            empty_sql_count = sum(
                1 for r in qa_results 
                if not r.get('sql', '').strip() or r.get('sql', '').strip() == 'SELECT 1'
            )
            
            # Skip cache if rerun_empty_sql flag is set and there are empty SQLs
            if rerun_empty_sql and empty_sql_count > 0:
                pass  # Force re-run
            elif qa_results and all('sql' in r for r in qa_results) and empty_sql_count < len(qa_results) * 0.5:
                # Cache is valid if less than 50% empty SQL
                correct = sum(1 for r in qa_results if r.get("correct", [False, 0])[0])
                total_f1 = sum(r.get("correct", [False, 0])[1] for r in qa_results)
                result['status'] = 'success'
                result['cached'] = True
                result['correct'] = correct
                result['total_f1'] = total_f1
                update_stats('completed')
                update_stats('success')
                update_stats('cached')
                update_stats('total_questions', len(qa_results))
                update_stats('total_correct', correct)
                update_stats('total_f1', total_f1)
                return result
        except:
            pass
    
    if not markdown_content and not (prebuilt_csv and Path(prebuilt_csv).exists()):
        result['status'] = 'no_input'
        update_stats('completed')
        update_stats('failed')
        return result

    # Get metadata (optional column descriptions)
    metadata = get_table_metadata(table_id)

    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Build input CSV from prebuilt CSV or from markdown
    csv_path = run_dir / "input.csv"
    if prebuilt_csv and Path(prebuilt_csv).exists():
        shutil.copy2(prebuilt_csv, csv_path)
    elif not markdown_to_csv(markdown_content, csv_path):
        result['status'] = 'csv_conversion_failed'
        update_stats('completed')
        update_stats('failed')
        return result
    
    # Create column description file for this table
    col_meta = {table_id: metadata.get("raw_description", json.dumps(metadata.get("column_descriptions", {})))}
    col_meta_path = run_dir / "col_meta.json"
    with open(col_meta_path, 'w') as f:
        json.dump(col_meta, f, indent=2)

    # Named copy of input CSV (pipeline_steps_1_2.py uses the stem as table_id)
    table_csv_path = run_dir / f"{table_id}.csv"
    if csv_path != table_csv_path:
        shutil.copy2(csv_path, table_csv_path)
    
    # Run transformation steps
    transform_result = run_transformation_steps(table_id, table_csv_path, run_dir, col_meta_path)
    
    if not transform_result['success']:
        result['status'] = 'transform_failed'
        update_stats('completed')
        update_stats('failed')
        return result
    
    final_csv = transform_result['final_csv']
    
    # Answer questions using SQL with retry for empty SQL
    qa_results = []
    MAX_SQL_RETRIES = 3
    
    for q in questions:
        q_text = q.get("text", "")
        targets = q.get("ground_truth", [])
        
        # Retry loop for empty SQL
        best_result = None
        for retry in range(MAX_SQL_RETRIES):
            qa_result = answer_question_with_sql(q_text, final_csv, table_csv_path)
            
            sql_query = qa_result.get("sql", "")
            
            # If we got valid SQL, use it
            if sql_query and sql_query.strip() and sql_query.strip() != 'SELECT 1':
                best_result = qa_result
                break
            
            # Store this attempt even if SQL is empty
            if best_result is None or (qa_result.get("answer") and not best_result.get("answer")):
                best_result = qa_result
            
            if retry < MAX_SQL_RETRIES - 1:
                time.sleep(1.0)  # Wait before retry
        
        qa_result = best_result if best_result else qa_result
        
        predicted = qa_result.get("answer", "")
        sql_query = qa_result.get("sql", "")

        # Check correctness via evaluate.py
        score = score_prediction(predicted, targets)
        is_correct, f1_score = score["exact_match"], score["f1"]
        
        # Log SQL query
        log_sql_query(sql_log_file, table_id, q_text, sql_query, predicted, targets, is_correct)
        
        qa_results.append({
            "qid": q.get("qid", ""),
            "question": q_text,
            "predicted": predicted,
            "targets": targets,
            "correct": [is_correct, f1_score],
            "sql": sql_query,
            "failure_reason": q.get("failure_reason", ""),
            "reasoning": qa_result.get("reasoning", "")[:300]
        })
        
        time.sleep(0.3)
    
    # Save results
    with open(qa_results_path, 'w') as f:
        json.dump(qa_results, f, indent=2, ensure_ascii=False)
    
    correct = sum(1 for r in qa_results if r.get("correct", [False, 0])[0])
    total_f1 = sum(r.get("correct", [False, 0])[1] for r in qa_results)
    total_questions = len(qa_results)
    
    # Save qa_summary.json
    qa_summary = {
        'table_id': table_id,
        'model': model_key,
        'original_csv': str(csv_path),
        'transformed_csv': str(final_csv),
        'timestamp': datetime.now().isoformat(),
        'total_questions': total_questions,
        'correct_answers': correct,
        'accuracy_percent': (correct / total_questions * 100) if total_questions > 0 else 0,
        'avg_f1_score': (total_f1 / total_questions) if total_questions > 0 else 0,
    }
    
    qa_summary_path = run_dir / "qa_summary.json"
    with open(qa_summary_path, 'w') as f:
        json.dump(qa_summary, f, indent=2, ensure_ascii=False)
    
    # Save questions.json (the input questions)
    questions_path = run_dir / "questions.json"
    with open(questions_path, 'w') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    result['status'] = 'success'
    result['correct'] = correct
    result['total_f1'] = total_f1
    result['time'] = time.time() - start
    
    update_stats('completed')
    update_stats('success')
    update_stats('total_questions', len(qa_results))
    update_stats('total_correct', correct)
    update_stats('total_f1', total_f1)
    
    return result


# ===================== MAIN PIPELINE =====================

def run_pipeline(model_key: str, limit: int = 0, start_idx: int = None, end_idx: int = None, force: bool = False, rerun_empty_sql: bool = False):
    """Run the full pipeline for a model."""
    
    config = MODEL_CONFIGS[model_key]
    
    # Set environment for model
    os.environ["MODEL_ID"]      = config["model_id"]
    os.environ["API_TYPE"]      = config["api_type"]
    os.environ["CLOUD_PROJECT"] = config["project"]
    os.environ["CLOUD_LOCATION"] = config["location"]
    os.environ["LLM_MAX_TOKENS"] = "65536"

    safe_print(f"\n{'='*70}")
    safe_print(f"QUIETT Pipeline — {config['description']}")
    safe_print(f"Model: {config['model_id']}")
    if rerun_empty_sql:
        safe_print("Re-running tables with empty SQL")
    safe_print(f"{'='*70}")

    output_dir = OUTPUT_BASE_DIR / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    sql_log_file = output_dir / "sql_query_log.txt"

    # Load column descriptions (optional; silently skipped if file not present)
    load_col_descriptions()
    safe_print(f"Column descriptions loaded: {len(PRELOADED_COL_DESCS)} entries")

    # Load dataset
    tables = load_dataset()
    safe_print(f"Loaded {len(tables)} tables from {DATASET_DIR}")
    
    # Apply start/end filtering
    if start_idx is not None or end_idx is not None:
        start = start_idx or 0
        end = end_idx or len(tables)
        tables = tables[start:end]
        safe_print(f"📋 Filtering to tables [{start}:{end}] = {len(tables)} tables")
    
    if limit > 0:
        tables = tables[:limit]
        safe_print(f"📋 Limiting to {limit} tables")
    
    # Add rerun_empty_sql flag to each table_info
    for t in tables:
        t['rerun_empty_sql'] = rerun_empty_sql
    
    # Reset stats
    global global_stats
    global_stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in global_stats.items()}
    
    start_time = time.time()
    
    # Get worker count from args (passed through run_pipeline signature)
    num_workers = getattr(run_pipeline, '_num_workers', 5)
    safe_print(f"🔧 Using {num_workers} parallel workers")
    
    if HAS_TQDM:
        pbar = tqdm(total=len(tables), desc=f"Processing ({model_key})", unit="table")
    
    def process_with_force(table_info):
        """Wrapper to handle force flag with rate limiting."""
        # Add delay between tables to prevent rate limiting
        time.sleep(0.3)  # 0.3 second delay between table starts
        
        if force:
            table_output_dir = output_dir / f"{table_info['index']}"
            if table_output_dir.exists():
                shutil.rmtree(table_output_dir)
        return process_table(table_info, model_key, output_dir, sql_log_file)
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_with_force, t): t for t in tables}
        
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                safe_print(f"Error processing table: {e}")
            
            if HAS_TQDM:
                pbar.update(1)
                if global_stats['total_questions'] > 0:
                    acc = global_stats['total_correct'] / global_stats['total_questions'] * 100
                    avg_f1 = global_stats['total_f1'] / global_stats['total_questions']
                    pbar.set_postfix({'Acc': f"{acc:.1f}%", 'F1': f"{avg_f1:.3f}", 'Cached': global_stats['cached']})
    
    if HAS_TQDM:
        pbar.close()
    
    elapsed = time.time() - start_time
    
    # Final stats
    total_q = global_stats['total_questions']
    total_c = global_stats['total_correct']
    total_f1 = global_stats['total_f1']
    
    accuracy = (total_c / total_q * 100) if total_q > 0 else 0
    avg_f1 = (total_f1 / total_q) if total_q > 0 else 0
    
    safe_print(f"\n{'='*70}")
    safe_print(f"📊 RESULTS - {config['description']}")
    safe_print(f"{'='*70}")
    safe_print(f"   Tables Processed: {global_stats['completed']}")
    safe_print(f"   Tables Cached: {global_stats['cached']}")
    safe_print(f"   Total Questions: {total_q}")
    safe_print(f"   Correct Answers: {total_c}")
    safe_print(f"   Accuracy: {accuracy:.2f}%")
    safe_print(f"   Average F1 Score: {avg_f1:.4f}")
    safe_print(f"   Time: {elapsed:.1f}s")
    safe_print(f"   SQL Log: {sql_log_file}")
    safe_print(f"{'='*70}\n")
    
    summary = {
        "model": model_key,
        "model_id": config["model_id"],
        "timestamp": datetime.now().isoformat(),
        "tables_processed": global_stats['completed'],
        "tables_cached": global_stats['cached'],
        "total_questions": total_q,
        "correct_answers": total_c,
        "accuracy_percent": accuracy,
        "average_f1_score": avg_f1,
        "elapsed_seconds": elapsed
    }
    
    with open(output_dir / "pipeline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    import argparse

    model_keys = list(MODEL_CONFIGS.keys())
    parser = argparse.ArgumentParser(description="QUIETT Pipeline Runner")
    parser.add_argument("--model", choices=model_keys, default=model_keys[0],
                        help=f"Model to use (defined in models.json or env vars). Choices: {model_keys}")
    parser.add_argument("--all-models", action="store_true", help="Run all configured models sequentially")
    parser.add_argument("--limit",   type=int, default=0, help="Max tables to process (0 = all)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers")
    parser.add_argument("--start",   type=int, default=None, help="Start from table N (0-indexed)")
    parser.add_argument("--end",     type=int, default=None, help="End at table N (exclusive)")
    parser.add_argument("--force",   action="store_true", help="Force re-run even if cached")
    parser.add_argument("--rerun-empty-sql", action="store_true", help="Re-run tables with empty SQL")
    
    args = parser.parse_args()
    
    # Store range arguments globally for filtering
    global START_TABLE, END_TABLE, FORCE_RERUN, RERUN_EMPTY_SQL
    START_TABLE = args.start
    END_TABLE = args.end
    FORCE_RERUN = args.force
    RERUN_EMPTY_SQL = getattr(args, 'rerun_empty_sql', False)
    
    if args.all_models:
        models = list(MODEL_CONFIGS.keys())
    else:
        models = [args.model]
    
    all_summaries = []
    for model_key in models:
        # Pass workers to run_pipeline via function attribute
        run_pipeline._num_workers = args.workers
        summary = run_pipeline(
            model_key, args.limit, 
            start_idx=args.start, end_idx=args.end, 
            force=args.force, rerun_empty_sql=RERUN_EMPTY_SQL
        )
        all_summaries.append(summary)
    
    if len(all_summaries) > 1:
        safe_print(f"\n{'='*70}")
        safe_print("📊 MODEL COMPARISON")
        safe_print(f"{'='*70}")
        safe_print(f"{'Model':<20} {'Accuracy':>10} {'F1 Score':>10} {'Questions':>12}")
        safe_print("-" * 70)
        for s in all_summaries:
            safe_print(f"{s['model']:<20} {s['accuracy_percent']:>9.2f}% {s['average_f1_score']:>10.4f} {s['total_questions']:>12}")
        safe_print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
