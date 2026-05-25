#!/usr/bin/env python3

import os
import sys
import json
import sqlite3
import re
import time
import csv
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import pandas as pd

# Direct HTTP requests for GPT-OSS-120B via Vertex AI Global OpenAPI
import requests

# ===================== LLM Configuration =====================
# All values can be overridden via environment variables.
PROJECT_ID = os.environ.get("CLOUD_PROJECT", "YOUR_PROJECT_ID")
LOCATION   = os.environ.get("CLOUD_LOCATION", "us-central1")
MODEL_ID   = os.environ.get("MODEL_ID", "your-model-id")
API_TYPE   = os.environ.get("API_TYPE", "sdk")
STEP4_MAX_TOKENS = int(os.environ.get("STEP4_MAX_TOKENS", "4096"))  # CoT SQL QA (paper §3.4)
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "65536"))        # Fallback

def _get_gcloud_token():
    """Get auth token from gcloud."""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        raise Exception(f"Failed to get gcloud token: {e}")

# ===================== PROMPT (paper §3.4 — CoT→SQL) =====================

PROMPT = """You are answering a question about a structured table.
Analyze the table carefully and generate a SQL query to retrieve the answer.

## TABLE: {table_name}

### Columns:
{column_description}

### Data (up to 50 rows shown):
{table_data}

---

## QUESTION: {question}

---

### Format (follow this exactly):

<reasoning>
1. Interpret the question — what value or fact is it asking for?
2. Identify relevant columns — which column(s) hold that data?
3. Determine row-level filters — any WHERE conditions? (entity name, date, value)
4. Identify the required operation:
   - Direct lookup → SELECT col WHERE ...
   - Count rows → COUNT
   - Maximum / minimum → MAX / MIN or ORDER BY ... LIMIT 1
   - Sum / average → SUM / AVG
5. Confirm the answer from the data, then write the SQL.
</reasoning>

<sql_plan>
SELECT: [target column(s)]
FROM: {table_name}
WHERE: [filter condition, or —]
ORDER BY: [ordering, or —]
AGGREGATION: [function, or —]
</sql_plan>

```sql
SELECT ... FROM "{table_name}" ...
```

---

### Example

**TABLE:** medals

**Columns:**
- Country (TEXT): e.g. USA, China, UK
- Gold (INTEGER): e.g. 10, 7, 3
- Silver (INTEGER): e.g. 8, 5, 4
- Total (INTEGER): e.g. 28, 18, 10

**Data:**
| Country | Gold | Silver | Total |
|---------|------|--------|-------|
| USA     | 10   | 8      | 28    |
| China   | 7    | 5      | 18    |
| UK      | 3    | 4      | 10    |

**QUESTION:** Which country won the most gold medals?

<reasoning>
1. The question asks for the country with the highest number of gold medals.
2. Relevant columns: Country (the answer), Gold (used to rank).
3. No entity-level filter — we want the maximum across all rows.
4. Operation: ORDER BY Gold DESC LIMIT 1 to get the top row.
5. USA has Gold=10, the highest value, so the answer should be USA.
</reasoning>

<sql_plan>
SELECT: Country
FROM: medals
WHERE: —
ORDER BY: Gold DESC
AGGREGATION: LIMIT 1
</sql_plan>

```sql
SELECT "Country" FROM "medals" ORDER BY "Gold" DESC LIMIT 1
```
"""

# ===================== MODEL API =====================

def call_llm_openapi(prompt: str, temperature: float = 0.2, max_tokens: int = None) -> str:
    """Call any OpenAI-compatible REST endpoint (chat/completions)."""
    if max_tokens is None:
        max_tokens = MAX_TOKENS

    url = (
        f"https://aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/global/endpoints/openapi/chat/completions"
    )
    headers = {
        "Authorization": f"Bearer {_get_gcloud_token()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": min(max_tokens, MAX_TOKENS),
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise Exception(f"No response from {MODEL_ID}")
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        raise Exception(f"Empty response from {MODEL_ID}")
    return content


def call_llm_sdk(prompt: str, temperature: float = 0.2, max_tokens: int = None) -> str:
    """Call a model via the google-genai SDK."""
    from google import genai
    from google.genai.types import GenerateContentConfig

    if max_tokens is None:
        max_tokens = MAX_TOKENS

    SYSTEM_INSTRUCTION = "You are a precise SQL generator. Follow the instructions in the prompt exactly and output only what is requested."

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
    return response.text.strip()


def call_model(prompt: str, temperature: float = 0.2, max_tokens: int = None) -> str:
    """Dispatch to the configured LLM backend."""
    if API_TYPE == "sdk":
        return call_llm_sdk(prompt, temperature, max_tokens)
    else:
        return call_llm_openapi(prompt, temperature, max_tokens)


def generate_with_backoff(prompt: str, max_retries: int = 3, initial_delay: float = 2.0, temperature: float = 0.2) -> str:
    """Generate with exponential backoff for rate limits."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return call_model(prompt, temperature=temperature, max_tokens=STEP4_MAX_TOKENS)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower():
                if attempt == max_retries:
                    raise e
                sleep_time = delay + random.uniform(0, 1)
                time.sleep(sleep_time)
                delay *= 2
            else:
                if attempt == max_retries:
                    raise e
                time.sleep(1)
    return ""

# ===================== TABLE HELPERS =====================

def guess_sqlite_type(dtype: Any) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    return "TEXT"

def build_schema_text(df: pd.DataFrame, table_name: str) -> str:
    cols = []
    for c in df.columns:
        t = guess_sqlite_type(df[c].dtype)
        cols.append(f'  "{c}" {t}')
    return f'CREATE TABLE "{table_name}" (\n' + ",\n".join(cols) + "\n);"


def build_column_description(df: pd.DataFrame) -> str:
    """Build a column description string listing name, type, and sample values."""
    parts = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            type_str = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            type_str = "REAL"
        else:
            type_str = "TEXT"
        samples = df[col].dropna().unique()[:3]
        sample_str = ", ".join(str(v) for v in samples)
        parts.append(f"- {col} ({type_str}): e.g. {sample_str}")
    return "\n".join(parts)


def build_full_table_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Build full table markdown for Phase 1 analysis."""
    if len(df) > max_rows:
        return df.head(max_rows).to_markdown(index=False) + f"\n\n... ({len(df) - max_rows} more rows)"
    return df.to_markdown(index=False)


def build_sample_markdown(df: pd.DataFrame, n_rows: int = 10) -> str:
    return df.head(n_rows).to_markdown(index=False)


def sanitize_table_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "t"


def df_to_sqlite(df: pd.DataFrame, table_name: str) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    df_copy = df.copy()
    df_copy.columns = [str(c) for c in df_copy.columns]
    
    # Deduplicate column names
    seen_lower = {}
    new_columns = []
    for col in df_copy.columns:
        col_lower = col.lower()
        if col_lower in seen_lower:
            seen_lower[col_lower] += 1
            new_columns.append(f"{col}_{seen_lower[col_lower]}")
        else:
            seen_lower[col_lower] = 0
            new_columns.append(col)
    
    df_copy.columns = new_columns
    df_copy.to_sql(table_name, con, index=False, if_exists="replace")
    return con

# ===================== CoT SQL REASONING =====================

def analyze_and_generate_sql(table_name: str, df: pd.DataFrame, question: str) -> Dict:
    """Single-pass CoT: reason over the table then generate SQL."""
    column_description = build_column_description(df)
    table_data = build_full_table_markdown(df, max_rows=50)
    prompt = PROMPT.format(
        table_name=table_name,
        column_description=column_description,
        table_data=table_data,
        question=question
    )
    try:
        raw_output = generate_with_backoff(prompt, temperature=0.1)
        result = {"raw_output": raw_output, "reasoning": "", "sql": ""}

        # Extract reasoning
        m = re.search(r"<reasoning>(.*?)</reasoning>", raw_output, re.DOTALL | re.IGNORECASE)
        if m:
            result["reasoning"] = m.group(1).strip()

        # Extract SQL from fenced code block first, then bare SELECT
        sql_match = re.search(r"```(?:sql)?\s*(SELECT.+?)```", raw_output, re.DOTALL | re.IGNORECASE)
        if sql_match:
            result["sql"] = sql_match.group(1).strip().rstrip(";")
        else:
            sel = re.search(r"(?i)(SELECT\s+.+?FROM\s+.+?)(?:;|\n\n|$)", raw_output, re.DOTALL)
            if sel:
                result["sql"] = sel.group(1).strip().rstrip(";")

        return result
    except Exception as e:
        return {"error": str(e), "raw_output": "", "reasoning": "", "sql": "SELECT 1"}


def execute_sql(df: pd.DataFrame, table_name: str, sql: str) -> Tuple[str, Optional[str]]:
    """Execute SQL and return (result, error)."""
    if not sql or sql == "SELECT 1":
        return "", "No valid SQL"
    
    try:
        con = df_to_sqlite(df, table_name)
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        con.close()
        
        if not rows:
            return "EMPTY", None
        
        if len(rows) == 1 and len(rows[0]) == 1:
            val = rows[0][0]
            if val is None:
                return "NULL", None
            return str(val), None
        elif len(rows) == 1:
            return " | ".join(str(v) if v is not None else "NULL" for v in rows[0]), None
        else:
            results = []
            for row in rows[:10]:
                if len(row) == 1:
                    results.append(str(row[0]) if row[0] is not None else "NULL")
                else:
                    results.append(" | ".join(str(v) if v is not None else "NULL" for v in row))
            return ", ".join(results), None
            
    except Exception as e:
        return "", str(e)


def answer_question(question: str, transformed_csv_path: str, original_csv_path: str = None) -> Dict:
    """Main entry point: CoT reasoning + SQL generation + execution."""
    result = {
        "question": question,
        "answer": "",
        "sql": "",
        "sql_result": "",
        "reasoning": "",
        "error": None,
    }

    csv_path = transformed_csv_path if os.path.exists(transformed_csv_path) else original_csv_path
    if not csv_path or not os.path.exists(csv_path):
        result["error"] = "No CSV file found"
        return result

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result["error"] = f"Failed to load CSV: {e}"
        return result

    table_name = sanitize_table_name(Path(csv_path).stem)

    cot = analyze_and_generate_sql(table_name, df, question)
    result["reasoning"] = cot.get("reasoning", "")
    result["sql"]       = cot.get("sql", "")

    if cot.get("error"):
        result["error"] = cot["error"]
        return result

    sql_result, sql_error = execute_sql(df, table_name, cot.get("sql", ""))
    result["sql_result"] = sql_result
    if sql_error:
        result["error"] = f"SQL error: {sql_error}"
    result["answer"] = sql_result
    return result


# ===================== CLI INTERFACE =====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 4: CoT SQL Q&A")
    parser.add_argument("--csv", type=str, required=True, help="Path to transformed CSV")
    parser.add_argument("--question", type=str, required=True, help="Question to answer")
    parser.add_argument("--original-csv", type=str, help="Path to original CSV (fallback)")
    args = parser.parse_args()
    
    result = answer_question(
        question=args.question,
        transformed_csv_path=args.csv,
        original_csv_path=args.original_csv
    )
    
    print("\n" + "="*60)
    print("QUESTION:", args.question)
    print("="*60)
    print("\n📊 PHASE 1 ANALYSIS:")
    print(result.get("phase1_analysis", "N/A")[:500])
    print("\n🎯 PHASE 1 EXPECTED ANSWER:", result.get("phase1_expected_answer", "N/A"))
    print("\n💾 SQL QUERY:")
    print(result.get("sql", "N/A"))
    print("\n📋 SQL RESULT:", result.get("sql_result", "N/A"))
    print("\n✅ FINAL ANSWER:", result.get("answer", "N/A"))
    if result.get("error"):
        print("\n❌ ERROR:", result["error"])
    print("="*60)