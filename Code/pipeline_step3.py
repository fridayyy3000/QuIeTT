#!/usr/bin/env python3
import json
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import ops_runtime
import re
import time
import random
import inspect
import subprocess


import requests

# Import plan validator
try:
    from validate_plan import validate_plan, get_expected_columns, PlanValidationResult
    VALIDATOR_AVAILABLE = True
except Exception as e:
    VALIDATOR_AVAILABLE = False
    logging.warning(f"validate_plan module not available: {e}. Skipping plan validation.")

# --- Configuration ---
PIPELINE_ROOT = Path("testing/data")
PLAN_FILENAME = "plan.json"
STEP_PIPELINE_FILENAME = "step_pipeline.py"

PROJECT_ID = os.environ.get("VERTEX_PROJECT", "YOUR_GCP_PROJECT_ID")
LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
MODEL_ID = os.environ.get("VERTEX_MODEL_ID", "gemini-2.5-flash")
API_TYPE = os.environ.get("VERTEX_API_TYPE", "vertex_sdk")
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "0"))
STEP3_MAX_TOKENS = int(os.environ.get("STEP3_MAX_TOKENS", "1024"))  # Code generation (paper §3.3)
MAX_TOKENS = int(os.environ.get("VERTEX_MAX_TOKENS", "65536"))      # Fallback

# Import google-genai SDK for Gemini with thinking config
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

# Initialize genai client
_genai_client = None

def _get_genai_client():
    """Get or create genai client instance."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )
    return _genai_client

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

def call_gemini_sdk(prompt: str, temperature: float = 0.2, max_tokens: int = None) -> str:
    """Call Gemini model via google-genai SDK with thinking disabled."""
    if max_tokens is None:
        max_tokens = MAX_TOKENS
    
    # System instruction (paper §3.3)
    CONCISE_SYSTEM_INSTRUCTION = """You are a precise Python code generator. Output ONLY the requested Python code with no explanatory text before or after."""

    client = _get_genai_client()
    # Configure thinking budget (set THINKING_BUDGET=0 to match paper results)
    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=ThinkingConfig(thinking_budget=THINKING_BUDGET),
        system_instruction=CONCISE_SYSTEM_INSTRUCTION,
    )
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config,
    )
    return response.text.strip() if response.text else ""

def call_gpt_oss(prompt: str, temperature: float = 0.2, max_tokens: int = None) -> str:
    """Call GPT-OSS-120B via Vertex AI Global OpenAPI."""
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
    
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise Exception(f"No response from {MODEL_ID}")
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    
    if not content:
        raise Exception(f"Empty response from GPT-OSS-120B")
    
    return content

def call_model(prompt: str, temperature: float = 0.2, max_tokens: int = None) -> str:
    """Call appropriate model based on API_TYPE."""
    if API_TYPE == "vertex_sdk":
        return call_gemini_sdk(prompt, temperature, max_tokens)
    else:
        return call_gpt_oss(prompt, temperature, max_tokens)


def generate_with_backoff(prompt, max_retries=3, initial_delay=2.0):
    """
    Wraps model API call with exponential backoff for rate limit errors.
    """
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return call_model(prompt, temperature=0.2, max_tokens=STEP3_MAX_TOKENS)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower():
                if attempt == max_retries:
                    logging.error(f"Max retries reached: {e}")
                    raise e
                sleep_time = delay + random.uniform(0, 1)
                logging.warning(f"Rate limited. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                delay *= 2
            else:
                raise e
    
    return ""


def get_series(df, col_name):
    """Get a series by name, with case-insensitive fallback."""
    if col_name in df.columns:
        return df[col_name]
    
    col_lower = col_name.lower()
    for col in df.columns:
        if col.lower() == col_lower:
            return df[col]
    
    logging.warning(f"Column '{col_name}' not found. Returning NA.")
    return pd.Series(pd.NA, index=df.index, dtype=object)


def deduplicate_columns(df):
    """Ensure unique column names (case-insensitive)."""
    cols = list(df.columns)
    counts = {}
    new_cols = []
    
    for col in cols:
        col_lower = col.lower()
        if col_lower in counts:
            counts[col_lower] += 1
            new_cols.append(f"{col}_{counts[col_lower]}")
        else:
            counts[col_lower] = 0
            new_cols.append(col)
            
    df.columns = new_cols
    return df

def _filter_params_for_function(func, params):
    """
    Filter params dict to only include parameters that func actually accepts.
    """
    try:
        sig = inspect.signature(func)
        func_params = sig.parameters
        
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in func_params.values()):
            return params
        
        accepted_params = {k: v for k, v in params.items() if k in func_params}
        
        filtered_out = set(params.keys()) - set(accepted_params.keys())
        if filtered_out:
            logging.debug(f"Filtered out unsupported params for {func.__name__}: {filtered_out}")
        
        return accepted_params
    except Exception as e:
        logging.warning(f"Could not inspect function signature: {e}. Passing all params.")
        return params


def _fix_column_types(df, writes):
    """
    Fix common type issues in write columns after transformation.
    """
    for col in writes:
        if not col or col not in df.columns:
            continue
        
        if "day_of_week" in col.lower():
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                logging.warning(f"FIXED: Column '{col}' has datetime type but should be string. Converting.")
                df[col] = df[col].dt.day_name()
            df[col] = df[col].astype(str)
        
        elif any(x in col.lower() for x in ["_year", "_month"]):
            if not pd.api.types.is_numeric_dtype(df[col]):
                logging.info(f"FIXED: Converting '{col}' to numeric type")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        elif "_day" in col.lower() and "day_of_week" not in col.lower():
            if not pd.api.types.is_numeric_dtype(df[col]):
                logging.info(f"FIXED: Converting '{col}' to numeric type")
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def validate_step_output(df_before, step, df_after):
    """
    Validate that step execution created expected columns with valid data.
    Returns (is_valid, issues_list)
    """
    issues = []
    writes = step.get('writes', [])
    step_id = step.get('step_id', 'unknown')
    
    for col in writes:
        if not col:
            continue
            
        if col not in df_after.columns:
            issues.append(f"❌ Column '{col}' missing from output")
            continue
        
        if len(df_after) > 0 and df_after[col].isna().all():
            issues.append(f"⚠️  Column '{col}' is all NaN/empty")
        
        if df_after[col].dtype == 'object':
            error_values = df_after[col].astype(str).str.contains('ERROR|error|Error', na=False)
            if error_values.any():
                issues.append(f"⚠️  Column '{col}' contains error messages")
    
    if issues:
        logging.warning(f"[Validation] Step '{step_id}' has issues:")
        for issue in issues:
            logging.warning(f"  {issue}")
        return False, issues
    
    logging.debug(f"[Validation] Step '{step_id}' passed all checks")
    return True, []


# --- The Hybrid Engine ---

def _condition_to_pandas_expr(cond_str: str) -> str:
    """Convert a plan condition string (e.g. 'col contains \\'val\\'') to a
    pandas boolean expression string suitable for df.loc[mask, out] = value."""
    c = cond_str.strip()
    # "col contains 'value'"
    m = re.match(r"^(.+?)\s+contains\s+'(.+?)'$", c, re.IGNORECASE)
    if m:
        col, val = m.group(1).strip(), m.group(2)
        return f"df['{col}'].astype(str).str.contains({repr(val)}, na=False)"
    # "col startswith 'value'"
    m = re.match(r"^(.+?)\s+startswith\s+'(.+?)'$", c, re.IGNORECASE)
    if m:
        col, val = m.group(1).strip(), m.group(2)
        return f"df['{col}'].astype(str).str.startswith({repr(val)})"
    # "col == 'val'" / "col != 'val'" / "col <= 3" etc.
    m = re.match(r"^(.+?)\s*(==|!=|<=|>=|<|>)\s*(.+)$", c)
    if m:
        col, op_sym, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
        if rhs.startswith(("'", '"')):
            return f"df['{col}'] {op_sym} {rhs}"
        try:
            float(rhs)
            return f"pd.to_numeric(df['{col}'], errors='coerce') {op_sym} {rhs}"
        except ValueError:
            return f"df['{col}'] {op_sym} {repr(rhs)}"
    # Fallback — emit a TODO comment so reviewers notice it
    return f"pd.Series([False]*len(df))  # TODO: translate condition: {c!r}"


def generate_code_for_op(op, params, step_id, description=""):
    """Generate readable Python code for a deterministic operation."""
    code_lines = []
    code_lines.append(f"    # === Step {step_id}: {op} ===")
    if description:
        code_lines.append(f"    # {description}")
    
    if op == "add_row_id":
        out = params.get("out", "_row_id")
        code_lines.append(f"    df['{out}'] = range(1, len(df) + 1)")
    
    elif op == "rename":
        mapping = params.get("mapping", params.get("col_map", {}))
        code_lines.append(f"    df = df.rename(columns={mapping})")
    
    elif op == "keep_raw_snapshot":
        col = params.get("col")
        out = params.get("out")
        if col and out:
            code_lines.append(f"    df['{out}'] = df['{col}'].copy()")
        else:
            cols = params.get("cols", [])
            outs = params.get("outs", [])
            if cols and outs:
                code_lines.append(f"    for orig, snap in zip({cols}, {outs}):")
                code_lines.append(f"        df[snap] = df[orig].copy()")
    
    elif op == "extract_regex":
        col = params.get("col", params.get("column"))
        pattern = params.get("pattern", params.get("regex"))
        out_groups = params.get("out_groups", params.get("groups", []))
        code_lines.append(f"    # Extract regex groups from '{col}'")
        code_lines.append(f"    pattern = r\"{pattern}\"")
        code_lines.append(f"    extracted = df['{col}'].astype(str).str.extract(pattern, expand=True)")
        for i, g in enumerate(out_groups):
            code_lines.append(f"    df['{g}'] = extracted[{i}] if {i} < len(extracted.columns) else None")
    
    elif op == "parse_number":
        col = params.get("col")
        out_value = params.get("out_value", "value")
        code_lines.append(f"    # Parse numeric value from '{col}'")
        code_lines.append(f"    df['{out_value}'] = pd.to_numeric(df['{col}'].astype(str).str.replace(',', ''), errors='coerce')")
    
    elif op == "parse_date_text":
        col = params.get("col")
        out_date = params.get("out_date", params.get("out_iso", params.get("out")))
        code_lines.append(f"    # Parse date from '{col}'")
        code_lines.append(f"    df['{out_date}'] = pd.to_datetime(df['{col}'], errors='coerce')")
    
    elif op == "derive_conditional":
        out = params.get("out")
        conditions = params.get("conditions", params.get("cases", []))
        default = params.get("default")
        code_lines.append(f"    # Conditional derivation for '{out}'")
        code_lines.append(f"    df['{out}'] = {repr(default)}")
        for cond in conditions:
            c = cond.get("condition", cond.get("if", ""))
            v = cond.get("value", cond.get("then"))
            if not c:
                continue
            pandas_expr = _condition_to_pandas_expr(c)
            code_lines.append(f"    df.loc[{pandas_expr}, '{out}'] = {repr(v)}")
    
    elif op == "derive_math":
        out = params.get("out")
        expr = params.get("expr")
        code_lines.append(f"    # Math derivation: {out} = {expr}")
        code_lines.append(f"    df['{out}'] = pd.eval(\"{expr}\", local_dict={{c: df[c] for c in df.columns}})")
    
    elif op == "cast_column":
        col = params.get("col")
        dtype = params.get("dtype")
        out_col = params.get("out", col)
        code_lines.append(f"    df['{out_col}'] = df['{col}'].astype('{dtype}')")
    
    elif op == "trim_whitespace":
        cols = params.get("cols", [])
        code_lines.append(f"    for col in {cols}:")
        code_lines.append(f"        if col in df.columns:")
        code_lines.append(f"            df[col] = df[col].astype(str).str.strip()")
    
    elif op == "fillna_static":
        col = params.get("col")
        value = params.get("value")
        code_lines.append(f"    df['{col}'] = df['{col}'].fillna({repr(value)})")
    
    elif op == "replace_string":
        col = params.get("col")
        pattern = params.get("pattern")
        replacement = params.get("replacement", "")
        out_col = params.get("out") or (writes[0] if writes else col)
        code_lines.append(f"    df['{out_col}'] = df['{col}'].astype(str).str.replace(r\"{pattern}\", \"{replacement}\", regex=True)")
    
    elif op == "combine_columns":
        cols = params.get("cols", [])
        out = params.get("out")
        sep = params.get("sep", " ")
        code_lines.append(f"    df['{out}'] = df[{cols}].astype(str).agg('{sep}'.join, axis=1)")
    
    else:
        code_lines.append(f"    # {op}: ops_runtime.{op}(df, **{params})")
    
    code_lines.append("")
    return "\n".join(code_lines)


def execute_step(df, step, generated_codes: list = None):
    """
    Executes a single step using deterministic logic or LLM fallback.
    
    Args:
        df: Input DataFrame
        step: Step dictionary from plan
        generated_codes: List to collect generated code snippets (mutated in place)
    
    Returns:
        Transformed DataFrame
    """
    op = step.get("op")
    params = step.get("params", {})
    writes = step.get("writes", [])
    step_id = step.get("step_id", "unknown")
    description = step.get("description", "")
    
    # Allow select/project — the plan's final select enforces the smart lossless column policy
    # (drops fully-captured originals, keeps partially-captured ones)
    if op == "project":
        logging.warning(f"[Step {step_id}] SKIPPING 'project' operation")
        return df
    
    # SAFEGUARD: Skip raw_snapshot since we're keeping all original columns anyway
    if op == "keep_raw_snapshot":
        logging.warning(f"[Step {step_id}] SKIPPING 'keep_raw_snapshot' - original columns already preserved")
        return df
    
    logging.info(f"Executing Step {step_id}: {op}")

    # 1. Try Deterministic Execution First
    if op != "custom":
        try:
            result_df = None
            
            if op == "add_row_id":
                # Use writes field if out not in params (Gemini puts output in writes, not params)
                out_col = params.get("out") or (writes[0] if writes else "_row_id")
                result_df = ops_runtime.add_row_id(df, out=out_col)

            elif op == "clean_column_names":
                result_df = ops_runtime.clean_column_names(df)

            elif op == "parse_date":
                result_df = ops_runtime.parse_date_text(
                    df, 
                    col=params["col"], 
                    formats=params.get("formats", []),
                    out_date=params.get("out_iso"),
                    out_parts=params.get("out_parts"),
                    part_names=params.get("part_names")
                )

            elif op == "rename":
                col_map = params.get("mapping", params.get("col_map", {}))
                result_df = df.copy()
                for old, new in col_map.items():
                    series = get_series(result_df, old)
                    if pd.isna(series).all(): continue
                    actual_old = series.name
                    
                    if new in result_df.columns and new != actual_old:
                        result_df = result_df.drop(columns=[new])
                    
                    result_df = result_df.rename(columns={actual_old: new})

            elif op == "filter":
                result_df = ops_runtime.filter_rows(df, include=params.get("include"), exclude=params.get("exclude"))

            elif op == "project":
                result_df = ops_runtime.select_columns(df, cols=params.get("cols"))

            elif op == "select":
                # Final column selection — enforce smart lossless output
                cols = params.get("cols", [])
                if cols:
                    # Only keep columns that actually exist
                    valid_cols = [c for c in cols if c in df.columns]
                    missing = [c for c in cols if c not in df.columns]
                    if missing:
                        logging.warning(f"[select] Columns not found, skipping: {missing}")
                    result_df = df[valid_cols].copy() if valid_cols else df
                else:
                    result_df = df

            elif op == "pivot_longer":
                result_df = ops_runtime.pivot_longer(
                    df, 
                    index=params.get("index"), 
                    names_to=params.get("names_to"), 
                    values_to=params.get("values_to")
                )
            
            elif op == "extract_regex":
                # Handle extract_regex with proper params
                col = params.get("col")
                pattern = params.get("pattern")
                out_groups = params.get("out_groups", writes)
                if col and pattern:
                    result_df = ops_runtime.extract_regex(df, col=col, pattern=pattern, out_groups=out_groups)
            
            elif op == "parse_number":
                # Handle parse_number with proper params
                col = params.get("col")
                out_value = params.get("out_value", writes[0] if writes else None)
                out_unit = params.get("out_unit")
                if col and out_value:
                    result_df = ops_runtime.parse_number(df, col=col, out_value=out_value, out_unit=out_unit)
            
            elif op == "parse_date_text":
                # Handle parse_date_text with writes field for output column names
                col = params.get("col")
                out_date = params.get("out_date") or params.get("out") or params.get("out_col")
                out_parts = params.get("out_parts")
                formats = params.get("formats")
                
                # If out_date not specified, try to get from writes
                if not out_date and writes:
                    for w in writes:
                        if 'date' in w.lower() or 'parsed' in w.lower():
                            out_date = w
                            break
                    if not out_date:
                        out_date = writes[0]
                
                result_df = ops_runtime.parse_date_text(
                    df,
                    col=col,
                    out_date=out_date,
                    out_parts=out_parts,
                    formats=formats,
                    _expected_writes=writes
                )
            
            elif hasattr(ops_runtime, op):
                func = getattr(ops_runtime, op)
                
                if 'out' not in params and writes and len(writes) == 1:
                    # For ops that use 'col' (singular)
                    col = params.get('col')
                    # For ops that use 'cols' (plural, e.g. trim_whitespace) — check first element
                    if not col:
                        cols_list = params.get('cols', [])
                        col = cols_list[0] if cols_list else None
                    if col and writes[0] != col:
                        params['out'] = writes[0]
                        logging.debug(f"[{op}] Inferred out='{writes[0]}' from writes")
                
                if op == 'parse_date_text' and writes:
                    params['_expected_writes'] = writes
                
                filtered_params = _filter_params_for_function(func, params)
                logging.debug(f"[{op}] Calling with params: {filtered_params}")
                result_df = func(df, **filtered_params)
            
            # If deterministic execution succeeded, record the code and return
            if result_df is not None:
                if generated_codes is not None:
                    code_snippet = generate_code_for_op(op, params, step_id, description)
                    generated_codes.append({
                        'step_id': step_id,
                        'op': op,
                        'code': code_snippet,
                        'success': True,
                        'deterministic': True
                    })
                return result_df
            
        except Exception as e:
            logging.warning(f"Deterministic execution for '{op}' failed: {e}. Falling back to LLM.")

    # 2. LLM Execution (Custom or Fallback)
    logging.info(f"Op '{op}' requires LLM generation (Custom or Fallback).")
    
    # Generate sample data for LLM context
    sample_data = ""
    try:
        if len(df) > 0:
            reads = step.get('reads', [])
            relevant_cols = [c for c in reads if c in df.columns]
            if not relevant_cols:
                relevant_cols = list(df.columns[:3])
            
            sample_df = df[relevant_cols].head(3)
            sample_data = f"\n**Sample Data (first 3 rows)**:\n{sample_df.to_string()}\n"
    except:
        pass
    
    prompt = f"""
You are a Python Data Engineer. Write a snippet to transform a DataFrame `df`.

**CONTEXT**:
- Current columns: {list(df.columns)}
- DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns{sample_data}

**TASK**: {step.get('description')}

**OPERATION**: {op}
**Parameters**: {json.dumps(params, indent=2)}
**Input Columns (reads)**: {step.get('reads', [])}
**Output Columns (writes)**: {writes}

**CRITICAL REQUIREMENTS**:
1. ALL output columns {writes} MUST be created
2. If extraction/parsing fails, use the ORIGINAL column value as fallback
3. Do NOT create columns with all NaN values
4. Handle missing data gracefully (use .fillna(), coalesce, etc.)
5. Do NOT drop rows unless explicitly required by the operation
6. Test your regex/extraction patterns against the sample data above

**EXAMPLE PATTERNS**:
```python
# For extraction with fallback:
df['new_col'] = df['source_col'].str.extract(r'pattern')[0]
df['new_col'] = df['new_col'].fillna(df['source_col'])  # Fallback!

# For parsing with error handling:
df['parsed'] = pd.to_datetime(df['date_col'], errors='coerce')

# For transformations:
df['clean'] = df['raw'].str.strip().str.lower()
```

Return ONLY valid Python code. Assume `df`, `pd`, `np`, `re` exist.
"""
    
    # Retry loop
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            code = generate_with_backoff(prompt)
            code = code.replace("```python", "").replace("```", "").strip()
            
            # Execute the snippet
            local_scope = {"df": df.copy(), "pd": pd, "np": np, "re": re, "ops_runtime": ops_runtime}
            exec(code, {}, local_scope)
            new_df = local_scope["df"]
            
            failed_reasons = []

            # Schema Existence
            missing_cols = [c for c in writes if c not in new_df.columns]
            if missing_cols:
                failed_reasons.append(f"Columns {missing_cols} are missing from the output.")
            
            # Type validation
            for col in writes:
                if col and col in new_df.columns:
                    if "day_of_week" in col.lower() and pd.api.types.is_datetime64_any_dtype(new_df[col]):
                        failed_reasons.append(f"Column '{col}' has datetime type but should be string")
                    elif any(x in col.lower() for x in ["_year", "_month", "_day"]) and "day_of_week" not in col.lower():
                        if not pd.api.types.is_numeric_dtype(new_df[col]) and not new_df[col].isna().all():
                            failed_reasons.append(f"Column '{col}' should be numeric but is {new_df[col].dtype}")

            # Row Count
            # Ops that legitimately change row count are excluded from this check.
            # split_multi_value and explode_entities increase rows intentionally.
            _row_change_ok_ops = {
                "filter", "filter_rows",
                "pivot_longer", "pivot_wider",
                "aggregate", "summarize",
                "split_multi_value", "explode_entities",
                "deduplicate_rows",
            }
            if op not in _row_change_ok_ops:
                if len(new_df) != len(df):
                    failed_reasons.append(f"Row count changed from {len(df)} to {len(new_df)} but op is '{op}'.")

            # NaN check
            failed_cols = []
            for col in writes:
                if col in new_df.columns and new_df[col].isna().all() and not df.empty:
                    failed_cols.append(col)
            if failed_cols:
                failed_reasons.append(f"Columns {failed_cols} are all NaN.")
            
            if failed_reasons and attempt < max_retries:
                logging.warning(f"Attempt {attempt+1} failed: {'; '.join(failed_reasons)}. Retrying...")
                prompt += f"\n\nERROR: The code failed validation: {'; '.join(failed_reasons)}. Please fix the logic."
                continue
            
            # FALLBACK if all retries exhausted with issues
            if failed_reasons and attempt == max_retries:
                logging.error(f"All retries exhausted. Final issues: {'; '.join(failed_reasons)}")
                logging.warning("FALLBACK STRATEGY: Preserving original columns when transformation fails")
                
                safe_df = df.copy()
                reads = step.get('reads', [])
                
                for w in writes:
                    if w and w not in safe_df.columns:
                        fallback_used = False
                        
                        if len(reads) == 1 and reads[0] in df.columns:
                            logging.warning(f"  → Keeping original column '{reads[0]}' as fallback for '{w}'")
                            safe_df[w] = df[reads[0]].copy()
                            fallback_used = True
                        
                        elif len(reads) > 1:
                            for read_col in reads:
                                if read_col in df.columns and (read_col.lower() in w.lower() or w.lower() in read_col.lower()):
                                    logging.warning(f"  → Keeping original column '{read_col}' as fallback for '{w}'")
                                    safe_df[w] = df[read_col].copy()
                                    fallback_used = True
                                    break
                            
                            if not fallback_used:
                                for read_col in reads:
                                    if read_col in df.columns:
                                        logging.warning(f"  → Using '{read_col}' as fallback for '{w}'")
                                        safe_df[w] = df[read_col].copy()
                                        fallback_used = True
                                        break
                        
                        if not fallback_used:
                            logging.warning(f"  → No fallback available for '{w}'. Creating empty column.")
                            safe_df[w] = pd.NA
                
                # Save the fallback code
                if generated_codes is not None:
                    fallback_code = f"# Step {step_id}: {op} (FALLBACK - LLM code failed validation)\n# Original generated code:\n'''\n{code}\n'''\n# Fallback: preserved original columns"
                    generated_codes.append({
                        'step_id': step_id,
                        'op': op,
                        'code': fallback_code,
                        'success': False
                    })
                
                return safe_df
            
            # Fix column types
            new_df = _fix_column_types(new_df, writes)
            
            # *** KEY CHANGE: Save the generated code ***
            if generated_codes is not None:
                generated_codes.append({
                    'step_id': step_id,
                    'op': op,
                    'code': code,
                    'success': True
                })
            
            return new_df
            
        except Exception as e:
            logging.error(f"Attempt {attempt+1} error: {e}")
            if attempt < max_retries:
                prompt += f"\n\nERROR: The code raised an exception: {e}. Please fix it."
            else:
                logging.error("All retries failed. Returning original DataFrame.")
                if generated_codes is not None:
                    generated_codes.append({
                        'step_id': step_id,
                        'op': op,
                        'code': f"# Step {step_id}: {op} (FAILED - Exception: {e})",
                        'success': False
                    })
                return df
    
    return df

class StrictModeError(Exception):
    """Raised when strict mode validation fails."""
    pass


def make_sql_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-processing pass that makes the final DataFrame safe for SQL ingestion:
    1. Sanitize column names  → snake_case, only [a-zA-Z0-9_]
    2. Cast pure-numeric object columns → int64 / float64
    3. Drop columns that are 100% NULL (failed derivations)
    """
    import re

    # ── 1. Sanitize column names ──────────────────────────────────────────────
    rename_map = {}
    seen = {}
    for col in df.columns:
        new = col
        # Replace any run of non-alphanumeric (except underscore) with a single _
        new = re.sub(r'[^a-zA-Z0-9]+', '_', new)
        new = new.strip('_')
        # If starts with a digit, prefix with col_
        if new and new[0].isdigit():
            new = 'col_' + new
        if not new:
            new = 'col'
        # Resolve collision (two original names → same sanitised name)
        base = new
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            new = f"{base}_{seen[base]}"
        if new != col:
            rename_map[col] = new
            logging.info(f"[SQL-READY] Renamed column '{col}' → '{new}'")
    if rename_map:
        df = df.rename(columns=rename_map)

    # ── 2. Drop 100%-NULL columns ─────────────────────────────────────────────
    all_null = [c for c in df.columns if df[c].isna().all()]
    if all_null:
        df = df.drop(columns=all_null)
        logging.info(f"[SQL-READY] Dropped 100%% NULL columns: {all_null}")

    # ── 3. Cast pure-numeric object columns ──────────────────────────────────
    for col in df.columns:
        if df[col].dtype != object:
            continue
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        numeric_series = pd.to_numeric(non_null, errors='coerce')
        if numeric_series.notna().all():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Downcast to int when there are no fractional parts
            if df[col].dropna().apply(lambda x: x == int(x)).all():
                df[col] = df[col].astype('Int64')  # nullable integer
            logging.info(f"[SQL-READY] Cast column '{col}' from object → numeric")

    # ── 4. Reassign row_id if duplicated (explode ran before add_row_id) ─────
    id_col = 'row_id' if 'row_id' in df.columns else ('_row_id' if '_row_id' in df.columns else None)
    if id_col and df[id_col].duplicated().any():
        df[id_col] = range(len(df))
        logging.info(f"[SQL-READY] Reassigned '{id_col}' sequentially (duplicates after row-count-changing op)")

    return df


def generate_step_pipeline_with_csv(run_dir: Path, csv_path: Path, plan_filename: str = PLAN_FILENAME, output_filename: str = "final.csv", strict_mode: bool = True):
    """
    Executes the pipeline directly using the Hybrid Engine.
    
    KEY IMPROVEMENT: Saves actual generated code to step_pipeline.py
    """
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Plan
    plan_path = run_dir / plan_filename
    if not plan_path.exists():
        logging.error(f"Plan not found: {plan_path}")
        return

    with open(plan_path) as f:
        plan = json.load(f)
    
    # Runtime safety net: if the plan has no select step, auto-inject one from
    # final_output.columns keeping only canonical + derived roles.  This mirrors
    # what sanitize_plan() does in test.py and ensures clean output even when
    # running step_3.py standalone against an older plan.json.
    _has_select = any(s.get("op") == "select" for s in plan.get("steps", []))
    if not _has_select:
        _fo_cols = plan.get("final_output", {}).get("columns", [])
        _keep_roles = {"canonical", "derived"}
        _select_cols = [
            (c["name"] if isinstance(c, dict) else c)
            for c in _fo_cols
            if not isinstance(c, dict) or c.get("role", "derived") in _keep_roles
        ]
        if "_row_id" in _select_cols:
            _select_cols = ["_row_id"] + [c for c in _select_cols if c != "_row_id"]
        if _select_cols:
            plan["steps"].append({
                "step_id": "auto_final_select",
                "op": "select",
                "description": "Runtime auto-injected select: drop raw_snapshot originals.",
                "reads": [], "writes": _select_cols,
                "params": {"cols": _select_cols},
                "fixes_issues": [], "depends_on": []
            })
            logging.info(f"[Auto-inject] No select in plan — injected select with {len(_select_cols)} cols")
    del _has_select
    if VALIDATOR_AVAILABLE:
        logging.info("[Validation] Checking plan against operation schema...")
        validation_result = validate_plan(plan)
        
        if not validation_result.is_valid:
            logging.error("[Validation] Plan failed schema validation:")
            for err in validation_result.errors:
                logging.error(f"  ❌ {err}")
            
            error_report = {
                "status": "PLAN_INVALID",
                "errors": [str(e) for e in validation_result.errors],
                "warnings": [str(w) for w in validation_result.warnings]
            }
            with open(run_dir / "validation_errors.json", "w") as f:
                json.dump(error_report, f, indent=2)
            
            if strict_mode:
                raise StrictModeError(f"Plan validation failed with {len(validation_result.errors)} errors")
        else:
            logging.info("[Validation] Plan passed schema validation ✓")
            if validation_result.warnings:
                for warn in validation_result.warnings:
                    logging.warning(f"  ⚠️  {warn}")
    
    # 1c. Calculate expected columns
    expected_columns = set()
    for step in plan.get("steps", []):
        expected_columns.update(step.get("writes", []))
    expected_columns = {c for c in expected_columns if c}
    
    # If the plan ends with a select step, the FINAL expected columns are only those
    # retained by the select — intermediate columns intentionally dropped shouldn't
    # count as "missing" and shouldn't trigger STRICT MODE preservation.
    _select_step = next((s for s in reversed(plan.get("steps", [])) if s.get("op") == "select"), None)
    if _select_step:
        _select_cols = set(_select_step.get("params", {}).get("cols", []))
        if _select_cols:
            expected_columns = _select_cols
            logging.info(f"[Plan] Select step found — expected final columns: {sorted(expected_columns)}")
        del _select_cols
    del _select_step
    logging.info(f"[Plan] Expecting {len(expected_columns)} output columns: {sorted(expected_columns)}")
        
    # 2. Load Data
    try:
        df = pd.read_csv(csv_path)
        df = deduplicate_columns(df)
        original_columns = set(df.columns)
    except Exception as e:
        logging.warning(f"Standard CSV read failed: {e}. Trying with error handling...")
        try:
            df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
            df = deduplicate_columns(df)
            original_columns = set(df.columns)
            logging.info(f"CSV loaded with fallback reader: {len(df)} rows")
        except Exception as e2:
            logging.error(f"Failed to read CSV even with fallback: {e2}")
            return
    
    # 3. Execute Steps with Validation
    # *** KEY CHANGE: Collect generated code snippets ***
    generated_codes = []
    validation_failures = []
    step_results = []
    df_original = df.copy()  # **NEW: Keep original unmodified**
    
    for i, step in enumerate(plan["steps"], 1):
        step_id = step.get('step_id', f'step_{i}')
        df_before = df.copy()
        
        try:
            df = execute_step(df, step, generated_codes)
        except Exception as e:
            logging.error(f"[Step {step_id}] Execution failed: {e}")
            logging.warning(f"[Step {step_id}] Keeping DataFrame unchanged, will retry with LLM")
            # **IMPROVEMENT: Don't fail the whole pipeline on one step**
            step_results.append({
                'step_id': step_id,
                'status': 'FAILED',
                'error': str(e)
            })
            # Keep df unchanged, continue to next step
            continue
        
        is_valid, issues = validate_step_output(df_before, step, df)
        
        step_result = {
            'step_id': step_id,
            'op': step.get('op'),
            'expected_writes': step.get('writes', []),
            'actual_new_cols': list(set(df.columns) - set(df_before.columns)),
            'status': 'OK' if is_valid else 'ISSUES',
            'issues': issues if issues else []
        }
        step_results.append(step_result)
        
        if not is_valid:
            validation_failures.append(step_result)
    
    # 3.5 POST-EXECUTION: Fix empty columns with fallback strategies
    original_row_count = len(pd.read_csv(csv_path))
    new_columns = set(df.columns) - original_columns
    
    empty_columns_fixed = 0
    for col in new_columns:
        if df[col].notna().sum() == 0:
            # Try fallback strategies for empty columns
            logging.warning(f"[POST-FIX] Column '{col}' is empty, attempting fallback...")
            
            # Strategy 1: If column name suggests numeric, try to extract from related columns
            if any(x in col.lower() for x in ['numeric', 'value', 'count', 'amount', 'total']):
                # Find source column from plan
                for step in plan.get('steps', []):
                    if col in step.get('writes', []):
                        source_col = step.get('params', {}).get('col')
                        if source_col and source_col in df.columns:
                            # Try generic numeric extraction
                            try:
                                df[col] = df[source_col].astype(str).str.replace(',', '').str.extract(r'(\d+\.?\d*)')[0].astype(float)
                                if df[col].notna().sum() > 0:
                                    logging.info(f"[POST-FIX] Fixed '{col}' with numeric extraction from '{source_col}'")
                                    empty_columns_fixed += 1
                            except:
                                pass
                        break
            
            # Strategy 2: If column name suggests boolean, try simple contains check
            elif any(x in col.lower() for x in ['is_', 'has_', 'flag']):
                for step in plan.get('steps', []):
                    if col in step.get('writes', []):
                        conditions = step.get('params', {}).get('conditions', [])
                        for cond in conditions:
                            cond_str = cond.get('condition', cond.get('if', ''))
                            if 'contains' in cond_str.lower():
                                # Extract the search value
                                match = re.search(r"contains\s+['\"]([^'\"]+)['\"]", cond_str, re.IGNORECASE)
                                if match:
                                    search_val = match.group(1)
                                    # Find source column
                                    for c in df.columns:
                                        if df[c].astype(str).str.contains(search_val, case=False, na=False).any():
                                            df[col] = df[c].astype(str).str.contains(search_val, case=False, na=False)
                                            if df[col].notna().sum() > 0:
                                                logging.info(f"[POST-FIX] Fixed '{col}' with contains check for '{search_val}'")
                                                empty_columns_fixed += 1
                                            break
                        break
    
    if empty_columns_fixed > 0:
        logging.info(f"[POST-FIX] Fixed {empty_columns_fixed} empty columns with fallback strategies")
    
    # 3.6 ROW COUNT VALIDATION
    current_row_count = len(df)
    if current_row_count != original_row_count:
        # Row count changes are intentional for explode/filter ops — log at INFO, not WARNING.
        logging.info(f"[ROW COUNT] Changed from {original_row_count} to {current_row_count} (explode or filter op)")
    
    # 4. Final Validation
    actual_columns = set(df.columns)
    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - original_columns - expected_columns
    
    completion_rate = len(expected_columns - missing_columns) / len(expected_columns) * 100 if expected_columns else 100
    
    logging.info(f"\n{'='*60}")
    logging.info(f"[FINAL VALIDATION]")
    logging.info(f"  Expected columns: {len(expected_columns)}")
    logging.info(f"  Got columns: {len(expected_columns - missing_columns)}")
    logging.info(f"  Missing: {len(missing_columns)}")
    logging.info(f"  Completion: {completion_rate:.1f}%")
    logging.info(f"{'='*60}")
    
    if missing_columns:
        logging.warning(f"  Missing columns: {sorted(missing_columns)}")
    
    # Save execution report
    report = {
        "status": "COMPLETE" if completion_rate == 100 else "INCOMPLETE",
        "completion_rate": completion_rate,
        "expected_columns": sorted(expected_columns),
        "missing_columns": sorted(missing_columns),
        "row_count_original": original_row_count,
        "row_count_final": current_row_count,
        "empty_columns_fixed": empty_columns_fixed,
        "step_results": step_results,
        "validation_failures": len(validation_failures)
    }
    with open(run_dir / "execution_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Strict mode
    if strict_mode and missing_columns and completion_rate < 100:
        logging.error(f"[STRICT MODE] Pipeline incomplete! Missing {len(missing_columns)} columns.")
        # **IMPROVEMENT:** Still preserve original columns in final CSV
        logging.info(f"[PRESERVATION] Adding original columns back: {sorted(original_columns)}")
        for orig_col in original_columns:
            if orig_col not in df.columns:
                # Reload original to get these columns
                df_orig = pd.read_csv(csv_path)
                df_orig = deduplicate_columns(df_orig)
                if orig_col in df_orig.columns:
                    df[orig_col] = df_orig[orig_col]
    
    if validation_failures:
        logging.warning(f"[Pipeline] {len(validation_failures)} step(s) had validation issues")
        for failure in validation_failures:
            logging.warning(f"  - {failure['step_id']}: {len(failure['issues'])} issue(s)")
    else:
        logging.info(f"[Pipeline] All {len(plan['steps'])} steps passed validation ✓")
    
    # **IMPROVEMENT:** Filter to final_output columns if specified, remove intermediate _raw columns
    final_output = plan.get('final_output', {})
    final_columns_spec = final_output.get('columns', [])
    
    # Primary source for the final column ORDER is the last select step in the plan.
    # This is the most explicit and authoritative declaration of the final schema.
    # Fall back to final_output.columns (excluding raw_snapshot) only if no select exists.
    _last_select = next((s for s in reversed(plan.get('steps', [])) if s.get('op') == 'select'), None)
    if _last_select:
        _select_order = _last_select.get('params', {}).get('cols', [])
        if _select_order:
            # Use select order directly; the STRICT MODE fix already trimmed expected_columns
            # to this set, so any column in select that exists in df will be kept.
            final_column_names = list(_select_order)
            non_rowid_cols = [c for c in final_column_names if c not in ('row_id', '_row_id')]
            logging.info(f"[COLUMNS] Using select step column order ({len(final_column_names)} cols)")
        else:
            _last_select = None  # fall through
    
    if not _last_select:
        # Check if final_output is valid (has meaningful columns, not just row_id/#)
        # Exclude raw_snapshot columns — they are source columns that have been fully
        # captured into derived columns and should be dropped from the final table.
        final_column_names = [
            (c['name'] if isinstance(c, dict) else c)
            for c in final_columns_spec
            if not (isinstance(c, dict) and c.get('role') == 'raw_snapshot')
        ]
        non_rowid_cols = [c for c in final_column_names if c not in ('row_id', '_row_id')]
    
    # final_output is valid if:
    # 1. Has at least 3 non-rowid columns, OR
    # 2. Covers at least 50% of original columns
    min_valid_cols = max(3, len(original_columns) // 2)
    final_output_valid = len(non_rowid_cols) >= min_valid_cols
    
    if final_columns_spec and final_output_valid:
        # Use the columns specified in final_output (it's valid)
        # Also always include row_id variants
        if 'row_id' not in final_column_names and '_row_id' not in final_column_names:
            if 'row_id' in df.columns:
                final_column_names.insert(0, 'row_id')
            elif '_row_id' in df.columns:
                final_column_names.insert(0, '_row_id')
        
        # Filter to only columns that exist
        column_order = [c for c in final_column_names if c in df.columns]
        
        # Log what we're dropping
        dropped = set(df.columns) - set(column_order)
        if dropped:
            logging.info(f"[COLUMNS] Dropping intermediate columns: {sorted(dropped)}")
    else:
        # Fallback: final_output is incomplete - keep all columns except _raw
        logging.info(f"[COLUMNS] final_output incomplete ({len(non_rowid_cols)} cols, need {min_valid_cols}), keeping all columns except _raw")
        column_order = list(original_columns) + list(new_columns)
        column_order = [c for c in column_order if c in df.columns and not c.endswith('_raw')]
    
    if column_order:
        df = df[column_order]
        logging.info(f"[COLUMNS] Final: {len(column_order)} columns")
        
    # 5. SQL-readiness post-processing
    df = make_sql_ready(df)

    # 6. Save Result
    output_path = run_dir / output_filename
    df.to_csv(output_path, index=False)
    logging.info(f"Saved result to {output_path}")
    
    # *** KEY CHANGE: Save ACTUAL generated code to step_pipeline.py ***
    if plan_filename == PLAN_FILENAME:
        code_lines = [
            "#!/usr/bin/env python3",
            '"""',
            "Auto-generated transformation pipeline code.",
            "Generated by step_3.py using GPT OSS 120B",
            '"""',
            "import pandas as pd",
            "import numpy as np",
            "import re",
            "",
            "",
            "def transform(df, plan=None):",
            '    """',
            "    Apply all transformation steps to the DataFrame.",
            "    ",
            "    Args:",
            "        df: Input pandas DataFrame",
            "        plan: Optional plan dict (not used, transformations are inline)",
            "    ",
            "    Returns:",
            "        Transformed DataFrame",
            '    """',
            ""
        ]
        
        if not generated_codes:
            code_lines.append("    # No transformations were recorded")
            code_lines.append("    return df")
        else:
            for i, code_info in enumerate(generated_codes):
                step_id = code_info['step_id']
                op = code_info['op']
                code = code_info['code']
                success = code_info['success']
                is_deterministic = code_info.get('deterministic', False)
                
                if is_deterministic:
                    # Deterministic code is already properly formatted with indentation
                    for line in code.split('\n'):
                        code_lines.append(line)
                else:
                    # LLM-generated code needs header and indentation
                    code_lines.append(f"    # === Step {step_id}: {op} (LLM-generated) ===")
                    if success:
                        code_lines.append(f"    # Status: SUCCESS")
                    else:
                        code_lines.append(f"    # Status: FAILED/FALLBACK")
                    
                    # Indent the code properly for the function
                    for line in code.split('\n'):
                        if line.strip():
                            code_lines.append(f"    {line}")
                        else:
                            code_lines.append("")
                code_lines.append("")
            
            code_lines.append("    return df")
        
        code_lines.append("")
        code_lines.append("")
        code_lines.append("if __name__ == '__main__':")
        code_lines.append("    import sys")
        code_lines.append("    if len(sys.argv) < 2:")
        code_lines.append("        print('Usage: python step_pipeline.py <input.csv>')")
        code_lines.append("        sys.exit(1)")
        code_lines.append("    ")
        code_lines.append("    df = pd.read_csv(sys.argv[1])")
        code_lines.append("    result = transform(df)")
        code_lines.append("    print(result.head())")
        code_lines.append("    print(f'Shape: {result.shape}')")
        
        with open(run_dir / STEP_PIPELINE_FILENAME, "w") as f:
            f.write('\n'.join(code_lines))
        
        logging.info(f"Saved generated code to {run_dir / STEP_PIPELINE_FILENAME}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--plan-file", type=str, default=PLAN_FILENAME, help="Name of the plan file (default: plan.json)")
    parser.add_argument("--output-file", type=str, default="final.csv", help="Name of the output CSV file (default: final.csv)")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory override")
    args = parser.parse_args()

    csv_path = args.csv_path
    
    if args.run_dir:
        run_dir = args.run_dir
    else:
        pipeline_root_str = os.environ.get("PIPELINE_ROOT_OVERRIDE", str(PIPELINE_ROOT))
        pipeline_root = Path(pipeline_root_str)
        
        run_dir = pipeline_root / csv_path.parent.name / csv_path.stem
        
    run_dir.mkdir(parents=True, exist_ok=True)
    
    generate_step_pipeline_with_csv(run_dir, csv_path, plan_filename=args.plan_file, output_filename=args.output_file)