#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ops_runtime.py
==============
Implements the universal transformation registry.
Each op is a function: (df, **params) -> df
"""

import pandas as pd
import numpy as np
import re
import logging


# ---------------- CORE OPS ----------------

def add_row_id(df, out: str = "_row_id", **kwargs):
    """Add a stable 1-based integer row id (1 = first row)."""
    df = df.reset_index(drop=True)
    df[out] = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
    return df


def clean_column_names(df):
    """
    Standardize column names to snake_case.
    - Lowercase
    - Replace spaces, newlines, and special chars with underscores
    - Deduplicate
    """
    new_cols = []
    for c in df.columns:
        c = str(c).strip().lower()
        c = re.sub(r'[\s\n]+', '_', c)
        c = re.sub(r'[^a-z0-9_]', '', c)
        new_cols.append(c)
    
    # Deduplicate
    seen = {}
    final_cols = []
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            final_cols.append(c)
            
    df.columns = final_cols
    return df


def keep_raw_snapshot(df, sidecar_cols, **kwargs):
    """
    Snapshot raw columns before destructive transforms.

    Supports three formats:

    1) Dict:  {"source_col": "dest_col", ...}

    2) List of dicts (what the planner emits):
        [{"col": "name", "out": "raw_name"}, ...]

    3) Legacy list of strings:
        ["raw_price", "raw_date"]
        - infers base col ("price", "date") and copies df[base] -> df[raw_*]
    """
    if not sidecar_cols:
        logging.warning("[keep_raw_snapshot] No sidecar_cols specified; skipping.")
        return df

    # Dict format: {"source": "dest", ...}
    if isinstance(sidecar_cols, dict):
        for src_col, dst_col in sidecar_cols.items():
            if src_col not in df.columns:
                logging.warning(f"[keep_raw_snapshot] Source column '{src_col}' not found; skipping.")
                continue
            df[dst_col] = df[src_col].copy()
        return df

    if not isinstance(sidecar_cols, list):
        return df

    # List of dicts: [{"col": "source", "out": "dest"}, ...]
    if sidecar_cols and isinstance(sidecar_cols[0], dict):
        for spec in sidecar_cols:
            src = spec.get("col")
            out = spec.get("out") or (f"raw_{src}" if src else None)
            if not src or not out:
                continue
            if src in df.columns and out not in df.columns:
                df[out] = df[src].copy()
        return df

    # Legacy list of strings: ["raw_price", "raw_date"]
    for col in sidecar_cols:
        base = col.replace("raw_", "")
        raw_name = col if col.startswith("raw_") else f"raw_{base}"
        if base in df.columns and raw_name not in df.columns:
            df[raw_name] = df[base].copy()
    return df


def trim_whitespace(df, cols, out=None, **kwargs):
    """Strip leading/trailing whitespace from string columns.
    
    If a single column is provided in cols and out is set (and differs from that column),
    the trimmed result is written to the out column instead of in-place.
    """
    for c in cols:
        if c not in df.columns:
            logging.warning(f"[trim_whitespace] Column '{c}' not found; skipping.")
            continue
        # Use .str.strip() directly (not .astype(str).str.strip()) so that NaN/pd.NA
        # values are preserved as NaN rather than being converted to the string "nan"
        # or "<NA>", which would fool the all-null column detection in make_sql_ready().
        if df[c].dtype == object:
            trimmed = df[c].str.strip()
        else:
            # Non-string column: convert, strip, then restore nulls
            trimmed = df[c].astype(str).str.strip().where(df[c].notna(), other=pd.NA)
        dest = out if (out and len(cols) == 1 and out != c) else c
        df[dest] = trimmed
    return df


def sanitize_text(df, cols=None, case: str = "lower", col=None, out=None, **kwargs):
    """
    Normalize whitespace and case for text columns.

    Supports two calling styles:

    1) Old style (in-place):
       sanitize_text(df, cols=["city"], case="title")

    2) Planner style for normalized columns:
       sanitize_text(df, col="Position", case="title")
       -> writes to '<slug(col)>_normalized'
          e.g. "Position" -> "position_normalized"
    """
    # Normalize cols argument
    if cols is None:
        if col is None:
            logging.warning("[sanitize_text] Neither 'cols' nor 'col' provided; no-op.")
            return df
        cols = [col]
    elif isinstance(cols, str):
        cols = [cols]

    for c in cols:
        if c not in df.columns:
            logging.warning(f"[sanitize_text] Column '{c}' not found; skipping.")
            continue

        series = df[c].astype(str)

        if case == "lower":
            series = series.str.lower()
        elif case == "upper":
            series = series.str.upper()
        elif case == "title":
            series = series.str.title()

        # collapse internal whitespace
        series = series.str.replace(r"\s+", " ", regex=True)

        # Decide target column:
        # - if explicit out is provided, use that
        # - else, if this came from a single 'col', generate '<slug>_normalized'
        target_col = out
        if target_col is None and col is not None and len(cols) == 1:
            slug = re.sub(r"[^0-9a-zA-Z]+", "_", c).lower().strip("_")
            target_col = f"{slug}_normalized"

        if target_col:
            df[target_col] = series
        else:
            # Old behavior: modify in-place
            df[c] = series

    return df


def split_multi_value(
    df,
    col,
    separators,
    value_out,
    order_out=None,
    count_out=None,
    explode=True,
    original_col_rename=None,
):
    """
    Split a multi-value cell into a list, optionally explode rows.

    This op is used by some plans directly, separate from explode_entities.

    If `original_col_rename` is provided, we first copy the original `col`
    into that new column before splitting (so we preserve the original text).
    """
    if col not in df.columns:
        logging.warning(f"[split_multi_value] Column '{col}' not found; skipping.")
        return df

    # Optional: keep original column under a new name
    if original_col_rename and original_col_rename not in df.columns:
        df[original_col_rename] = df[col]

    pattern = "|".join(map(re.escape, separators)) if separators else r","
    # Split and strip whitespace from each element
    df[value_out] = df[col].astype(str).str.split(pattern).apply(
        lambda x: [s.strip() for s in x] if isinstance(x, list) else x
    )

    if count_out:
        df[count_out] = df[value_out].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
    if order_out:
        df[order_out] = df[value_out].apply(
            lambda x: list(range(1, len(x) + 1)) if isinstance(x, list) else []
        )

    if explode:
        df = df.explode(value_out).reset_index(drop=True)

    return df



def combine_columns(df, cols, out, sep=" "):
    """Combine multiple columns into one string column."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logging.warning(f"[combine_columns] Missing columns {missing}; combining only existing ones.")
    existing = [c for c in cols if c in df.columns]
    if not existing:
        df[out] = ""
        return df
    df[out] = df[existing].astype(str).agg(sep.join, axis=1)
    return df


def normalize_col_name(name):
    """
    Normalize column name for fuzzy matching:
    - Lowercase
    - Replace newlines/tabs with space
    - Collapse multiple spaces to one
    - Strip whitespace
    """
    if not isinstance(name, str):
        return str(name)
    return " ".join(name.lower().replace('\n', ' ').replace('\t', ' ').split())

def find_column_fuzzy(df, target_col):
    """
    Find a column in df that matches target_col fuzzily.
    Returns the actual column name in df, or None if no match.
    """
    if target_col in df.columns:
        return target_col
    
    target_norm = normalize_col_name(target_col)
    
    # 1. Try normalized match (exact match after normalization)
    for col in df.columns:
        if normalize_col_name(col) == target_norm:
            return col
            
    # 2. Try "contains" match for common abbreviations (e.g. "High School" -> "school")
    # Only if target is a single word (to avoid false positives)
    if ' ' not in target_norm:
        for col in df.columns:
            col_norm = normalize_col_name(col)
            if target_norm in col_norm:
                return col
                
    return None

def rename(df, col_map=None, mapping=None, **kwargs):
    """
    Rename columns using a dict mapping with fuzzy matching support.
    
    Args:
        col_map: Dict of {old_name: new_name} (legacy parameter)
        mapping: Dict of {old_name: new_name} (new schema parameter)
        
    If BOTH col_map and mapping are provided, applies col_map first, then mapping.
    This handles cases like: col_map cleans newlines, mapping renames to snake_case.
    """
    # Collect all mappings to apply
    all_mappings = []
    if col_map is not None:
        all_mappings.append(col_map)
    if mapping is not None:
        all_mappings.append(mapping)
    
    if not all_mappings:
        logging.warning("[rename] No mapping provided; returning unchanged.")
        return df
    
    # Apply each mapping in order
    for rename_map in all_mappings:
        final_map = {}
        for src, dst in rename_map.items():
            actual_src = find_column_fuzzy(df, src)
            if actual_src:
                final_map[actual_src] = dst
            else:
                logging.debug(f"[rename] Could not find source column '{src}' (even fuzzily). Available: {list(df.columns)}")
        
        if final_map:
            df = df.rename(columns=final_map)
            
    return df


def parse_date_text(
    df,
    col,
    formats=None,
    out=None,  # Generic output column name
    out_col=None,  # Alias for out
    out_date=None,
    out_iso=None,  # Alias for out_date (ISO format output)
    out_parts=None,
    part_names=None,
    default_year=None,
    _expected_writes=None,  # Expected output column names from plan's writes
    **kwargs,
):
    """
    Parse a text column into datetime and optional components.
    Robustly handles pre-1900 dates and various formats.
    
    Args:
        col: Source column with date text
        formats: List of strftime formats to try
        out/out_col: Output column for parsed datetime
        out_date: Output column for parsed datetime (alternative name)
        out_iso: Alias for out_date (ISO format)
        out_parts: List of part names ['year', 'month', 'day', 'day_of_week']
        part_names: Dict mapping part names to column names
        default_year: Default year for dates without year
        _expected_writes: Expected column names from plan (used for exact naming)
        **kwargs: Catch-all for plan variations
    """
    # Handle parameter aliases - plans use various names for the ISO date output
    # Priority: out > out_col > out_date > out_iso
    if out is not None and out_date is None:
        out_date = out
    elif out_col is not None and out_date is None:
        out_date = out_col
    elif out_iso is not None and out_date is None:
        out_date = out_iso
    
    if col not in df.columns:
        logging.warning(f"[parse_date_text] Column '{col}' not found; skipping.")
        return df

    series = df[col].astype(str)
    
    # Pre-process: Remove common suffixes like "; 32 years ago"
    def clean_date_string(val):
        if pd.isna(val):
            return val
        s = str(val)
        # Remove "; X years ago" suffix
        s = re.sub(r';\s*\d+\s*years?\s*ago.*', '', s, flags=re.IGNORECASE)
        # Remove "(age X)" suffix
        s = re.sub(r'\s*\(age\s*\d+\).*', '', s, flags=re.IGNORECASE)
        # Remove parenthetical notes
        s = re.sub(r'\s*\([^)]*years?[^)]*\)', '', s, flags=re.IGNORECASE)
        return s.strip()
    
    series = series.apply(clean_date_string)
    
    # Check if dates have explicit year (4-digit number)
    def _has_year(val):
        if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
            return True  # Don't try to fix empty values
        return bool(re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', str(val)))
    
    has_explicit_year = series.apply(_has_year)

    # Helper for robust parsing
    def _parse_one(val, infer_year=None):
        if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
            return pd.NaT
        
        # 1. Try explicit formats
        if formats:
            for fmt in formats:
                try:
                    return pd.to_datetime(val, format=fmt)
                except:
                    pass
        
        # 2. Try pandas default
        try:
            return pd.to_datetime(val)
        except:
            pass
            
        # 3. Try dateutil (good for "18 January 1849")
        try:
            from dateutil import parser
            parsed_date = parser.parse(str(val))
            # If we inferred a year and the parsed year is current year, replace it
            if infer_year and parsed_date.year >= 2020:
                parsed_date = parsed_date.replace(year=infer_year)
            return parsed_date
        except:
            pass
            
        return pd.NaT
    
    # Try to infer year from other columns in the dataframe that might have years
    inferred_year = None
    if not has_explicit_year.all():
        # First try to find years in this column
        years_in_col = []
        for val in series:
            match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', str(val))
            if match:
                years_in_col.append(int(match.group(1)))
        
        if years_in_col:
            # Use the most common year from this column as default
            from collections import Counter
            inferred_year = Counter(years_in_col).most_common(1)[0][0]
            logging.info(f"[parse_date_text] Inferred default year {inferred_year} from column '{col}'")
        else:
            # Look for years in other columns (e.g., "Passed" column for "Referendum")
            for other_col in df.columns:
                if other_col == col:
                    continue
                other_years = []
                for val in df[other_col]:
                    match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', str(val))
                    if match:
                        other_years.append(int(match.group(1)))
                if other_years:
                    from collections import Counter
                    inferred_year = Counter(other_years).most_common(1)[0][0]
                    logging.info(f"[parse_date_text] Inferred default year {inferred_year} from column '{other_col}'")
                    break
    
    # For row-by-row parsing, try to use year from same row if available
    def _parse_with_row_context(row_idx, val):
        if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
            return pd.NaT
        
        # Check if this value already has a year
        if _has_year(val):
            return _parse_one(val, None)
        
        # Try to get year from the same row in other columns
        row_year = inferred_year
        for other_col in df.columns:
            if other_col == col:
                continue
            other_val = df.iloc[row_idx][other_col]
            match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', str(other_val))
            if match:
                row_year = int(match.group(1))
                break
        
        return _parse_one(val, row_year)

    # Apply row-wise (slower but safer for mixed/messy dates)
    # We try vectorized first for speed, then fallback
    try:
        parsed = pd.to_datetime(series, errors='raise')
    except:
        # Use row-aware parsing if we need to infer years
        if not has_explicit_year.all():
            parsed = pd.Series([_parse_with_row_context(i, v) for i, v in enumerate(series)])
        else:
            parsed = series.apply(lambda v: _parse_one(v, inferred_year))

    # 2) Apply default_year if requested
    if default_year is not None:
        def _apply_year(d):
            if pd.notna(d):
                try:
                    return d.replace(year=default_year)
                except:
                    return d
            return d
        parsed = parsed.apply(_apply_year)

    # 3) Write full date column
    if out_date:
        df[out_date] = parsed
    elif out_date is None and out_parts is None and part_names is None:
        df[col] = parsed

    # 4) Write components - use _expected_writes if available for exact column names
    
    # Helper to find expected column name for a part
    def _find_expected_col(part_type):
        """Find the expected column name from _expected_writes that matches this part type."""
        if not _expected_writes:
            return None
        part_type_lower = part_type.lower()
        for expected in _expected_writes:
            expected_lower = expected.lower()
            # Match patterns like 'Date_year', 'game_year', 'year', etc.
            if part_type_lower in expected_lower:
                # Make sure it's the right part (not 'day' matching 'day_of_week')
                if part_type_lower == 'day' and 'week' in expected_lower:
                    continue
                return expected
        return None
    
    if part_names and isinstance(part_names, dict):
        year_col = part_names.get("year") or _find_expected_col("year")
        month_col = part_names.get("month") or _find_expected_col("month")
        day_col = part_names.get("day") or _find_expected_col("day")
        dow_col = part_names.get("day_of_week") or _find_expected_col("day_of_week")

        if year_col:
            df[year_col] = parsed.apply(lambda x: x.year if pd.notna(x) else np.nan).astype('Int64')
        if month_col:
            df[month_col] = parsed.apply(lambda x: x.month if pd.notna(x) else np.nan).astype('Int64')
        if day_col:
            df[day_col] = parsed.apply(lambda x: x.day if pd.notna(x) else np.nan).astype('Int64')
        if dow_col:
            df[dow_col] = parsed.dt.day_name()
            
    elif out_parts:
        # Handle out_parts as list of part names
        if isinstance(out_parts, list):
            for part in out_parts:
                # First try to find exact expected column name
                col_name = _find_expected_col(part)
                
                # If not found, generate name using source column as prefix
                if not col_name:
                    col_name = f"{col}_{part}"
                
                if part.lower() == "year":
                    df[col_name] = parsed.apply(lambda x: x.year if pd.notna(x) else np.nan).astype('Int64')
                elif part.lower() == "month":
                    df[col_name] = parsed.apply(lambda x: x.month if pd.notna(x) else np.nan).astype('Int64')
                elif part.lower() == "day":
                    df[col_name] = parsed.apply(lambda x: x.day if pd.notna(x) else np.nan).astype('Int64')
                elif "week" in part.lower() or part == "day_of_week":
                    df[col_name] = parsed.dt.day_name()

    return df

def expand_year_range(df, col, patterns=None, out="year", explode=True):
    """
    Expand ranges like '1989/90' into two rows 1989, 1990 (or similar).
    """
    if col not in df.columns:
        logging.warning(f"[expand_year_range] Column '{col}' not found; skipping.")
        return df

    def expand(val):
        if pd.isna(val):
            return []
        s = str(val).strip()
        m = re.match(r"^(\d{4})/(\d{2,4})$", s)
        if m:
            y1 = int(m.group(1))
            y2_str = m.group(2)
            y2 = int(y2_str) if len(y2_str) == 4 else int(str(y1)[:2] + y2_str)
            return [y1, y2]
        return [val]

    df[out] = df[col].apply(expand)
    if explode:
        df = df.explode(out).reset_index(drop=True)
    return df


def parse_pattern(df,
                  col,
                  pattern,
                  output_cols,
                  type_map=None,
                  **kwargs):
    """Extract multiple groups from a regex pattern into multiple columns.
    
    Args:
        col: Source column
        pattern: Regex pattern with capture groups
        output_cols: List of output column names (one per group)
        type_map: Dict mapping column name to type (integer/float/string)
        **kwargs: Catch-all for plan variations
    """
    if col not in df.columns:
        logging.warning(f"[parse_pattern] Column '{col}' not found; skipping.")
        return df
    
    series = df[col].astype(str)
    
    def extract_groups(val):
        if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
            return [None] * len(output_cols)
        
        match = re.search(pattern, str(val))
        if match:
            # Extract all groups
            groups = []
            for i in range(1, len(output_cols) + 1):
                try:
                    groups.append(match.group(i))
                except:
                    groups.append(None)
            return groups
        return [None] * len(output_cols)
    
    # Apply extraction
    extracted = series.apply(extract_groups)
    
    # Create output columns
    for i, col_name in enumerate(output_cols):
        df[col_name] = extracted.apply(lambda x: x[i] if x else None)
        
        # Apply type conversion if specified
        if type_map and col_name in type_map:
            dtype = type_map[col_name]
            if dtype in ['integer', 'int']:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Int64')
            elif dtype in ['float', 'number']:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            # string is the default, no conversion needed
    
    return df


def parse_number(df,
                 col=None,
                 cols=None,  # Support multiple columns
                 allow_commas=True,
                 allow_percent=True,
                 allow_currency=True,
                 out_value=None,
                 out_values=None,  # Support multiple outputs
                 out_unit="unit",
                 type=None,
                 regex=None,
                 pattern=None,
                 **kwargs):
    """Parse numeric text with optional % or currency symbol into value + unit.
    Can also use regex to extract specific numeric values.
    Supports batch processing of multiple columns.
    
    Args:
        col/cols: Source column(s) - use cols for batch processing
        regex/pattern: Optional regex to extract number
        out_value/out_values: Output column(s) for numeric values
        out_unit: Output column for unit (%, $, etc.)
        type: Optional type hint (integer/float)
        **kwargs: Catch-all for plan variations
    """
    # Handle Gemini-style: col as list with out_value as list
    if isinstance(col, list) and isinstance(out_value, list):
        if len(col) != len(out_value):
            logging.warning(f"[parse_number] col and out_value list length mismatch; skipping.")
            return df
        
        for src_col, dst_col in zip(col, out_value):
            if src_col not in df.columns:
                logging.warning(f"[parse_number] Column '{src_col}' not found; skipping.")
                continue
            
            # Recursive call for each column
            df = parse_number(df, col=src_col, out_value=dst_col, out_unit=None,
                            regex=regex, pattern=pattern, type=type,
                            allow_commas=allow_commas, allow_percent=allow_percent,
                            allow_currency=allow_currency)
        return df
    
    # Handle batch processing
    # Mode 1: explicit out_values list
    if cols is not None and out_values is not None:
        if len(cols) != len(out_values):
            logging.warning(f"[parse_number] cols and out_values length mismatch; skipping.")
            return df
        
        for src_col, dst_col in zip(cols, out_values):
            if src_col not in df.columns:
                logging.warning(f"[parse_number] Column '{src_col}' not found; skipping.")
                continue
            
            # Recursive call for each column
            df = parse_number(df, col=src_col, out_value=dst_col, out_unit=None,
                            regex=regex, pattern=pattern, type=type,
                            allow_commas=allow_commas, allow_percent=allow_percent,
                            allow_currency=allow_currency)
        return df
    
    # Mode 2: cols with out_value_suffix (e.g., cols=['Price'], suffix='_numeric' → 'Price_numeric')
    out_value_suffix = kwargs.get('out_value_suffix')
    if cols is not None and out_value_suffix is not None:
        for src_col in cols:
            if src_col not in df.columns:
                logging.warning(f"[parse_number] Column '{src_col}' not found; skipping.")
                continue
            
            dst_col = f"{src_col}{out_value_suffix}"
            
            # Recursive call for each column
            df = parse_number(df, col=src_col, out_value=dst_col, out_unit=None,
                            regex=regex, pattern=pattern, type=type,
                            allow_commas=allow_commas, allow_percent=allow_percent,
                            allow_currency=allow_currency)
        return df
    
    # Single column mode
    if col is None:
        logging.warning(f"[parse_number] No col specified; skipping.")
        return df
    
    if col not in df.columns:
        logging.warning(f"[parse_number] Column '{col}' not found; skipping.")
        return df
    
    # Set default out_value if not provided
    if out_value is None:
        out_value = "value"
    
    # Handle regex parameter alias
    if pattern is not None and regex is None:
        regex = pattern
    
    # MODE 1: Regex extraction (if regex provided)
    if regex:
        series = df[col].astype(str)
        
        # Enhance regex pattern to handle comma-separated numbers
        # If pattern is just [0-9]+, upgrade to [0-9,]+ 
        enhanced_regex = regex
        enhanced_regex = re.sub(r'\[0-9\]\+', r'[0-9,]+', enhanced_regex)
        enhanced_regex = re.sub(r'\\d\+', r'[\\d,]+', enhanced_regex)
        
        def extract_with_regex(val):
            if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                return None
            
            # Try enhanced regex first, then original
            for pattern_to_try in [enhanced_regex, regex]:
                match = re.search(pattern_to_try, str(val))
                if match:
                    # Try to convert first group to number
                    try:
                        # Remove commas before converting
                        num_str = match.group(1).replace(',', '')
                        return float(num_str)
                    except:
                        continue
            return None
        
        df[out_value] = series.apply(extract_with_regex)
        
        # Create out_unit column if explicitly requested (not None in plan)
        if out_unit is not None and out_unit != "":
            df[out_unit] = None
    
    # MODE 2: Traditional parsing (strip commas, currency, percent)
    else:
        def parse(v):
            if pd.isna(v):
                return (None, None)
            s = str(v).strip()
            unit = None

            if allow_percent and s.endswith("%"):
                unit, s = "%", s[:-1]
            if allow_currency and s and s[0] in "$€£₹":
                unit, s = s[0], s[1:]
            if allow_commas:
                s = s.replace(",", "")

            try:
                val = float(s)
            except Exception:
                val = None
            return val, unit

        parsed = df[col].apply(parse)
        df[out_value] = parsed.apply(lambda x: x[0])
        if out_unit is not None and out_unit != "":
            df[out_unit] = parsed.apply(lambda x: x[1])
    
    # Apply type conversion if requested
    if type == 'integer' or type == 'int':
        df[out_value] = pd.to_numeric(df[out_value], errors='coerce').astype('Int64')
    elif type == 'float':
        df[out_value] = pd.to_numeric(df[out_value], errors='coerce')
    
    return df


def map_values(df, col=None, cols=None, mapping=None, out=None, **kwargs):
    """Map values using a dict, leaving unmapped values unchanged.
    
    Args:
        col: Source column to map (single mode)
        cols: Multiple columns to map (batch mode)
        mapping: Dict of old_value -> new_value
        out: Optional output column name (defaults to col, single mode only)
        **kwargs: Catch-all for plan variations
    """
    # Batch mode: apply mapping to multiple columns
    if cols:
        for c in cols:
            if c not in df.columns:
                logging.warning(f"[map_values] Column '{c}' not found; skipping.")
                continue
            df[c] = df[c].map(mapping).fillna(df[c])
        return df
    
    # Single mode
    if not col:
        logging.warning("[map_values] No 'col' or 'cols' specified; skipping.")
        return df
    
    if col not in df.columns:
        logging.warning(f"[map_values] Column '{col}' not found; skipping.")
        return df
    
    out_col = out if out is not None else col
    df[out_col] = df[col].map(mapping).fillna(df[col])
    return df


def one_hot(df, col, prefix=None, **kwargs):
    """One-hot encode a categorical column."""
    if col not in df.columns:
        logging.warning(f"[one_hot] Column '{col}' not found; skipping.")
        return df
    if prefix is None:
        prefix = col
    dummies = pd.get_dummies(df[col], prefix=prefix)
    return pd.concat([df, dummies], axis=1)


def derive_math(df, out, expr, reads=None, **kwargs):
    """
    Compute a column via an arithmetic expression over existing columns.

    Works in two modes:
    - Old style: called with an explicit `reads` list.
    - New style: called with just out/expr; we build env from all columns.
    
    Handles both simple expressions (col1 + col2) and complex pandas expressions
    (df['col'].fillna(0) - df['col2']).
    
    Also handles special functions:
    - len(col) -> string length
    - dayofweek(col), month(col), year(col), day(col) -> datetime components
    """
    # Fallback if out/expr missing
    if out is None or expr is None:
        logging.warning("[derive_math] Missing 'out' or 'expr'; no-op.")
        return df

    # If reads is provided, restrict env to that; else use all columns
    if reads:
        env = {c: df[c] for c in reads if c in df.columns}
    else:
        env = {c: df[c] for c in df.columns}

    try:
        # First, check if expression uses df['col'] syntax (complex pandas expression)
        if "df[" in expr:
            # Execute as Python code with df in scope
            local_env = {'df': df, 'pd': pd, 'np': np}
            result = eval(expr, {}, local_env)
            df[out] = result
        else:
            # Handle special functions that LLMs commonly generate
            safe_expr = expr
            
            # Handle len(col) -> string length
            len_match = re.match(r'len\((\w+)\)', expr)
            if len_match:
                col_name = len_match.group(1)
                # Find matching column
                for c in df.columns:
                    if c == col_name or col_name.lower() in c.lower():
                        df[out] = df[c].astype(str).str.len()
                        return df
                logging.warning(f"[derive_math] Column '{col_name}' not found for len()")
                df[out] = pd.NA
                return df
            
            # Handle datetime functions: dayofweek(col), month(col), year(col), day(col)
            dt_match = re.match(r'(dayofweek|weekday|month|year|day|hour|minute)\((\w+)\)', expr, re.IGNORECASE)
            if dt_match:
                func_name = dt_match.group(1).lower()
                col_name = dt_match.group(2)
                # Find matching column
                for c in df.columns:
                    if c == col_name or col_name.lower() in c.lower():
                        try:
                            dt_col = pd.to_datetime(df[c], errors='coerce')
                            if func_name in ['dayofweek', 'weekday']:
                                df[out] = dt_col.dt.dayofweek
                            elif func_name == 'month':
                                df[out] = dt_col.dt.month
                            elif func_name == 'year':
                                df[out] = dt_col.dt.year
                            elif func_name == 'day':
                                df[out] = dt_col.dt.day
                            elif func_name == 'hour':
                                df[out] = dt_col.dt.hour
                            elif func_name == 'minute':
                                df[out] = dt_col.dt.minute
                            return df
                        except Exception as e:
                            logging.warning(f"[derive_math] Failed datetime function: {e}")
                            df[out] = pd.NA
                            return df
                logging.warning(f"[derive_math] Column '{col_name}' not found for {func_name}()")
                df[out] = pd.NA
                return df
            
            # Handle column names with spaces or special characters
            # Replace column names in expression with safe references
            used_cols = []
            
            # Sort columns by length (longest first) to avoid partial replacements
            sorted_cols = sorted(df.columns, key=len, reverse=True)
            
            for col in sorted_cols:
                if col in safe_expr:
                    # Create a safe variable name
                    safe_name = f"__col_{len(used_cols)}__"
                    safe_expr = safe_expr.replace(col, safe_name)
                    env[safe_name] = pd.to_numeric(df[col], errors='coerce')
                    used_cols.append(col)
            
            # Try pd.eval with the safe expression
            try:
                df[out] = pd.eval(safe_expr, local_dict=env, engine="python")
            except Exception:
                # Fallback: try direct Python eval
                local_env = {'df': df, 'pd': pd, 'np': np}
                local_env.update(env)
                result = eval(safe_expr, {"__builtins__": {}}, local_env)
                df[out] = result
                
    except Exception as e:
        logging.warning(f"[derive_math] Failed expr '{expr}': {e}")
        df[out] = pd.NA
    return df



def _eval_natural_language_cond(df, cond: str):
    """
    Parse natural language conditions from LLM-generated plans.
    
    Handles patterns like:
    - "Column contains 'value'"
    - "Column is not empty"
    - "Column > 5"
    - "Column == 'value'"
    - Compound: "X is null and Y contains 'Z'"
    """
    cond = cond.strip()
    
    # Pattern 0: Handle compound conditions with "and" / "or"
    # Split on ' and ' or ' or ' (case-insensitive)
    and_match = re.split(r'\s+and\s+', cond, flags=re.IGNORECASE)
    if len(and_match) > 1:
        # Compound AND condition
        masks = []
        for sub_cond in and_match:
            sub_mask = _eval_natural_language_cond(df, sub_cond.strip())
            if sub_mask is None:
                return None  # Can't evaluate, fall back
            masks.append(sub_mask)
        result = masks[0]
        for m in masks[1:]:
            result = result & m
        return result
    
    or_match = re.split(r'\s+or\s+', cond, flags=re.IGNORECASE)
    if len(or_match) > 1:
        # Compound OR condition
        masks = []
        for sub_cond in or_match:
            sub_mask = _eval_natural_language_cond(df, sub_cond.strip())
            if sub_mask is None:
                return None  # Can't evaluate, fall back
            masks.append(sub_mask)
        result = masks[0]
        for m in masks[1:]:
            result = result | m
        return result
    
    # Pattern 0.5: Handle "not (X)" pattern
    m = re.match(r"not\s*\((.+)\)", cond, re.IGNORECASE)
    if m:
        inner_cond = m.group(1).strip()
        inner_mask = _eval_natural_language_cond(df, inner_cond)
        if inner_mask is not None:
            return ~inner_mask
        return None
    
    # Pattern 0.6: Handle "Column is null" or "Column is not null"
    m = re.match(r"(.+?)\s+is\s+(not\s+)?null", cond, re.IGNORECASE)
    if m:
        col_name = m.group(1).strip().strip("'\"")
        is_not = m.group(2) is not None
        
        # Find the column
        matched_col = None
        for c in df.columns:
            if c.lower() == col_name.lower() or col_name.lower() == c.lower():
                matched_col = c
                break
        
        if not matched_col:
            logging.warning(f"[_eval_natural_language_cond] Column '{col_name}' not found for null check")
            return None
        
        if is_not:
            return df[matched_col].notna()
        else:
            return df[matched_col].isna()
    
    # Pattern 1: "Column contains 'value'" or "Column contains 'X' or 'Y'"
    m = re.match(r"(.+?)\s+contains\s+(.+)", cond, re.IGNORECASE)
    if m:
        col_name = m.group(1).strip().strip("'\"")
        values_str = m.group(2)
        
        # Find the column (handle spaces in names)
        matched_col = None
        for c in df.columns:
            if c.lower() == col_name.lower() or col_name.lower() in c.lower():
                matched_col = c
                break
        
        if not matched_col:
            logging.warning(f"[_eval_natural_language_cond] Column '{col_name}' not found")
            return None
        
        # Extract all quoted values
        values = re.findall(r"'([^']+)'", values_str)
        if not values:
            values = [values_str.strip().strip("'\"")]
        
        # Build OR mask for all values
        mask = pd.Series(False, index=df.index)
        for val in values:
            mask = mask | df[matched_col].astype(str).str.contains(val, case=False, na=False)
        
        return mask
    
    # Pattern 2: "Column is not empty" or "Column is empty"
    m = re.match(r"(.+?)\s+is\s+(not\s+)?empty", cond, re.IGNORECASE)
    if m:
        col_name = m.group(1).strip().strip("'\"")
        is_not = m.group(2) is not None
        
        # Find the column
        matched_col = None
        for c in df.columns:
            if c.lower() == col_name.lower() or col_name.lower() in c.lower():
                matched_col = c
                break
        
        if not matched_col:
            logging.warning(f"[_eval_natural_language_cond] Column '{col_name}' not found")
            return None
        
        if is_not:
            return df[matched_col].notna() & (df[matched_col].astype(str).str.strip() != '')
        else:
            return df[matched_col].isna() | (df[matched_col].astype(str).str.strip() == '')
    
    # Pattern 3: "Column > value" or "Column >= value" etc.
    m = re.match(r"(.+?)\s*(>|>=|<|<=|==|!=)\s*(.+)", cond)
    if m:
        col_name = m.group(1).strip().strip("'\"")
        op = m.group(2)
        value = m.group(3).strip().strip("'\"")
        
        # Find the column
        matched_col = None
        for c in df.columns:
            if c.lower() == col_name.lower() or col_name.lower() in c.lower():
                matched_col = c
                break
        
        if not matched_col:
            logging.warning(f"[_eval_natural_language_cond] Column '{col_name}' not found")
            return None
        
        try:
            # Try numeric comparison
            val_num = float(value)
            col_num = pd.to_numeric(df[matched_col], errors='coerce')
            
            if op == '>':
                return col_num > val_num
            elif op == '>=':
                return col_num >= val_num
            elif op == '<':
                return col_num < val_num
            elif op == '<=':
                return col_num <= val_num
            elif op == '==':
                return col_num == val_num
            elif op == '!=':
                return col_num != val_num
        except ValueError:
            # String comparison
            if op == '==':
                return df[matched_col].astype(str) == value
            elif op == '!=':
                return df[matched_col].astype(str) != value
    
    # Pattern 4: "Column in ['value1', 'value2']" or "Column in [list]"
    m = re.match(r"(.+?)\s+in\s+\[(.+)\]", cond, re.IGNORECASE)
    if m:
        col_name = m.group(1).strip().strip("'\"")
        values_str = m.group(2)
        
        # Find the column
        matched_col = None
        for c in df.columns:
            if c.lower() == col_name.lower() or col_name.lower() in c.lower():
                matched_col = c
                break
        
        if not matched_col:
            logging.warning(f"[_eval_natural_language_cond] Column '{col_name}' not found for 'in' check")
            return None
        
        # Extract all quoted values
        values = re.findall(r"'([^']+)'", values_str)
        if not values:
            # Try without quotes
            values = [v.strip().strip("'\"") for v in values_str.split(',')]
        
        # Check if column value is in the list
        return df[matched_col].astype(str).isin(values)
    
    # Pattern 5: "Column startswith 'X'" or "Column endswith 'X'"
    m = re.match(r"(.+?)\s+(startswith|endswith)\s+['\"](.+?)['\"]", cond, re.IGNORECASE)
    if m:
        col_name = m.group(1).strip().strip("'\"")
        func = m.group(2).lower()
        value = m.group(3)
        
        # Find the column
        matched_col = None
        for c in df.columns:
            if c.lower() == col_name.lower() or col_name.lower() in c.lower():
                matched_col = c
                break
        
        if not matched_col:
            logging.warning(f"[_eval_natural_language_cond] Column '{col_name}' not found")
            return None
        
        if func == 'startswith':
            return df[matched_col].astype(str).str.startswith(value, na=False)
        else:
            return df[matched_col].astype(str).str.endswith(value, na=False)
    
    return None  # Signal to caller to fall back to eval


def _eval_str_contains_cond(df, cond: str):
    """
    Safe evaluator for patterns like:
      Event.str.contains('Women''s', case=False, na=False)
    """
    m = re.match(r"(\w+)\.str\.contains\((.+)\)", cond.strip())
    if not m:
        return None  # signal to caller to fall back

    col = m.group(1)
    args = m.group(2)

    if col not in df.columns:
        logging.warning(f"[derive_conditional] Column '{col}' not in df for cond '{cond}'")
        return pd.Series(False, index=df.index)

    # Get first argument = pattern
    first_arg, *_ = args.split(",", 1)
    pattern = first_arg.strip()

    # Strip outer quotes if any
    if (pattern.startswith("'") and pattern.endswith("'")) or (
        pattern.startswith('"') and pattern.endswith('"')
    ):
        pattern = pattern[1:-1]

    # Fix doubled single quotes: Women''s -> Women's
    pattern = pattern.replace("''", "'")

    # For now, we just honor case=False, na=False if present; otherwise defaults
    case_flag = "case=False" in args
    na_flag = "na=False" in args

    series = df[col].astype(str)
    return series.str.contains(pattern, case=not case_flag, na=not na_flag)


def derive_conditional(df, out=None, cases=None, conditions=None, default=None, outputs=None, operations=None, **kwargs):
    """
    Derive a column via conditional expressions over df.

    `cond` is a Python-like expression. We special-case simple
    `.str.contains(...)` expressions to avoid fragile eval/quoting issues.
    
    Args:
        out: Output column name (single mode)
        outputs: Alias for out (some plans use this)
        cases: List of {if: condition, then: value} dicts (single mode)
        conditions: Alias for cases (new schema parameter)
        default: Default value if no cases match (single mode)
        operations: List of operations for batch mode (each has out, cases, default)
        **kwargs: Catch-all for plan variations
    """
    # Support both 'conditions' (new schema) and 'cases' (legacy)
    if conditions is not None and cases is None:
        cases = conditions
    
    # Batch mode: operations contains multiple conditional derivations
    if operations:
        for op in operations:
            df = derive_conditional(
                df,
                out=op.get('out'),
                cases=op.get('cases', op.get('conditions')),
                default=op.get('default')
            )
        return df
    
    # Single mode
    # Handle alias
    if outputs is not None and out is None:
        out = outputs
    
    if not out or not cases:
        logging.warning("[derive_conditional] Missing 'out' or 'cases'; skipping.")
        return df
    
    # Handle default value - if it's a column name, use column values
    if isinstance(default, str) and default in df.columns:
        result = df[default].copy()
    elif isinstance(default, str) and re.match(r'^\w+\s*[\+\-\*/]\s*-?\d+\.?\d*$', default):
        # Handle simple expressions like "latitude_decimal * -1" as default
        try:
            col_name = re.match(r'^(\w+)', default).group(1)
            if col_name in df.columns:
                env = {col: df[col] for col in df.columns}
                env.update({"np": np, "pd": pd})
                result = eval(default, {"__builtins__": {}}, env)
                result = pd.Series(result, index=df.index)
            else:
                result = pd.Series(default, index=df.index)
        except:
            result = pd.Series(default, index=df.index)
    else:
        result = pd.Series(default, index=df.index)

    for case in cases:
        # Handle multiple parameter formats from different LLM outputs
        # Format 1: {"if": ..., "then": ...}
        # Format 2: {"condition": ..., "value": ...}
        # Format 3: {"default": ...} - skip, handled above
        cond = case.get("if") if "if" in case else case.get("condition")
        then_val = case.get("then") if "then" in case else case.get("value")
        
        # Skip default entries
        if "default" in case and not cond:
            continue
        
        if not cond:
            continue

        try:
            # 1) Try safe .str.contains(...) fast-path
            mask = _eval_str_contains_cond(df, cond)

            # 2) Fallback to eval if not a .str.contains pattern
            if mask is None:
                # Try natural language parsing first
                mask = _eval_natural_language_cond(df, cond)
            
            # 3) Fallback to eval if natural language didn't work
            if mask is None:
                # Convert SQL syntax to Python
                safe = cond
                safe = re.sub(r"\btrue\b", "True", safe, flags=re.IGNORECASE)
                safe = re.sub(r"\bfalse\b", "False", safe, flags=re.IGNORECASE)
                
                # Convert SQL LIKE to pandas str.contains
                # "col LIKE '%pattern%'" → "col.str.contains('pattern', na=False)"
                # "col LIKE 'pattern%'" → "col.str.startswith('pattern', na=False)" 
                # "col LIKE '%pattern'" → "col.str.endswith('pattern', na=False)"
                def convert_like(match):
                    col_name = match.group(1)
                    pattern = match.group(2).strip("'\"")
                    
                    if pattern.startswith('%') and pattern.endswith('%'):
                        # Contains pattern
                        inner = pattern[1:-1]
                        return f"{col_name}.str.contains('{inner}', na=False, regex=False)"
                    elif pattern.startswith('%'):
                        # Ends with
                        inner = pattern[1:]
                        return f"{col_name}.str.endswith('{inner}', na=False)"
                    elif pattern.endswith('%'):
                        # Starts with
                        inner = pattern[:-1]
                        return f"{col_name}.str.startswith('{inner}', na=False)"
                    else:
                        # Exact match
                        return f"({col_name} == '{pattern}')"
                
                safe = re.sub(r"(\w+)\s+LIKE\s+(['\"][^'\"]+['\"])", convert_like, safe, flags=re.IGNORECASE)
                
                # Convert SQL NULL checks to pandas
                # "col IS NOT NULL" → "col.notna()"
                # "col IS NULL" → "col.isna()"
                safe = re.sub(r"(\w+)\s+IS\s+NOT\s+NULL", r"\1.notna()", safe, flags=re.IGNORECASE)
                safe = re.sub(r"(\w+)\s+IS\s+NULL", r"\1.isna()", safe, flags=re.IGNORECASE)
                
                # Convert SQL AND/OR to Python
                safe = re.sub(r"\bAND\b", "and", safe)
                safe = re.sub(r"\bOR\b", "or", safe)
                safe = re.sub(r"\bNOT\b", "not", safe)
                
                env = {col: df[col] for col in df.columns}
                env.update({"np": np, "pd": pd, "re": re})
                mask = eval(safe, {"__builtins__": {}}, env)

            if not isinstance(mask, (pd.Series, np.ndarray, list)):
                mask = pd.Series(bool(mask), index=df.index)
            else:
                mask = pd.Series(mask, index=df.index)

            # Check if then_val is a column reference or expression
            if isinstance(then_val, str):
                if then_val in df.columns:
                    # Direct column reference
                    result.loc[mask] = df.loc[mask, then_val].values
                elif re.match(r'^[\w\s\+\-\*/\.\(\)]+$', then_val) and any(col in then_val for col in df.columns):
                    # ── Special-case: int(col) / float(col) / str(col) ──────
                    _cast_match = re.match(r'^(int|float|str)\((\w+)\)$', then_val.strip())
                    if _cast_match:
                        _cast_fn, _src_col = _cast_match.group(1), _cast_match.group(2)
                        if _src_col in df.columns:
                            try:
                                if _cast_fn in ('int', 'float'):
                                    # Keep as float64 to avoid dtype mismatch when
                                    # assigning into a mixed-type result Series.
                                    # make_sql_ready() will downcast to Int64 later.
                                    _vals = pd.to_numeric(df.loc[mask, _src_col], errors='coerce')
                                else:
                                    _vals = df.loc[mask, _src_col].astype(str)
                                result.loc[mask] = _vals.values
                            except Exception as e:
                                logging.warning(f"[derive_conditional] Cast '{then_val}' failed: {e}")
                                result[mask] = then_val
                        else:
                            result[mask] = then_val
                    else:
                        # Expression involving columns (e.g., "latitude_decimal * -1")
                        try:
                            env = {col: df[col] for col in df.columns}
                            env.update({"np": np, "pd": pd,
                                        "int": int, "float": float,
                                        "str": str, "bool": bool, "len": len})
                            expr_result = eval(then_val, {"__builtins__": {}}, env)
                            if isinstance(expr_result, pd.Series):
                                result.loc[mask] = expr_result.loc[mask].values
                            else:
                                result[mask] = expr_result
                        except Exception as e:
                            logging.warning(f"[derive_conditional] Failed to eval then_val '{then_val}': {e}")
                            result[mask] = then_val
                else:
                    result[mask] = then_val
            else:
                result[mask] = then_val
        except Exception as e:
            logging.warning(f"[derive_conditional] Failed condition '{cond}': {e}")
            continue

    df[out] = result
    return df

def pivot_wider(df, index, names_from, values_from):
    """Wide pivot."""
    return df.pivot(index=index, columns=names_from, values=values_from).reset_index()


def pivot_longer(df, cols, index=None, names_to=None, values_to=None, **kwargs):
    """Long pivot. Supports aliases for LLM robustness."""
    # Handle aliases from pd.melt
    index = index or kwargs.get("id_vars")
    names_to = names_to or kwargs.get("var_name")
    values_to = values_to or kwargs.get("value_name")
    
    # Drop target columns if they exist to avoid "value_name cannot match element" error
    if names_to in df.columns:
        df = df.drop(columns=[names_to])
    if values_to in df.columns:
        df = df.drop(columns=[values_to])

    return df.melt(id_vars=index, value_vars=cols, var_name=names_to, value_name=values_to)


def fillna_static(df, col, value, replace_values=None, **kwargs):
    """Fill missing values with a static constant.
    
    Args:
        col: Column(s) to fill - can be string or list
        value: Value to fill NaNs with
        replace_values: Alias/alternative parameter (accepted for compatibility)
        **kwargs: Catch-all for plan variations
    """
    # Handle list of columns
    if isinstance(col, list):
        for c in col:
            if c in df.columns:
                if value is not None:
                    df[c] = df[c].fillna(value)
        return df
    
    if col not in df.columns:
        logging.warning(f"[fillna_static] Column '{col}' not found; skipping.")
        return df
    
    if value is None:
        # If value is None, we can't fill NA with it (it's already NA).
        # If the intent was to replace string "nan" with NA, that should be a replace op.
        # We'll assume this is a no-op or a request to ensure the column is nullable.
        logging.warning(f"[fillna_static] value is None for column '{col}'; no fill performed.")
        return df

    df[col] = df[col].fillna(value)
    return df


def fillna_dynamic(df, col=None, cols=None, method="median"):
    """Fill missing values using median/mean/mode/ffill/bfill.
    
    Args:
        col: Single column name (legacy)
        cols: List of column names (newer plans use this)
        method: Fill method - median, mean, mode, ffill, bfill
    """
    # Handle both col (single) and cols (multiple) parameters
    columns_to_fill = []
    if cols is not None:
        columns_to_fill = cols if isinstance(cols, list) else [cols]
    elif col is not None:
        columns_to_fill = [col]
    else:
        logging.warning("[fillna_dynamic] No columns specified.")
        return df
    
    for column in columns_to_fill:
        if column not in df.columns:
            logging.warning(f"[fillna_dynamic] Column '{column}' not found; skipping.")
            continue

        if method == "median":
            val = df[column].median()
        elif method == "mean":
            val = df[column].mean()
        elif method == "mode":
            val = df[column].mode().iloc[0] if not df[column].mode().empty else None
        elif method == "ffill":
            df[column] = df[column].ffill()
            continue
        elif method == "bfill":
            df[column] = df[column].bfill()
            continue
        else:
            val = None

        if val is not None:
            df[column] = df[column].fillna(val)
        else:
            logging.warning(f"[fillna_dynamic] Unknown method '{method}' for column '{column}'; no fill performed.")
    
    return df


def deduplicate_rows(df, keys, keep="first"):
    """Drop duplicate rows based on key columns."""
    existing_keys = [k for k in keys if k in df.columns]
    if not existing_keys:
        logging.warning("[deduplicate_rows] No key columns present; skipping.")
        return df
    before = len(df)
    df = df.drop_duplicates(subset=existing_keys, keep=keep).reset_index(drop=True)
    after = len(df)
    if after < before:
        logging.info(f"[deduplicate_rows] Dropped {before - after} duplicate rows.")
    return df


def filter_rows(df, include=None, exclude=None):
    """Filter rows by boolean expressions."""
    before = len(df)
    if include:
        try:
            df = df.query(include)
        except Exception as e:
            logging.warning(f"[filter_rows] Failed include '{include}': {e}")
    if exclude:
        try:
            df = df.query(f"not ({exclude})")
        except Exception as e:
            logging.warning(f"[filter_rows] Failed exclude '{exclude}': {e}")
    after = len(df)
    if after < before:
        logging.warning(f"[filter_rows] Dropped {before - after} rows — ensure plan.row_change_reason justifies this.")
    return df


def reorder_columns(df, order):
    """Reorder columns exactly as specified."""
    existing = [c for c in order if c in df.columns]
    missing = [c for c in order if c not in df.columns]
    if missing:
        logging.warning(f"[reorder_columns] Missing columns {missing}; they will be skipped.")
    return df[existing]


def sort(df, by, ascending=True):
    """Sort rows by one or more columns."""
    existing = [c for c in by if c in df.columns]
    if not existing:
        logging.warning(f"[sort] None of the sort columns exist: {by}")
        return df
    return df.sort_values(by=existing, ascending=ascending)


def bin_numeric(df, col, bins, labels, out):
    """Bin a numeric column into ranges."""
    if col not in df.columns:
        logging.warning(f"[bin_numeric] Column '{col}' not found; skipping.")
        return df
    df[out] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df


def explode_entities(df,
                     col,
                     pattern=None,
                     out_entity=None,
                     out=None,  # Alias for value_out
                     sep=None,  # Alias for pattern/separators
                     separators=None,
                     value_out=None,
                     explode=True,
                     trim_whitespace=True,
                     **kwargs):
    """
    Generalized multi-entity explode.

    Planner usage (example):
      {
        "op": "explode_entities",
        "col": "normalized_name_list",
        "separators": [","],
        "value_out": "athlete_name",
        "explode": true,
        "trim_whitespace": true
      }
    """
    # Handle parameter aliases
    if out is not None and value_out is None:
        value_out = out
    if out_entity is not None and value_out is None:
        value_out = out_entity
    if sep is not None and pattern is None and separators is None:
        separators = [sep]
    
    if col not in df.columns:
        logging.warning(f"[explode_entities] Column '{col}' not found; skipping.")
        return df

    # Normalize arguments from planner / legacy styles
    if separators is not None and not pattern:
        pattern = "|".join(map(re.escape, separators))
    if not pattern:
        pattern = r","

    if out_entity is None:
        out_entity = value_out if value_out is not None else col

    df[out_entity] = df[col].astype(str).str.split(pattern)
    
    # Trim whitespace from each entity if requested
    if trim_whitespace:
        df[out_entity] = df[out_entity].apply(
            lambda lst: [s.strip() if isinstance(s, str) else s for s in lst] if isinstance(lst, list) else lst
        )
    
    if explode:
        df = df.explode(out_entity).reset_index(drop=True)
    return df


def select(df, cols, action='keep'):
    """Select or drop a subset of columns.
    
    Args:
        cols: List of column names
        action: 'keep' to keep only these columns, 'drop' to remove them
    """
    if action == 'drop':
        # Drop specified columns
        existing = [c for c in cols if c in df.columns]
        df = df.drop(columns=existing)
        return df
    
    # Default: keep only specified columns
    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logging.warning(f"[select] Missing columns {missing}; they will be filled later or treated as NA.")
    return df[existing].copy()


def extract_regex(
    df: pd.DataFrame,
    col: str = None,
    column: str = None,  # Alias for col
    pattern: str = None,
    patterns: list = None,
    out_groups=None,
    output_groups=None,  # Alias for out_groups
    groups=None,
    out_cols=None,  # Alternative parameter name used by some plans
    regex: str = None,  # Alternative parameter name for pattern
    data_types=None,  # Type hints for output columns (ignored, but accepted)
    writes=None,  # Accept writes from plan
    **kwargs  # Catch any other unexpected parameters
):
    """
    Extract named regex groups from a text column into new columns.
    
    Args:
        col/column: Source column(s) to extract from - can be list for multi-column
        pattern: Single regex pattern (legacy)
        patterns: List of regex patterns to try in order (newer plans use this)
        regex: Alias for pattern
        out_groups/output_groups: List of output column names for captured groups
            Can also be list of dicts: [{"name": "_F", "type": "float"}, ...]
        groups: Alias for out_groups
        out_cols: Alias for out_groups (used by some plans)
    """
    # Handle parameter aliases for col
    if col is None and column is not None:
        col = column
    
    # Handle parameter aliases for out_groups
    if out_groups is None and output_groups is not None:
        out_groups = output_groups
    
    # Handle parameter aliases
    if regex is not None and pattern is None:
        pattern = regex
    
    if out_cols is not None and out_groups is None:
        out_groups = out_cols
    
    if out_groups is None and groups is not None:
        out_groups = groups
    
    # Handle out_groups as list of dicts: [{"name": "_F", "type": "float"}, ...]
    out_group_suffixes = None
    out_group_types = {}
    if out_groups and isinstance(out_groups[0], dict):
        out_group_suffixes = [g.get('name', '') for g in out_groups]
        out_group_types = {g.get('name', ''): g.get('type', 'string') for g in out_groups}
        out_groups = out_group_suffixes  # Will be used as suffixes for multi-column mode
    
    # MULTI-COLUMN MODE: If col is a list, apply extraction to each column
    if isinstance(col, list):
        logging.info(f"[extract_regex] Multi-column mode: processing {len(col)} columns")
        for src_col in col:
            if src_col not in df.columns:
                logging.warning(f"[extract_regex] Column '{src_col}' not found; skipping.")
                continue
            
            series = df[src_col].astype("string")
            
            # Try to extract with the pattern
            try:
                extracted = series.str.extract(pattern, expand=True)
                
                # Name output columns based on source column + suffix
                for i, suffix in enumerate(out_group_suffixes or out_groups or []):
                    if i < len(extracted.columns):
                        out_col_name = f"{src_col}{suffix}"
                        df[out_col_name] = extracted[i]
                        
                        # Apply type conversion
                        if out_group_types.get(suffix) in ['float', 'number']:
                            df[out_col_name] = pd.to_numeric(df[out_col_name], errors='coerce')
                        elif out_group_types.get(suffix) in ['int', 'integer']:
                            df[out_col_name] = pd.to_numeric(df[out_col_name], errors='coerce').astype('Int64')
            except Exception as e:
                logging.warning(f"[extract_regex] Failed for column '{src_col}': {e}")
                continue
        
        return df
    
    if col not in df.columns:
        logging.warning(f"[extract_regex] Column '{col}' not found; no-op.")
        return df

    # ensure string dtype
    series = df[col].astype("string")
    
    # NORMALIZE UNICODE: Replace common Unicode variants with ASCII equivalents
    # This fixes issues where regex uses ASCII quotes but data has Unicode primes
    def normalize_unicode(text):
        if pd.isna(text):
            return text
        s = str(text)
        # Prime/quote normalization
        s = s.replace('′', "'")   # Unicode prime → ASCII single quote
        s = s.replace('″', '"')   # Unicode double prime → ASCII double quote
        s = s.replace(''', "'")   # Right single quote → ASCII
        s = s.replace(''', "'")   # Left single quote → ASCII
        s = s.replace('"', '"')   # Left double quote → ASCII
        s = s.replace('"', '"')   # Right double quote → ASCII
        # Dash normalization
        s = s.replace('–', '-')   # En dash → ASCII hyphen
        s = s.replace('—', '-')   # Em dash → ASCII hyphen
        s = s.replace('−', '-')   # Minus sign → ASCII hyphen
        # Other common replacements
        s = s.replace('\ufeff', '')  # BOM removal
        s = s.replace('\xa0', ' ')   # Non-breaking space → regular space
        return s
    
    series_normalized = series.apply(normalize_unicode)
    
    # NEW MODE: patterns as list of dicts (advanced format)
    if patterns and isinstance(patterns, list) and len(patterns) > 0 and isinstance(patterns[0], dict):
        # Each dict has: regex, out_col, type, default
        for pat_config in patterns:
            regex_pat = pat_config.get('regex') or pat_config.get('pattern')
            out_col = pat_config.get('out_col')
            dtype = pat_config.get('type')
            default_val = pat_config.get('default')
            
            if not regex_pat or not out_col:
                continue
            
            def extract_with_default(val):
                if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                    return default_val if default_val is not None else None
                
                match = re.search(regex_pat, str(val))
                if match:
                    try:
                        extracted = match.group(1) if match.groups() else match.group(0)
                        
                        # Apply type conversion
                        if dtype == 'integer' or dtype == 'int':
                            try:
                                return int(extracted)
                            except:
                                return default_val if default_val is not None else None
                        elif dtype == 'boolean' or dtype == 'bool':
                            return bool(match)  # Match exists = True
                        elif dtype == 'float' or dtype == 'number':
                            try:
                                return float(extracted)
                            except:
                                return default_val if default_val is not None else None
                        
                        return extracted
                    except:
                        return default_val if default_val is not None else None
                return default_val if default_val is not None else None
            
            df[out_col] = series_normalized.apply(extract_with_default)
            
            # Post-process type
            if dtype == 'integer' or dtype == 'int':
                df[out_col] = pd.to_numeric(df[out_col], errors='coerce').astype('Int64')
            elif dtype == 'boolean' or dtype == 'bool':
                df[out_col] = df[out_col].astype(bool)
            elif dtype == 'float' or dtype == 'number':
                df[out_col] = pd.to_numeric(df[out_col], errors='coerce')
        
        return df

    # OLD MODE: patterns as list of regex strings
    patterns_to_try = []
    if patterns is not None:
        if isinstance(patterns, list):
            patterns_to_try = patterns
        else:
            patterns_to_try = [patterns]
    elif pattern is not None:
        patterns_to_try = [pattern]
    
    if not patterns_to_try:
        logging.warning("[extract_regex] No pattern specified; no-op.")
        return df

    if not isinstance(out_groups, (list, tuple)):
        logging.warning("[extract_regex] 'out_groups' must be a list/tuple; no-op.")
        return df

    # ENHANCEMENT: Improve numeric patterns to handle comma-separated numbers
    # Patterns like [\d]+ should be [\d,]+ to capture "123,456,789"
    enhanced_patterns = []
    for pat in patterns_to_try:
        # Add comma to digit character classes that might be extracting numbers
        # Pattern: [\\d]+ or [\d]+ → [\\d,]+ or [\d,]+
        enhanced = re.sub(r'\[\\?d\]\+', r'[\\d,]+', pat)
        # Also handle \d+ without brackets → [\d,]+
        enhanced = re.sub(r'(?<![\[\\])\\d\+', r'[\\d,]+', enhanced)
        if enhanced != pat:
            logging.debug(f"[extract_regex] Enhanced pattern: {pat} → {enhanced}")
        enhanced_patterns.append(enhanced)
    
    # Try enhanced patterns first, then original patterns
    all_patterns_to_try = enhanced_patterns + patterns_to_try
    # Remove duplicates while preserving order
    seen = set()
    unique_patterns = []
    for p in all_patterns_to_try:
        if p not in seen:
            seen.add(p)
            unique_patterns.append(p)
    patterns_to_try = unique_patterns
    
    # SMART FALLBACK: Add common alternative patterns based on output column names
    if out_groups:
        out_names_lower = [g.lower() for g in out_groups]
        
        # If extracting vote-related data, add common vote patterns
        if any('vote' in n or 'yes' in n or 'no' in n for n in out_names_lower):
            fallback_vote_patterns = [
                r'([0-9,]+)[–—-]([0-9,]+)',  # "46,153-14,747" format (with various dashes)
                r'(\d[\d,]*)\s*[–—-]\s*(\d[\d,]*)',  # Same with optional spaces
                r'([0-9,]+)',  # Just extract first number if only one group needed
            ]
            for fp in fallback_vote_patterns:
                if fp not in patterns_to_try:
                    patterns_to_try.append(fp)
        
        # If extracting coordinate-related data (lat/lon), add coordinate fallback patterns
        if any(x in out_names_lower for x in ['lat_deg', 'lat_min', 'lon_deg', 'lon_min', 'lat', 'lon', 'latitude', 'longitude']):
            # Check if we need 8 groups (full DMS) or 4 groups (simpler format)
            need_8_groups = len(out_groups) >= 8
            fallback_coord_patterns = []
            
            if need_8_groups:
                # DMS format patterns (8 groups)
                fallback_coord_patterns = [
                    # DMS format with space separator: 39°31'45"N 75°48'50"W
                    r"(\d+)°(\d+)'(?:(\d+)\")?([NS])\s+(\d+)°(\d+)'(?:(\d+)\")?([EW])",
                    # DMS format with / separator: 39°31'45"N / 75°48'50"W
                    r"(\d+)°(\d+)'(?:(\d+)\")?([NS])\s*/\s*(\d+)°(\d+)'(?:(\d+)\")?([EW])",
                ]
            else:
                # Simpler decimal format (4 groups)
                fallback_coord_patterns = [
                    r"(\d+\.?\d*)°([NS])\s+(\d+\.?\d*)°([EW])",
                ]
            
            for fp in fallback_coord_patterns:
                if fp not in patterns_to_try:
                    patterns_to_try.insert(0, fp)  # Insert at front to try first
        
        # If extracting number pairs, add generic patterns
        if len(out_groups) == 2:
            fallback_pair_patterns = [
                r'([0-9,]+)[–—:\-/]\s*([0-9,]+)',  # Number separator Number
                r'(\d+)[–—:\-/]\s*(\d+)',  # Simpler version
            ]
            for fp in fallback_pair_patterns:
                if fp not in patterns_to_try:
                    patterns_to_try.append(fp)

    # Try each pattern until one succeeds
    best_extracted = None
    best_success_rate = 0
    num_expected_groups = len(out_groups)
    
    for pat_idx, pat in enumerate(patterns_to_try):
        try:
            extracted = series_normalized.str.extract(pat, expand=True)
            
            # Skip patterns that don't produce enough groups for our output
            if len(extracted.columns) < num_expected_groups:
                logging.debug(f"[extract_regex] Pattern {pat_idx+1} produced {len(extracted.columns)} groups, need {num_expected_groups}; trying next")
                continue
            
            # Calculate success rate (non-NaN values)
            success_rate = extracted.notna().sum().sum() / (len(extracted) * len(extracted.columns))
            
            if success_rate > best_success_rate:
                best_extracted = extracted
                best_success_rate = success_rate
                
            # If we got good extraction (>30% success), stop trying
            if success_rate > 0.3:
                logging.info(f"[extract_regex] Pattern {pat_idx+1}/{len(patterns_to_try)} succeeded ({success_rate*100:.1f}% extracted)")
                break
                
        except re.error as e:
            logging.warning(f"[extract_regex] Pattern {pat_idx+1} failed: {e}")
            continue
    
    # If no pattern worked well, use fallback strategy
    if best_extracted is None or best_success_rate == 0:
        logging.warning(f"[extract_regex] All patterns failed for column '{col}'. Using fallback strategy.")
        # Fallback: Keep original column for first output group
        for i, g in enumerate(out_groups):
            if i == 0:
                df[g] = df[col]  # Fallback to original
                logging.warning(f"  → Kept original column '{col}' as '{g}'")
            else:
                df[g] = pd.NA
        return df
    
    extracted = best_extracted
    
    # Map extracted columns to output groups with fallback
    if len(extracted.columns) == len(out_groups) and not any(isinstance(c, str) for c in extracted.columns):
        # Unnamed groups (0, 1, 2...)
        for i, g in enumerate(out_groups):
            df[g] = extracted[i]
            # Fallback: If extraction is mostly empty, use original column
            if df[g].notna().sum() < len(df) * 0.1 and len(df) > 0:
                logging.warning(f"[extract_regex] Group '{g}' mostly empty, using original column as fallback")
                df[g] = df[g].fillna(df[col])
    else:
        # Named groups or mismatch
        for g in out_groups:
            if g in extracted.columns:
                df[g] = extracted[g]
                # Fallback for sparse extractions
                if df[g].notna().sum() < len(df) * 0.1 and len(df) > 0:
                    logging.warning(f"[extract_regex] Group '{g}' mostly empty, using original column as fallback")
                    df[g] = df[g].fillna(df[col])
            else:
                logging.warning(f"[extract_regex] Group '{g}' not present in regex result; using original column")
                df[g] = df[col]  # Fallback to original instead of NA

    return df

# --- Escape Hatch ---
# --- Escape Hatch ---
def custom(df, name, reads=None, writes=None, code=None, **kwargs):
    """
    Escape hatch for custom ops.

    Supported:
    - name = "extract_birth_date_and_age"
    - name = "is_prime_number"
    - name = "extract_regex_groups"
    Everything else is a logged no-op.
    """

    # -------------------------------------------------------
    # 1) Extract birth-date text and age from a raw string
    # -------------------------------------------------------
    if name == "extract_birth_date_and_age":
        col = kwargs.get("col")
        date_pattern = kwargs.get("date_pattern")
        age_pattern = kwargs.get("age_pattern")
        out_date = kwargs.get("out_date") or (writes[0] if writes else None)
        out_age = kwargs.get("out_age") or (writes[1] if writes and len(writes) > 1 else None)

        if not col:
            logging.warning("[custom/extract_birth_date_and_age] Missing 'col'; no-op.")
            return df
        if col not in df.columns:
            logging.warning(f"[custom/extract_birth_date_and_age] Source column '{col}' not in df; no-op.")
            return df
        if not date_pattern or not age_pattern:
            logging.warning("[custom/extract_birth_date_and_age] Missing date_pattern or age_pattern; no-op.")
            return df

        try:
            date_re = re.compile(date_pattern)
            age_re = re.compile(age_pattern)
        except re.error as e:
            logging.warning(f"[custom/extract_birth_date_and_age] Invalid regex: {e}; no-op.")
            return df

        def _extract(val):
            if pd.isna(val):
                return None, None
            s = str(val)
            d_match = date_re.search(s)
            a_match = age_re.search(s)
            d = d_match.group(1) if d_match else None
            a = a_match.group(1) if a_match else None
            return d, a

        dates = []
        ages = []
        for v in df[col]:
            d, a = _extract(v)
            dates.append(d)
            ages.append(a)

        if out_date:
            df[out_date] = dates
        if out_age:
            # keep as string; later steps / schema casting can make it numeric
            df[out_age] = ages

        return df

    # -------------------------------------------------------
    # 2) Flag whether a number is prime
    # -------------------------------------------------------
    if name == "is_prime_number":
        col = kwargs.get("col")
        out = kwargs.get("out") or (writes[0] if writes else None)

        if not col or not out:
            logging.warning("[custom/is_prime_number] Missing 'col' or 'out'; no-op.")
            return df
        if col not in df.columns:
            logging.warning(f"[custom/is_prime_number] Column '{col}' not in df; no-op.")
            return df

        def _is_prime(x):
            try:
                n = int(x)
            except Exception:
                return pd.NA
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            i = 3
            while i * i <= n:
                if n % i == 0:
                    return False
                i += 2
            return True

        df[out] = df[col].apply(_is_prime)
        return df

    # -------------------------------------------------------
    # 3) Generic regex-group extractor (your existing behavior)
    # -------------------------------------------------------
    if name == "extract_regex_groups":
        col = kwargs.get("col")
        pattern = kwargs.get("pattern")

        if not col:
            logging.warning("[custom/extract_regex_groups] Missing 'col' param; no-op.")
            return df
        if col not in df.columns:
            logging.warning(f"[custom/extract_regex_groups] Source column '{col}' not in df; no-op.")
            return df
        if not pattern:
            logging.warning("[custom/extract_regex_groups] Missing 'pattern' param; no-op.")
            return df

        try:
            regex = re.compile(pattern)
        except re.error as e:
            logging.warning(f"[custom/extract_regex_groups] Invalid regex pattern: {e}; no-op.")
            return df

        def _extract_groups(val):
            if pd.isna(val):
                return {}
            m = regex.search(str(val))
            return m.groupdict() if m else {}

        group_dicts = df[col].apply(_extract_groups)
        groups_df = pd.DataFrame(list(group_dicts), index=df.index)

        # Decide which columns to write:
        target_cols = []
        if writes:
            if isinstance(writes, str):
                target_cols = [writes]
            else:
                target_cols = list(writes)
        else:
            target_cols = list(groups_df.columns)

        for out_col in target_cols:
            if out_col in groups_df.columns:
                df[out_col] = groups_df[out_col]
            else:
                logging.warning(
                    f"[custom/extract_regex_groups] Requested write column '{out_col}' "
                    f"not present in regex named groups; filling with NA."
                )
                df[out_col] = pd.NA

        return df

    # -------------------------------------------------------
    # 4) Execute code_hint if provided
    # -------------------------------------------------------
    code_hint = kwargs.get('code_hint')
    out_cols = kwargs.get('out_cols', writes or [])
    
    if code_hint:
        try:
            # Create a safe execution environment
            local_env = {'df': df.copy(), 'pd': pd, 'np': np, 're': re}
            
            # Check if code_hint is a complete assignment or needs wrapping
            code_hint_clean = code_hint.strip()
            
            # Case 1: It's a function definition without application
            if code_hint_clean.startswith('def ') and out_cols:
                # Extract function name and apply it
                func_match = re.match(r'def\s+(\w+)\s*\(', code_hint_clean)
                if func_match:
                    func_name = func_match.group(1)
                    out_col = out_cols[0] if out_cols else 'result'
                    # Find which column to apply to (from reads)
                    source_col = reads[0] if reads else None
                    if source_col and source_col in df.columns:
                        full_code = f"{code_hint_clean}\ndf['{out_col}'] = df['{source_col}'].apply({func_name})"
                        exec(full_code, {}, local_env)
                        df = local_env['df']
                        logging.info(f"[custom op] Executed wrapped function code_hint for '{name}'")
                        return df
            
            # Case 2: It's an expression without assignment (e.g., df['col'].apply(...))
            elif not ('df[' in code_hint_clean and '=' in code_hint_clean) and out_cols:
                out_col = out_cols[0]
                full_code = f"df['{out_col}'] = {code_hint_clean}"
                exec(full_code, {}, local_env)
                df = local_env['df']
                logging.info(f"[custom op] Executed wrapped expression code_hint for '{name}'")
                return df
            
            # Case 3: It's already a complete assignment
            else:
                exec(code_hint_clean, {}, local_env)
                df = local_env['df']
                logging.info(f"[custom op] Executed code_hint successfully for '{name}'")
                return df
                
        except Exception as e:
            logging.warning(f"[custom op] code_hint execution failed for '{name}': {e}")
            # Fall through to create empty columns
    
    # -------------------------------------------------------
    # 5) Handle description-based simple operations
    # -------------------------------------------------------
    description = kwargs.get('description', '')
    if writes and description:
        # Try to infer operation from description
        desc_lower = description.lower()
        
        # Common patterns: "normalize X by lowercasing", "clean X", "trim X"
        if any(word in desc_lower for word in ['lowercase', 'lower', 'normalize', 'clean', 'trim']):
            # Find source column from reads
            source_col = reads[0] if reads else None
            if source_col and source_col in df.columns:
                for out_col in writes:
                    if out_col not in df.columns:
                        df[out_col] = df[source_col].astype(str).str.lower().str.strip()
                        logging.info(f"[custom op] Inferred normalize operation: {source_col} -> {out_col}")
                return df
    
    # -------------------------------------------------------
    # Default: Create empty columns for writes to prevent cascade failures
    # -------------------------------------------------------
    logging.warning(f"[custom op] '{name}' has no implementation. Creating empty columns for writes.")
    if writes:
        for w in writes:
            if w not in df.columns:
                df[w] = pd.NA
    return df

def parse_tennis_score(df, score_col, outcome_col, writes=None, winner_value="Winner", **kwargs):
    """
    Parse tennis scores with robust handling for various dash types.
    """
    if score_col not in df.columns or outcome_col not in df.columns:
        logging.warning(f"[parse_tennis_score] Required columns '{score_col}' or '{outcome_col}' not found; skipping.")
        if writes:
            for w_col in writes:
                if w_col not in df.columns:
                    df[w_col] = pd.NA
        return df

    def _parse_single_score_entry(score_str, outcome_str, winner_val):
        # Initialize all expected outputs with NA
        res = {k: pd.NA for k in (writes or [])} 
        
        if pd.isna(score_str):
            return res

        score_str = str(score_str).strip()
        is_winner_row = (str(outcome_str).strip() == winner_val)

        # Handle default scores first
        default_patterns = re.compile(r"DEF|W/O|RET|Walkover|Retired", re.IGNORECASE)
        if default_patterns.search(score_str):
            res["is_default_score"] = True
            return res
        res["is_default_score"] = False

        # Regex for individual set scores (e.g., "6-4", "7-6(3)")
        # Robust regex for dashes: [-–—] (hyphen, en-dash, em-dash)
        set_pattern = r"(\d+[-–—]\d+(?:\(\d+\))?)"
        sets = re.findall(set_pattern, score_str)

        total_sets_played = len(sets)
        player_sets_won = 0
        opponent_sets_won = 0
        player_games_won = 0
        opponent_games_won = 0
        player_won_first_set = pd.NA
        player_won_in_straight_sets = pd.NA

        if total_sets_played == 0:
            return res # No valid sets found

        for i, s in enumerate(sets):
            # Extract games from "X-Y" or "X-Y(tiebreak)"
            game_scores = re.match(r"(\d+)[-–—](\d+)", s)
            if game_scores:
                g1 = int(game_scores.group(1))
                g2 = int(game_scores.group(2))

                # Assumes score is always Player - Opponent
                current_player_games = g1
                current_opponent_games = g2
                
                player_games_won += current_player_games
                opponent_games_won += current_opponent_games

                if current_player_games > current_opponent_games:
                    player_sets_won += 1
                    if i == 0: player_won_first_set = True
                else:
                    opponent_sets_won += 1
                    if i == 0: player_won_first_set = False
            else:
                logging.warning(f"Could not parse set score: {s}")

        res["total_sets_played"] = total_sets_played
        res["player_sets_won"] = player_sets_won
        res["opponent_sets_won"] = opponent_sets_won
        res["total_games_played"] = player_games_won + opponent_games_won
        res["player_games_won"] = player_games_won
        res["opponent_games_won"] = opponent_games_won
        res["player_won_first_set"] = player_won_first_set
        
        if player_sets_won > 0 and opponent_sets_won == 0 and total_sets_played == player_sets_won:
            player_won_in_straight_sets = True
        else:
            player_won_in_straight_sets = False
        res["player_won_in_straight_sets"] = player_won_in_straight_sets

        return res
    
    # Apply the parsing function
    parsed_scores = df.apply(
        lambda row: _parse_single_score_entry(
            row[score_col],
            row[outcome_col],
            winner_value
        ),
        axis=1,
        result_type='expand'
    )
    
    # Assign the results to the DataFrame
    if writes:
        for output_col in writes:
            if output_col in parsed_scores.columns:
                df[output_col] = parsed_scores[output_col]
            else:
                df[output_col] = pd.NA 

    logging.info(f"[parse_tennis_score] Successfully parsed tennis scores.")
    return df

def forward_fill(df, col):
    """
    Forward fill missing values in a column.
    Useful for 'Header -> Detail' row structures where a value is only present in the first row of a group.
    """
    if col not in df.columns:
        logging.warning(f"[forward_fill] Column '{col}' not found; skipping.")
        return df
    
    # Replace empty strings with NaN before filling
    df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
    df[col] = df[col].ffill()
    return df

# --- Custom Ops Implementations ---

def derive_columns(df, columns_map=None, add_constant_col=None, **kwargs):
    """
    Renames columns and optionally adds a constant column.
    params:
        columns_map: dict {old_name: new_name}
        add_constant_col: dict {col_name: str, value: any}
    """
    # 1. Rename
    if columns_map:
        # Filter out columns that don't exist to avoid errors, 
        # or let rename handle it (pandas rename ignores missing cols by default)
        df = df.rename(columns=columns_map)
    
    # 2. Add constant
    if add_constant_col:
        col_name = add_constant_col.get("col_name")
        value = add_constant_col.get("value")
        if col_name:
            df[col_name] = value
            
    return df


def extract_pattern(df, col, pattern, capture_groups=None, **kwargs):
    """
    Wrapper for extract_regex to match the planner's 'extract_pattern' op.
    """
    return extract_regex(df, col=col, pattern=pattern, out_groups=capture_groups)


def union_tables(df, tables_to_union, column_mapping, all_steps=None, **kwargs):
    """
    Unions 'virtual tables' created by previous steps.
    Since we are in a single-df context, this implies:
    1. Identify columns belonging to each 'table' (step).
    2. Create a subset DF for each table.
    3. Rename columns to the target names.
    4. Concat.
    
    Requires 'all_steps' to be passed in params to look up 'writes' of previous steps.
    """
    if not all_steps:
        logging.warning("[union_tables] 'all_steps' not provided; cannot resolve table columns. Returning df.")
        return df

    dfs_to_concat = []
    
    # Create a lookup for steps by ID
    step_map = {s["step_id"]: s for s in all_steps}
    
    for step_id in tables_to_union:
        step = step_map.get(step_id)
        if not step:
            logging.warning(f"[union_tables] Step '{step_id}' not found in plan.")
            continue
            
        # Get columns written by this step
        # The planner puts the output columns in 'writes'
        source_cols = step.get("writes", [])
        
        # Filter for columns that are actually in the mapping
        # (Sometimes writes includes intermediate cols not in the final union)
        # But we need to map source -> target.
        # The column_mapping is global {source: target}.
        
        # Subset the mapping for this step
        step_mapping = {k: v for k, v in column_mapping.items() if k in source_cols}
        
        if not step_mapping:
            logging.warning(f"[union_tables] No columns from step '{step_id}' found in mapping.")
            continue
            
        # Select and rename
        # We must ensure the columns exist in df
        valid_cols = [c for c in step_mapping.keys() if c in df.columns]
        if not valid_cols:
             logging.warning(f"[union_tables] None of the columns {list(step_mapping.keys())} from step '{step_id}' exist in DF.")
             continue
             
        sub_df = df[valid_cols].copy()
        sub_df = sub_df.rename(columns=step_mapping)
        
        # If there are target columns missing (e.g. one table has 'album' but another doesn't),
        # pandas concat will handle it (filling NaN).
        # But we should try to ensure alignment if possible.
        
        dfs_to_concat.append(sub_df)
        
    if not dfs_to_concat:
        logging.warning("[union_tables] No valid tables to union.")
        return df
        
    # Concat
    result_df = pd.concat(dfs_to_concat, ignore_index=True)
    return result_df


def cast_column(df, col, dtype, out=None, **kwargs):
    """Cast a column to a specific data type. Supports single column or list of columns."""
    
    # Handle list of columns
    if isinstance(col, list):
        for c in col:
            if c in df.columns:
                df = cast_column(df, c, dtype, out=None)
        return df
    
    if col not in df.columns:
        logging.warning(f"[cast_column] Column '{col}' not found; skipping.")
        return df
    
    out_col = out if out else col
    
    try:
        if dtype in ['integer', 'int']:
            df[out_col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif dtype in ['float', 'number']:
            df[out_col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype in ['string', 'str', 'text']:
            df[out_col] = df[col].astype(str)
        elif dtype in ['boolean', 'bool']:
            df[out_col] = df[col].astype(bool)
        else:
            logging.warning(f"[cast_column] Unknown dtype '{dtype}'; skipping.")
    except Exception as e:
        logging.warning(f"[cast_column] Failed to cast '{col}' to {dtype}: {e}")
    
    return df


def copy_and_replace(df, col, out, replacements=None, **kwargs):
    """Copy a column and optionally apply replacements."""
    if col not in df.columns:
        logging.warning(f"[copy_and_replace] Column '{col}' not found; skipping.")
        return df
    
    df[out] = df[col].copy()
    
    if replacements:
        df[out] = df[out].replace(replacements)
    
    return df


def replace_value(df, col=None, columns=None, old_value=None, new_value=None, mapping=None, out=None, **kwargs):
    """
    Replace a specific value in a column.
    
    Args:
        col: Source column (single mode) - can be string or list
        columns: List of columns (batch mode)
        old_value: Value to replace (single mode)
        new_value: New value (single mode)
        mapping: Dict of {old_value: new_value} (batch mode)
        out: If specified, create a new column with the result instead of modifying in place
    """
    # Handle col as list
    if isinstance(col, list):
        for c in col:
            if c in df.columns:
                df[c] = df[c].replace(old_value, new_value)
        return df
    
    # Batch mode: multiple columns with mapping
    if columns and mapping:
        for c in columns:
            if c not in df.columns:
                logging.warning(f"[replace_value] Column '{c}' not found; skipping.")
                continue
            for old_val, new_val in mapping.items():
                df[c] = df[c].replace(old_val, new_val)
        return df
    
    # Single mode
    if col is None:
        logging.warning(f"[replace_value] No 'col' specified; skipping.")
        return df
        
    if col not in df.columns:
        logging.warning(f"[replace_value] Column '{col}' not found; skipping.")
        return df
    
    result = df[col].replace(old_value, new_value)
    
    # If 'out' is specified, create a new column
    if out:
        df[out] = result
    else:
        df[col] = result
    
    return df


def replace_string(df, col, pattern, replacement, regex=True, out=None, writes=None, **kwargs):
    """Replace string patterns in a column.
    
    Args:
        col: Source column (can be string or list)
        pattern: Pattern to replace
        replacement: Replacement string
        regex: Whether pattern is regex
        out: Output column name (if different from input)
        writes: List of output column names (alternative to out)
    """
    # Handle list of columns
    if isinstance(col, list):
        out_cols = writes or col
        for i, c in enumerate(col):
            if c in df.columns:
                out_col = out_cols[i] if i < len(out_cols) else c
                df[out_col] = df[c].astype(str).str.replace(pattern, replacement, regex=regex)
        return df
    
    # Single column
    if col not in df.columns:
        logging.warning(f"[replace_string] Column '{col}' not found; skipping.")
        return df
    
    # Determine output column
    out_col = out or (writes[0] if writes else col)
    
    df[out_col] = df[col].astype(str).str.replace(pattern, replacement, regex=regex)
    return df


def fill_down(df, col, **kwargs):
    """Forward-fill missing values in a column."""
    if col not in df.columns:
        logging.warning(f"[fill_down] Column '{col}' not found; skipping.")
        return df
    
    df[col] = df[col].ffill()
    return df


def clean_text(df, col, operations=None, **kwargs):
    """Clean text in a column (strip, lowercase, etc.)."""
    if col not in df.columns:
        logging.warning(f"[clean_text] Column '{col}' not found; skipping.")
        return df
    
    series = df[col].astype(str)
    
    if not operations:
        operations = ['strip']
    
    for op in operations:
        if op == 'strip':
            series = series.str.strip()
        elif op in ['lowercase', 'lower']:
            series = series.str.lower()
        elif op in ['uppercase', 'upper']:
            series = series.str.upper()
    
    df[col] = series
    return df


def explode_column(df, col, **kwargs):
    """Explode a column containing lists into multiple rows."""
    if col not in df.columns:
        logging.warning(f"[explode_column] Column '{col}' not found; skipping.")
        return df
    
    df = df.explode(col).reset_index(drop=True)
    return df


def extract_structured_data(df, col, out_cols, patterns=None, **kwargs):
    """Extract structured data from a column using patterns."""
    if col not in df.columns:
        logging.warning(f"[extract_structured_data] Column '{col}' not found; skipping.")
        return df
    
    if not patterns or len(patterns) != len(out_cols):
        logging.warning(f"[extract_structured_data] Pattern count mismatch; skipping.")
        return df
    
    series = df[col].astype(str)
    
    # Extract each pattern into its output column
    for pattern, out_col in zip(patterns, out_cols):
        def extract(val):
            if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                return None
            
            match = re.search(pattern, str(val))
            if match:
                try:
                    return match.group(1)
                except:
                    return None
            return None
        
        df[out_col] = series.apply(extract)
    
    return df


def extract_and_flag(df, col, pattern, out_extracted, out_flag, **kwargs):
    """Extract data from a column and create a boolean flag.
    
    Args:
        col: Source column
        pattern: Regex pattern to extract
        out_extracted: Output column for extracted value
        out_flag: Output column for boolean flag (True if extraction succeeded)
        **kwargs: Catch-all for plan variations
    """
    if col not in df.columns:
        logging.warning(f"[extract_and_flag] Column '{col}' not found; skipping.")
        return df
    
    series = df[col].astype(str)
    
    def extract(val):
        if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
            return None, False
        
        match = re.search(pattern, str(val))
        if match:
            try:
                return match.group(1), True
            except:
                return None, False
        return None, False
    
    results = series.apply(extract)
    df[out_extracted] = results.apply(lambda x: x[0])
    df[out_flag] = results.apply(lambda x: x[1])
    
    return df
