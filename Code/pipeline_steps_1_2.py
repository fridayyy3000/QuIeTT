import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import re
import pandas as pd
import subprocess
import time
import random

# JSON repair library
try:
    from json_repair import repair_json
    HAS_JSONREPAIR = True
except ImportError:
    HAS_JSONREPAIR = False
    print("[WARN] json-repair not installed. Install with: pip install json-repair")

# LLM API (REST + SDK)
import requests

# ---- LLM Configuration ----
# All values can be overridden via environment variables.
PROJECT_ID = os.environ.get("CLOUD_PROJECT", "YOUR_PROJECT_ID")
LOCATION   = os.environ.get("CLOUD_LOCATION", "us-central1")
MODEL_ID   = os.environ.get("MODEL_ID", "your-model-id")
API_TYPE   = os.environ.get("API_TYPE", "sdk")

# Per-stage token limits (paper §3.2)
STEP1_MAX_TOKENS = int(os.environ.get("STEP1_MAX_TOKENS", "8000"))   # Issue detection
STEP2_MAX_TOKENS = int(os.environ.get("STEP2_MAX_TOKENS", "6000"))   # Plan generation
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "65536"))       # Fallback

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

# Defaults — override via CLI args or environment variables
DEFAULT_INPUT_CSV = Path(os.environ.get("INPUT_CSV", "/path/to/your/table.csv"))
DEFAULT_COL_META_PATH = Path(os.environ.get("COL_DESC_PATH", "/path/to/column_descriptions.json"))

# =========================================================
# PROMPTS
# =========================================================

STEP1_PROMPT = """
You are generating analysis questions from a RAW table.

Table Title:
"{{table_title}}"

Column Descriptions:
{{column_description_text}}

Table (Markdown):
{{raw_markdown_table}}

Objective:
- Produce between 12–20 diverse, realistic questions which will cause issues for SQL generation and also
  cannot be reliably answered directly from this RAW table without cleaning, structuring, or transforming the data.

Output format:
- Return ONLY a JSON ARRAY (no prose, no explanation, no markdown, no backticks). 
- The entire response must be valid JSON starting with [ and ending with ].
- Each item MUST be a JSON OBJECT with these exact keys (strings in double quotes, no single quotes):
  - "qid": string like "Q1", "Q2", ... (contiguous, no gaps)
  - "text": natural language question (escape internal quotes with backslash)
  - "depends_on": list of raw column names, or ["unknown"] if unclear
  - "requires": list of data-prep needs
  - "failure_reason": short phrase on why naive analysis fails

IMPORTANT:
- Return ONLY raw JSON. Do not wrap in markdown code blocks.
- Do not include any text before or after the JSON array.
- All strings must use double quotes, not single quotes.
- Escape any quotes inside string values with backslash.
- No trailing commas.

""".strip()


def extract_issues_from_questions(step1_list):
    """
    Convert Step-1 question list into structured 'issues' array.
    Groups by identical failure_reason.
    """
    issue_map = {}
    counter = 1

    for q in step1_list:
        fr = (q.get("failure_reason") or "").strip()
        cols = q.get("depends_on") or []
        qid = q.get("qid")

        if not fr:
            continue

        if fr not in issue_map:
            issue_map[fr] = {
                "issue_id": f"iss_{counter:03d}",
                "description": fr,
                "cols_set": set(cols),
                "blocking_questions": [qid],
            }
            counter += 1
        else:
            issue_map[fr]["cols_set"].update(cols)
            issue_map[fr]["blocking_questions"].append(qid)

    issues = []
    for data in issue_map.values():
        issues.append({
            "issue_id": data["issue_id"],
            "description": data["description"],
            "cols": sorted(list(data["cols_set"])),
            "blocking_questions": data["blocking_questions"]
        })

    return issues


STEP2_PROMPT = r"""You are STEP_2_PLANNER, a specialist that designs JSON transformation plans
for arbitrary tabular data from any domain (sports, science, finance, government, arts, etc.).

You NEVER write code. You only output a MACHINE-EXECUTABLE PLAN in JSON
that another system will implement.

You are given:
- TABLE_TITLE: short description of the table.
- COLUMN_DESCRIPTIONS: optional text describing columns.
- TABLE_PREVIEW: first 3 rows in Markdown (may NOT show all data patterns).
- ISSUES_JSON: issues detected in Step 1.

=========================================================
YOUR JOB (STEP 2)
=========================================================

Design a transformation plan that:
1. FIXES ALL ISSUES from ISSUES_JSON
2. OPENS UP THE TABLE with derived columns
3. RESPECTS THE RAW SNAPSHOT POLICY

=========================================================
PLAN STRUCTURE
=========================================================

Output ONE JSON OBJECT with this shape:

{
  "table_id": string,
  "strategy": string,
  "steps": [
    {
      "step_id": string,
      "op": string,
      "description": string,
      "reads": [string],
      "writes": [string],
      "params": { ... },
      "fixes_issues": [string],
      "depends_on": [string]
    }
  ],
  "final_output": {
    "primary_key": [string],
    "columns": [
      {
        "name": string,
        "role": "canonical" | "derived" | "helper" | "raw_snapshot"
        // ROLE DEFINITIONS:
        // canonical   : original column KEPT in final table
        //               (derived columns do NOT fully capture all its information)
        // derived     : new column you create; always in final table
        // helper      : intermediate column used only for computation; DROPPED via select
        // raw_snapshot: original column DROPPED from final table
        //               (derived columns FULLY capture all its information)
      }
    ]
  }
}

=========================================================
STRICT OPERATION SCHEMA
=========================================================

| op                | REQUIRED params                            | optional params              |
|-------------------|--------------------------------------------|------------------------------|
| add_row_id        | (none)                                     | out                          |
| rename            | col_map                                    | (none)                       |
| select            | cols                                       | (none)                       |
| parse_date_text   | col, out_date                              | out_parts, formats           |
| parse_number      | col, out_value                             | out_unit, pattern            |
| extract_regex     | col, pattern, out_groups                   | (none)                       |
| derive_conditional| out, conditions                            | default                      |
| derive_math       | out, expr                                  | (none)                       |
| map_values        | col, mapping, out                          | default                      |
| replace_value     | col, old_value, new_value                  | (none)                       |
| replace_string    | col, pattern, replacement                  | regex, writes                |
| cast_column       | col, dtype                                 | out                          |
| fillna_static     | col, value                                 | (none)                       |
| fillna_dynamic    | col, method                                | source_col                   |
| combine_columns   | cols, out, sep                             | (none)                       |
| trim_whitespace   | cols                                       | out (single-col rename)      |
| filter_rows       | condition                                  | (none)                       |
| sort              | by                                         | ascending                    |
| deduplicate_rows  | (none)                                     | subset, keep                 |
| bin_numeric       | col, bins, labels, out                     | (none)                       |
| one_hot           | col                                        | prefix                       |
| split_multi_value | col, separators, value_out                 | explode (bool), count_out    |
| custom            | description, out_cols                      | code_hint                    |

NEVER USE keep_raw_snapshot -- it creates redundant _raw suffix columns.
Use role=raw_snapshot in final_output.columns and drop via select instead.

=========================================================
CRITICAL CONSTRAINTS
=========================================================

1. **derive_conditional format** -- use EXACT structure:
   {
     "conditions": [
       {"condition": "column_name contains 'value'", "value": true},
       {"condition": "column_name == 'other'", "value": false}
     ],
     "default": false
   }
   DO NOT use Python-style "X in ['a','b']" -- use multiple conditions instead.
   For multiple accepted values (e.g., both 'W' and 'Win' map to true), add two conditions.

2. **derive_math format**:
   - String/list length: "len(column_name)"
   - Date parts: "year(col)", "month(col)", "day(col)"
   - Arithmetic: "col_a + col_b", "col_a - col_b", "col_a * col_b"

3. **extract_regex patterns** -- include commas in numeric patterns: "[0-9,]+" not "[0-9]+"

4. **parse_number patterns** -- same: "[0-9,]+" to match comma-formatted numbers.

5. **replace_string with writes** -- to write to a NEW column without modifying the source:
   { "col": "Source", "pattern": "^prefix\\s*", "replacement": "", "writes": ["new_col"] }
   This creates "new_col", leaving "Source" unchanged.
   EACH replace_string step MUST write to a DIFFERENT column (chain sequential steps).

6. **cast_column with out** -- to rename while casting:
   { "col": "day_str", "dtype": "int", "out": "day" }
   Without "out", column keeps its original name (day_str stays day_str).

=========================================================
WHEN TO EXPLODE ROWS (split_multi_value)
=========================================================

Some tables pack MULTIPLE ENTITIES into a single cell.
Each entity deserves its own row so queries can filter or aggregate per entity.

USE split_multi_value with explode=true WHEN:
  - A cell contains a LIST of entities (newline, comma, semicolon, " and " separated)
  - AND each entity is a distinct first-class item (person, team, product, code, etc.)
  - AND the column name implies plurality: Drivers, Players, Members, Artists,
    Co-Drivers, Wrestlers, Authors, Recipients, Co-Stars, etc.
  Examples:
    - "Co-Drivers": "Lewis Hamilton\nNico Rosberg" -> 2 rows, one per driver
    - "Players": "Alice Brown\nBob Chen\nCarla Davis" -> 3 rows, one per player
    - "Wrestlers": "A.J. Petrucci and Doug Stahl" -> separators=[" and "]

  PARAMETERS:
    col: the multi-value column name
    separators: ["\n"] for newline, [","] for comma, [";"] for semicolons, [" and "] for and-sep
    value_out: name for the resulting single-entity column (e.g., "driver_name")
    explode: true

  AFTER EXPLODING (REQUIRED):
  - Add trim_whitespace step on value_out to clean extracted names
  - Add a new add_row_id step AFTER the explode (re-assign sequential IDs post-explode)
  - Mark the original multi-value column as raw_snapshot (DROPPED via select)
  - In strategy, note: "Row count increases -- exploding '<col>' (multi-value per row)"

DO NOT EXPLODE WHEN:
  - Multi-values are properties of one entity (phone prefixes, color codes, emission ranges)
  - The cell is a description, notes, or narrative text field
  - You only need the COUNT of values -> use split_multi_value with explode=false + count_out

=========================================================
COLUMN RETENTION STRATEGY -- LOSSLESS WITHOUT REDUNDANCY
=========================================================

KEEP original column (role=canonical) when:
  - Derived columns capture only PART of the information
  - The column contains free-form text, prose notes, or descriptions
  - Example: "Alice Brown (Chair, 2022)" -> name extracted but title+year are lost -> canonical
  - Example: Racing codes "DNF"/"DNS"/"DSQ" -> status distinctions matter -> canonical
  - Example: Any notes/comments/description column -> prose content -> canonical

DROP original column (role=raw_snapshot) when derived columns FULLY capture ALL info:
  - "1,234,567" -> value_numeric exists -> raw_snapshot
  - "45.2%" -> pct_numeric exists -> raw_snapshot
  - "W 27-7" -> is_win + score_for + score_against exist -> raw_snapshot
  - "September 11" -> month_num + day exist -> raw_snapshot
  - "1990-2005" -> start_year + end_year exist -> raw_snapshot
  - "at San Francisco 49ers" -> is_away + entity_name exist -> raw_snapshot
  - "#13 Entity Name" -> rank_numeric + entity_name exist -> raw_snapshot
  - "Yes"/"No" -> is_X boolean exists -> raw_snapshot
  - "4-2-2" (W-L-T) -> wins + losses + ties exist -> raw_snapshot
  - Multi-value col after explode -> entity_name rows fully represent it -> raw_snapshot

=========================================================
PATTERN LIBRARY -- EXACT STEPS BY DATA TYPE
=========================================================

Apply the matching recipe for each issue. Choose the recipe that fits the data.

----- NUMERIC PATTERNS -----

Numbers with commas ("88,622", "1,234,567", "2,200,000"):
  1. parse_number col="ColName" out_value="colname_numeric"
  DROP original (raw_snapshot).

Currency strings ("$469 billion", "pound 1,000,000", "$27,423"):
  1. parse_number col="ColName" out_value="colname_amount"
  2. IF scale matters: extract_regex col="ColName" pattern="(billion|million|thousand)"
     out_groups=["colname_scale"]
  DROP original if fully captured.

Percentages ("45.2%", "72%", "-0.12"):
  1. extract_regex col="ColName" pattern="(-?[\d.]+)%" out_groups=["pct_str"]
  2. cast_column col="pct_str" dtype="float" out="colname_pct"
  DROP original (raw_snapshot). pct_str -> helper.

Measurements with units ("54.43 m", "5,000 m", "610 mm", "130 feet", "36 mph"):
  1. parse_number col="ColName" out_value="measure_numeric"
  2. IF unit varies: extract_regex col="ColName" pattern="(km|mi|ft|feet|m|mm|kg|lb|mph|kph)"
     out_groups=["measure_unit"]
  DROP original if numeric + unit columns fully capture it.

----- DATE PATTERNS -----

Text month-day dates ("September 11", "Oct 7", "June 22", "August 3"):
  1. extract_regex col="ColName" pattern="([A-Za-z]+)\s+(\d+)" out_groups=["month_name","day_str"]
  2. map_values col="month_name" out="month_num"
     mapping={"January":"1","February":"2","March":"3","April":"4","May":"5","June":"6",
              "July":"7","August":"8","September":"9","October":"10","November":"11","December":"12",
              "Jan":"1","Feb":"2","Mar":"3","Apr":"4","Jun":"6","Jul":"7","Aug":"8",
              "Sep":"9","Oct":"10","Nov":"11","Dec":"12"}
  3. cast_column col="month_num" dtype="int"
  4. cast_column col="day_str" dtype="int" out="day"
  DROP original. month_name -> helper, day_str -> helper.

Full text dates ("September 9, 1990", "June 22 1997", "27 June 2007"):
  1. parse_date_text col="ColName" out_date="date_parsed" out_parts=["year","month_num","day"]
  DROP original.

Year ranges ("1990-2005", "2001-10", "1988-89"):
  1. extract_regex col="ColName" pattern="(\d{4})[-\u2013](\d{2,4})"
     out_groups=["yr_start_str","yr_end_str"]
  2. cast_column col="yr_start_str" dtype="int" out="start_year"
  3. cast_column col="yr_end_str" dtype="int" out="end_year_raw"
     NOTE: 2-digit end year ("89" from "1988-89") -> fix with:
  4. derive_math out="end_year" expr="start_year - (start_year % 100) + end_year_raw"
  5. derive_math out="duration_years" expr="end_year - start_year"
  DROP original. yr_start_str/yr_end_str/end_year_raw -> helper.

Year ranges with "present" ("2011-present"):
  Apply year range steps for the start year.
  derive_conditional out="is_current":
    conditions=[{"condition":"ColName contains 'present'","value":true}] default=false
  DROP original.

----- GAME / MATCH RESULT PATTERNS -----

Win/Loss with score ("W 27-7", "L 15-21", "OTL 3-2", "Win 110-98"):
  1. extract_regex col="ColName" pattern="(W|L|Win|Loss|OTL|OT)\s*(\d+)[-\u2013](\d+)"
     out_groups=["outcome_str","score_a_str","score_b_str"]
  2. derive_conditional out="is_win":
     conditions=[{"condition":"outcome_str == 'W'","value":true},
                 {"condition":"outcome_str == 'Win'","value":true}] default=false
  3. cast_column col="score_a_str" dtype="int" out="score_for"
  4. cast_column col="score_b_str" dtype="int" out="score_against"
  5. derive_math out="score_diff" expr="score_for - score_against"
  DROP original. outcome_str/score_a_str/score_b_str -> helper.
  NEVER output raw "W"/"L" string as a final column -- always produce is_win boolean.

Win-Loss record strings ("4-4", "13-14", "4-2-2" for W-L-T):
  1. extract_regex col="ColName" pattern="(\d+)-(\d+)(?:-(\d+))?"
     out_groups=["wins_str","losses_str","ties_str"]
  2. cast_column "wins_str" dtype="int" out="wins"
  3. cast_column "losses_str" dtype="int" out="losses"
  4. cast_column "ties_str" dtype="int" out="ties"   (NaN where no ties)
  5. derive_math out="games_played" expr="wins + losses"
  DROP original. *_str -> helper.

Two-leg match ("1-2 (H)", "7-1 (A)"):
  1. extract_regex col="ColName" pattern="(\d+)[-\u2013](\d+)\s*\(([HA])\)"
     out_groups=["goals_for_str","goals_against_str","venue_flag"]
  2. cast_column "goals_for_str" dtype="int" out="goals_for"
  3. cast_column "goals_against_str" dtype="int" out="goals_against"
  4. derive_conditional out="is_home_leg":
     conditions=[{"condition":"venue_flag == 'H'","value":true}] default=false
  DROP original. *_str -> helper.

----- ENTITY WITH QUALIFIER PREFIX -----
("at City", "vs. Team B", "#15 Name", "at #8 Name", "Name*"):

These encode multiple facts: home/away flag, rank, and clean entity name.
Use SEPARATE sequential replace_string steps -- EACH writes to a DIFFERENT column:

  1. derive_conditional out="is_away":
     conditions=[{"condition":"ColName startswith 'at '","value":true}] default=false
  2. extract_regex col="ColName" pattern="#(\d+)" out_groups=["rank_str"]
     parse_number col="rank_str" out_value="rank_numeric"
  3. replace_string col="ColName" pattern="^(at\s+|vs\.\s*)" replacement=""
     writes=["name_step1"]
     -> creates name_step1 (strips "at "/"vs."); leaves ColName unchanged
  4. replace_string col="name_step1" pattern="^#\d+\s*" replacement=""
     writes=["name_step2"]
     -> creates name_step2 (strips "#N "); leaves name_step1 unchanged
  5. replace_string col="name_step2" pattern="\*$" replacement=""
     writes=["entity_name"]
     -> creates entity_name (strips trailing *)
  6. trim_whitespace cols=["entity_name"]
  DROP ColName (raw_snapshot). name_step1/name_step2/rank_str -> helper.

----- RACING / TIME PATTERNS -----

Absolute lap times ("1:48:11.023", "2:28:50.8", "3:36"):
  1. extract_regex col="ColName" pattern="(?:(\d+):)?(\d+):(\d+\.?\d*)"
     out_groups=["hr_str","min_str","sec_str"]
  2. fillna_static col="hr_str" value="0"
  3. cast_column "hr_str" dtype="float" out="lap_hours"
  4. cast_column "min_str" dtype="float" out="lap_minutes"
  5. cast_column "sec_str" dtype="float" out="lap_seconds"
  6. derive_math out="lap_time_secs" expr="lap_hours*3600 + lap_minutes*60 + lap_seconds"
  HR_str/min_str/sec_str -> helper. DROP original.

Relative gap times ("+0.8 secs", "+07.3s"):
  1. parse_number col="ColName" out_value="gap_seconds"
  DROP original.

Racing status codes (DNF, DNS, DSQ, Ret, NC):
  KEEP original column as canonical -- status codes carry distinct meaning.
  Optional: derive_conditional out="is_classified":
    conditions=[{"condition":"ColName != 'DNF'","value":true},
                {"condition":"ColName != 'DNS'","value":true}] default=false

Compound location string ("Venue * City, ST", "Stadium, City - Country", "Arena (City)"):
  1. extract_regex using separator in the data (bullet, comma, dash, parens)
     to capture venue_name, city, state_or_country
  2. trim_whitespace on all extracted cols
  DROP original.

----- BOOLEAN / CATEGORICAL PATTERNS -----

Binary text ("Yes"/"No", "Active"/"Inactive", "Won"/"Nominated"):
  1. derive_conditional out="is_X":
     conditions=[{"condition":"ColName == 'Yes'","value":true}] default=false
  DROP original.

Ordinal text positions ("1st", "2nd", "3rd", "4th", "7th"):
  1. map_values col="ColName" out="pos_numeric"
     mapping={"1st":"1","2nd":"2","3rd":"3","4th":"4","5th":"5",
              "6th":"6","7th":"7","8th":"8","9th":"9","10th":"10"}
  2. cast_column col="pos_numeric" dtype="int"
  DROP original if pos_numeric fully captures it.

----- MULTI-VALUE CELL PATTERNS -----

Newline-separated entities ("Player1\nPlayer2\nPlayer3"):
  -> USE split_multi_value with explode=true (see WHEN TO EXPLODE ROWS above)
  separators=["\n"]

"X and Y" format ("Alice Brown and Bob Chen"):
  -> split_multi_value col="ColName" separators=[" and "] value_out="entity_name" explode=true

Comma/semicolon list where only COUNT matters:
  1. split_multi_value col="ColName" separators=[","] value_out="item_list" explode=false
  2. derive_math out="item_count" expr="len(item_list)"

----- TEXT / LABEL PATTERNS -----

Notes/description text (free-form prose):
  KEEP as canonical. Do NOT attempt to parse structured data from free-form notes.

Name with credential/suffix ("Alice Brown, PhD", "Dr. James Liu"):
  Keep original as canonical unless first_name/last_name extraction is specifically needed.

=========================================================
EXAMPLE PLAN A -- Athletics Competition Table
(numeric measurement, ordinal position, boolean, no explode needed)
=========================================================

Columns: "Year", "Competition", "Venue", "Position", "Event", "Best: 54.43 m", "Notes"
Issues: "Best" has unit string not queryable; "Position" is text rank not integer

Plan:
  1. add_row_id
  2. parse_number col="Best: 54.43 m" out_value="best_meters"
  3. map_values col="Position" out="pos_numeric"
     mapping={"1st":"1","2nd":"2","3rd":"3","4th":"4","5th":"5","6th":"6","7th":"7","8th":"8"}
  4. cast_column col="pos_numeric" dtype="int"
  5. derive_conditional out="is_podium":
     conditions=[{"condition":"pos_numeric <= 3","value":true}] default=false
  6. select -> [_row_id, "Year", "Competition", "Venue", pos_numeric, is_podium,
                "Event", best_meters, "Notes"]

  DROPPED: "Best: 54.43 m" -> raw_snapshot ("best_meters" fully captures it)
           "Position" -> raw_snapshot ("pos_numeric" + "is_podium" fully capture it)
  KEPT:    "Year","Competition","Venue","Event" -> canonical
           "Notes" -> canonical (free-form prose)

=========================================================
EXAMPLE PLAN B -- Race Results with Multi-Driver Rows
(multi-value explode, racing times, DNF handling)
=========================================================

Columns: "Pos", "Class", "Team", "Co-Drivers", "Car", "Laps", "Time/Retired"
Issues: "Co-Drivers" packs multiple names per row (newline-separated);
        "Time/Retired" mixes lap times ("1:48:11") and status codes ("DNF")

Plan:
  1. split_multi_value col="Co-Drivers" separators=["\n"] value_out="driver_name" explode=true
     -> row count INCREASES: every driver gets their own row
  2. trim_whitespace cols=["driver_name"]
  3. add_row_id  <- re-assign sequential IDs after explode
  4. derive_conditional out="is_classified":
     conditions=[{"condition":"Time/Retired != 'DNF'","value":true},
                 {"condition":"Time/Retired != 'DNS'","value":true}] default=false
  5. select -> [_row_id, "Pos", "Class", "Team", driver_name, "Car", "Laps",
                "Time/Retired", is_classified]

  DROPPED: "Co-Drivers" -> raw_snapshot (driver_name rows fully represent it)
  KEPT:    "Pos","Class","Team","Car","Laps" -> canonical
           "Time/Retired" -> canonical (mix of times+codes; is_classified only partially captures)
  STRATEGY: "Row count increases -- exploding Co-Drivers (newline-separated, ~2 drivers/row)"

=========================================================
EXAMPLE BAD PLAN -- anti-patterns to NEVER do
=========================================================

  1. writes raw "W" or "L" string as final column -> BAD: always produce is_win boolean
  2. one combined replace_string "^(at |vs. |#N )" -> BAD: misses combos; use sequential steps
  3. cast_column without out="day" -> BAD: column stays named day_str; must use out to rename
  4. keep_raw_snapshot on any column -> BAD: never use this op
  5. No select at end -> BAD: redundant raw columns remain; always end with select
  6. Exploding a notes/description column -> BAD: only explode genuine multi-entity list columns

=========================================================
MANDATORY: ALWAYS END WITH A select STEP
=========================================================

Every plan MUST end with a `select` step as the very last step.

The `cols` list MUST contain:
  - `_row_id` (always first)
  - All derived columns (role: derived)
  - All canonical columns
  - DO NOT include raw_snapshot columns
  - DO NOT include helper columns

ROLE ASSIGNMENT RULES for final_output.columns:

  raw_snapshot (WILL BE DROPPED) when:
  - Numeric string parsed: "1,234" -> colname_numeric -> raw_snapshot
  - Text date split: "Sept 11" -> month_num + day -> raw_snapshot
  - Score split: "W 49-25" -> is_win + score_for + score_against -> raw_snapshot
  - Location split: "Venue * City, ST" -> venue_name + city + state -> raw_snapshot
  - Prefix extracted: "#15 Name" -> rank_numeric + entity_name -> raw_snapshot
  - Boolean replaces text: "Yes"/"No" -> is_X -> raw_snapshot
  - Year range split: "1990-2005" -> start_year + end_year -> raw_snapshot
  - W-L record split: "4-2-2" -> wins + losses + ties -> raw_snapshot
  - Multi-value exploded: "D1\nD2" -> entity_name rows -> raw_snapshot

  canonical (WILL BE KEPT) when:
  - Only part of column info is captured by derived columns
  - Free-form text, notes, descriptions
  - Status codes (DNF/DNS/DSQ) that carry distinct meaning beyond a boolean
  - Name columns where derived cols only partially capture the content

  derived: every column YOUR steps CREATE
  helper: intermediate strings ("day_str", "rank_str", etc.) -- DROPPED via select

DO NOT put raw_snapshot or helper columns in select.cols.
Any raw_snapshot in select.cols will be automatically removed by the runtime.

If you omit the select step entirely, the system will try to infer it -- but YOU should always emit it explicitly.

Return ONLY the JSON object. No markdown, no backticks, no comments.
""".strip()


# =========================================================
# LLM API
# =========================================================

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

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise Exception(f"No response from {MODEL_ID}")
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        raise Exception(f"Empty response from {MODEL_ID}. Response: {data}")
    return content


def call_llm_sdk(prompt: str, temperature: float = 0.2, max_tokens: int = None) -> str:
    """Call a model via the google-genai SDK."""
    if max_tokens is None:
        max_tokens = MAX_TOKENS

    try:
        from google import genai
        from google.genai.types import GenerateContentConfig
    except ImportError:
        raise ImportError("google-genai package required. Install with: pip install google-genai")

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    SYSTEM_INSTRUCTION = (
        "You are a precise JSON generator. "
        "Output ONLY valid JSON exactly as specified in the prompt. "
        "Do not include markdown code fences, prose, or any text outside the JSON."
    )

    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
    if not response or not response.text:
        raise Exception(f"Empty response from {MODEL_ID}")
    return response.text.strip()


def call_model(prompt: str, temperature: float = 0.2, max_tokens: int = None) -> str:
    """Dispatch to the configured LLM backend."""
    if API_TYPE == "sdk":
        return call_llm_sdk(prompt, temperature, max_tokens)
    else:
        return call_llm_openapi(prompt, temperature, max_tokens)


def call_llm_raw(prompt: str, temperature: float = 0.2, max_tokens: int = 16384) -> str:
    """
    Call the configured model and return RAW text.
    Wraps with retry for rate limits.
    """
    max_retries = 3
    delay = 2.0
    
    for attempt in range(max_retries + 1):
        try:
            return call_model(prompt, temperature, max_tokens)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                if attempt == max_retries:
                    raise e
                sleep_time = delay + random.uniform(0, 1)
                print(f"[WARN] Rate limited. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                delay *= 2
            else:
                raise e
    
    return ""


# =========================================================
# HELPERS
# =========================================================

def df_to_markdown(df: pd.DataFrame, max_rows: int = None) -> str:
    """Convert a pandas DataFrame to a simple GitHub-flavored markdown table."""
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    cols = [str(c).replace("\n", " ") for c in df.columns]
    header = "|" + "|".join(cols) + "|"
    separator = "|" + "|".join(["---"] * len(cols)) + "|"

    rows = []
    for _, row in df.iterrows():
        cells = [str(x) for x in row.tolist()]
        rows.append("|" + "|".join(cells) + "|")

    return "\n".join([header, separator] + rows)


def load_title_and_descriptions(csv_path: Path, col_meta_path: Path) -> Tuple[str, str]:
    """Look up TABLE_TITLE and COLUMN_DESCRIPTIONS from metadata file."""
    if not col_meta_path.exists():
        return csv_path.stem, ""
    
    # Try NQTables format (single JSON object with table_id keys)
    try:
        with col_meta_path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("{"):
                all_data = json.loads(content)
                table_id = csv_path.stem
                if table_id in all_data:
                    desc = all_data[table_id]
                    title = table_id.split("_")[0] if "_" in table_id else table_id
                    return title, desc
    except Exception:
        pass
    
    # Fall back to WikiTQ JSONL format
    rel_context = f"csv/{csv_path.parent.name}/{csv_path.name}"
    
    with col_meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("context") == rel_context:
                title = obj.get("table_title", csv_path.stem)
                cols = obj.get("columns", [])
                if isinstance(cols, list):
                    parts = []
                    for col in cols:
                        name = col.get("name", "")
                        desc = col.get("description", "")
                        parts.append(f"{name}: {desc}")
                    desc_str = "\n".join(parts)
                else:
                    desc_str = str(cols)
                return title, desc_str

    return csv_path.stem, ""


def compute_auto_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Very lightweight AUTO_STATS: basic type hints and samples per column."""
    stats: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        sample = non_null.head(5).tolist()
        stats[col] = {
            "dtype": str(series.dtype),
            "n_unique": int(series.nunique(dropna=True)),
            "n_missing": int(series.isna().sum()),
            "sample_values": [str(x) for x in sample],
        }
    return stats


def clean_json_text(text: str) -> str:
    """Aggressively clean LLM output to extract valid JSON."""
    if not text:
        return ""
    
    # Remove markdown code blocks
    text = re.sub(r"^```[a-zA-Z]*\n", "", text.strip())
    text = re.sub(r"\n```$", "", text.strip())
    
    # Extract JSON object or array
    start = text.find('{')
    end = text.rfind('}')
    if start == -1:
        start = text.find('[')
        end = text.rfind(']')
    
    if start != -1 and end != -1:
        text = text[start:end+1]
    
    # Try json-repair library first if available
    if HAS_JSONREPAIR:
        try:
            repaired = repair_json(text)
            # Verify it's valid JSON
            json.loads(repaired)
            return repaired
        except Exception as e:
            print(f"[DEBUG] json-repair failed: {e}, trying regex cleanup...")
    
    # Fallback: regex-based cleaning
    text = re.sub(r"}\s*{", "}, {", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r'(?<!\\)"[\n\r]', '"', text)
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    
    return text


# Canonical separators for multi-value columns
CANONICAL_SEPARATORS = [",", ";", "/", " and ", " & ", "\n"]


def column_has_year(series: pd.Series, min_frac: float = 0.3) -> bool:
    """Check if column contains year values."""
    s = series.dropna().astype(str)
    if s.empty:
        return False
    mask = s.str.contains(r"\b(19|20)\d{2}\b", regex=True)
    return mask.mean() >= min_frac


def sanitize_plan(plan: dict, df_raw: pd.DataFrame) -> dict:
    """Normalize and fix Step-2 plans before execution."""
    steps = plan.get("steps") or []

    for step in steps:
        if not isinstance(step, dict):
            continue

        op = step.get("op")
        params = step.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        step["params"] = params

        # Upgrade separators for split_multi_value.
        # We always union the LLM-planned separators with CANONICAL_SEPARATORS so
        # that common variants the LLM may have missed (e.g. "\n" vs ",") are
        # handled without requiring a separate plan step (Appendix C.3).
        if op == "split_multi_value":
            user_seps = params.get("separators") or []
            if isinstance(user_seps, str):
                user_seps = [user_seps]
            elif not isinstance(user_seps, list):
                user_seps = []
            merged = list({*user_seps, *CANONICAL_SEPARATORS})
            merged.sort()
            params["separators"] = merged

        # Date-year guard
        if op == "parse_date_text":
            writes = step.get("writes") or []
            if isinstance(writes, str):
                writes = [writes]

            date_col = params.get("col")
            if date_col in df_raw.columns:
                if not column_has_year(df_raw[date_col]):
                    new_writes = [c for c in writes if "year" not in c.lower()]
                    step["writes"] = new_writes
                    params.pop("default_year", None)
                    out_parts = params.get("out_parts")
                    if isinstance(out_parts, list):
                        filtered = [p for p in out_parts if "year" not in str(p).lower()]
                        params["out_parts"] = filtered

    # Fix missing renames
    available_cols = set(df_raw.columns)
    rename_step = None
    
    for i, step in enumerate(steps):
        if step.get("op") == "rename":
            rename_step = step
            break
    
    needed_renames = {}

    for step in steps:
        reads = step.get("reads") or []
        for col in reads:
            if col not in available_cols:
                match = None
                col_lower = col.lower()
                
                for raw in df_raw.columns:
                    if raw.lower() == col_lower:
                        match = raw
                        break
                
                if not match:
                    col_norm = " ".join(col_lower.split())
                    for raw in df_raw.columns:
                        raw_norm = " ".join(raw.lower().split())
                        if raw_norm == col_norm:
                            match = raw
                            break
                
                if not match and col_lower == "school":
                    for raw in df_raw.columns:
                        if "school" in raw.lower():
                            match = raw
                            break
                            
                if not match:
                    for raw in df_raw.columns:
                        if raw.lower().replace(" ", "_") == col_lower:
                            match = raw
                            break

                if match:
                    needed_renames[match] = col
                    available_cols.add(col)
                    print(f"[Sanitize] Auto-fixing missing rename: '{match}' -> '{col}'")

        writes = step.get("writes") or []
        for w in writes:
            available_cols.add(w)
            
        if step.get("op") == "rename":
            col_map = step.get("params", {}).get("col_map", {})
            for src, dst in col_map.items():
                available_cols.add(dst)

    if needed_renames:
        if rename_step:
            current_map = rename_step.get("params", {}).get("col_map", {})
            for raw, new in needed_renames.items():
                if raw not in current_map:
                    current_map[raw] = new
            rename_step["params"]["col_map"] = current_map
            
            current_writes = set(rename_step.get("writes", []))
            for new in needed_renames.values():
                current_writes.add(new)
            rename_step["writes"] = list(current_writes)
        else:
            new_step = {
                "step_id": "s00_auto_rename",
                "op": "rename",
                "description": "Automatically injected rename step.",
                "reads": list(needed_renames.keys()),
                "writes": list(needed_renames.values()),
                "params": {"col_map": needed_renames},
                "fixes_issues": [],
                "depends_on": []
            }
            insert_idx = 0
            if steps and steps[0].get("op") == "add_row_id":
                insert_idx = 1
            steps.insert(insert_idx, new_step)

    plan["steps"] = steps

    # Update final_output schema
    final_out = plan.get("final_output") or {}
    cols_spec = final_out.get("columns") or []
    cleaned_cols = []

    for spec in cols_spec:
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        if not name:
            continue
        cleaned_cols.append(spec)

    final_out["columns"] = cleaned_cols
    plan["final_output"] = final_out

    # Build a set of column names the plan marks as raw_snapshot.
    # These are originals that derived columns fully capture — they must be
    # excluded from any select step's cols list.
    raw_snapshot_names = {
        c["name"] for c in cleaned_cols
        if isinstance(c, dict) and c.get("role") == "raw_snapshot"
    }

    # Enforce: remove raw_snapshot columns from any LLM-emitted select steps.
    if raw_snapshot_names:
        for step in plan.get("steps", []):
            if step.get("op") == "select":
                old_cols = step.get("params", {}).get("cols", [])
                new_cols = [c for c in old_cols if c not in raw_snapshot_names]
                if new_cols != old_cols:
                    step["params"]["cols"] = new_cols
                    step["writes"] = new_cols

    # Auto-inject a `select` step if the LLM didn't emit one.
    # Derive the keep-list from final_output.columns: canonical + derived only.
    # This is deterministic and works for every table regardless of LLM behaviour.
    has_select = any(s.get("op") == "select" for s in plan.get("steps", []))
    if not has_select and cleaned_cols:
        keep_roles = {"canonical", "derived"}
        select_cols = [
            c["name"] for c in cleaned_cols
            if c.get("role", "derived") in keep_roles
        ]
        # Always ensure _row_id is first if present
        if "_row_id" in select_cols:
            select_cols = ["_row_id"] + [c for c in select_cols if c != "_row_id"]
        if select_cols:
            plan["steps"].append({
                "step_id": "auto_final_select",
                "op": "select",
                "description": "Auto-injected: keep only canonical+derived columns, drop raw_snapshot originals.",
                "reads": [],
                "writes": select_cols,
                "params": {"cols": select_cols},
                "fixes_issues": [],
                "depends_on": []
            })

    return plan


# =========================================================
# MAIN LOGIC
# =========================================================

def main():
    # Parse CLI args
    input_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_CSV
    col_meta_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_COL_META_PATH

    print(f"[INFO] Using CSV:        {input_csv}")
    print(f"[INFO] Using meta:       {col_meta_path}")
    print(f"[INFO] Using GPT OSS 120B: {MODEL_ID}")

    if not input_csv.exists():
        print(f"[ERR] CSV file not found: {input_csv}")
        sys.exit(1)

    # Create output directory
    pipeline_root_str = os.environ.get("PIPELINE_ROOT_OVERRIDE", str(Path(__file__).parent / "wikitq_gptoss120b_output"))
    pipeline_root = Path(pipeline_root_str)
    
    out_dir = pipeline_root / input_csv.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    step1_raw_path = out_dir / "step1_raw_output.txt"
    step2_raw_path = out_dir / "step2_raw_output.txt"
    questions_path = out_dir / "questions.json"
    issues_path = out_dir / "issues.json"
    plan_path = out_dir / "plan.json"
    step3_code_path = out_dir / "step_pipeline.py"
    final_csv_path = out_dir / "final.csv"

    print(f"[INFO] Output directory: {out_dir}")

    # Load data
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[WARN] Standard read_csv failed: {e}. Retrying with fallback...")
        try:
            df = pd.read_csv(input_csv, engine='python', on_bad_lines='skip')
        except Exception as e2:
            print(f"[ERROR] Failed to load CSV: {e2}")
            raise e2

    raw_md_full = df_to_markdown(df)
    table_title, col_desc = load_title_and_descriptions(input_csv, col_meta_path)
    auto_stats = compute_auto_stats(df)
    auto_stats_json = json.dumps(auto_stats, ensure_ascii=False, indent=2)

    print(f"[INFO] Table title: {table_title}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # STEP 1
    print("[INFO] Calling GPT OSS 120B for STEP 1 (questions/issues)...")

    step1_user_block = f"""TABLE_TITLE: {table_title}

RAW_TABLE_MD:
{raw_md_full}

COLUMN_DESCRIPTIONS:
{col_desc}

AUTO_STATS (JSON):
{auto_stats_json}
"""

    full_step1_prompt = STEP1_PROMPT + "\n\n" + step1_user_block

    try:
        step1_raw = call_llm_raw(full_step1_prompt, temperature=0.2, max_tokens=STEP1_MAX_TOKENS)
    except Exception as e:
        print(f"[ERR] Step 1 failed: {e}")
        sys.exit(1)

    step1_raw_path.write_text(step1_raw, encoding="utf-8")
    print(f"[OK] Wrote Step-1 output to: {step1_raw_path}")

    # Parse Step 1
    step1_json_text = None
    issues = []  # Initialize issues in case of parsing failure
    try:
        cleaned = step1_raw.strip()
        cleaned = re.sub(r"^```json", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        
        # Handle array format
        if cleaned.startswith("["):
            step1_list = json.loads(cleaned)
        else:
            step1_list = json.loads(cleaned)

        questions_path.write_text(json.dumps(step1_list, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Wrote questions.json")

        issues = extract_issues_from_questions(step1_list)
        issues_path.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Wrote issues.json")

        step1_json = {"questions": step1_list, "issues": issues}
        step1_json_text = json.dumps(step1_json, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[WARN] Failed to parse Step-1 JSON: {e}")
        step1_json_text = None
        issues = []  # Ensure issues is initialized on error

    # STEP 2
    print("[INFO] Calling GPT OSS 120B for STEP 2 (planner)...")

    if step1_json_text is not None:
        step1_for_step2 = step1_json_text
        step1_label = "STEP1_JSON"
    else:
        step1_for_step2 = step1_raw
        step1_label = "STEP1_RAW_OUTPUT (fallback)"

    # Send full issues.json for better context
    issues_full = json.dumps(issues, ensure_ascii=False, indent=2) if issues else "{}"
    
    step2_block = f"""
TABLE_TITLE: {table_title}

TABLE_PREVIEW (first 3 rows):
{df_to_markdown(df, max_rows=3)}

COLUMN NAMES:
{', '.join(df.columns)}

ISSUES_JSON (Full context from Step 1):
{issues_full}
"""

    full_step2_prompt = STEP2_PROMPT + "\n\n" + step2_block

    max_retries = 5
    plan_obj = None
    step2_raw = ""
    
    for attempt in range(max_retries):
        print(f"--- Step 2 Attempt {attempt + 1}/{max_retries} ---")
        try:
            # Use STEP2_MAX_TOKENS per paper §3.2
            step2_raw = call_llm_raw(full_step2_prompt, temperature=0.2, max_tokens=STEP2_MAX_TOKENS)
        except Exception as e:
            print(f"[ERR] Step 2 call failed: {e}")
            if attempt == max_retries - 1:
                sys.exit(1)
            time.sleep(5)  # Wait before retry
            continue

        step2_raw_path.write_text(step2_raw, encoding="utf-8")
        
        candidate = clean_json_text(step2_raw)
        try:
            plan_obj = json.loads(candidate)
            print("[SUCCESS] Step 2 JSON parsed successfully.")
            break
        except json.JSONDecodeError as json_err:
            print(f"[WARN] Attempt {attempt+1}: JSON decode failed: {json_err}")
            
            # Try to improve: Ask LLM to fix the JSON
            if attempt < max_retries - 1:
                print(f"[INFO] Asking LLM to fix JSON...")
                fix_prompt = f"""The following JSON is malformed. Please fix it and return ONLY valid JSON:

{step2_raw[:2000]}...

Requirements:
- Return ONLY valid JSON (no markdown, no explanation)
- Ensure all quotes are properly escaped
- Ensure all commas are in correct places
- Start with {{ and end with }}
"""
                try:
                    fixed_raw = call_llm_raw(fix_prompt, temperature=0.2, max_tokens=4000)
                    fixed_candidate = clean_json_text(fixed_raw)
                    plan_obj = json.loads(fixed_candidate)
                    print("[SUCCESS] Fixed JSON parsed successfully.")
                    step2_raw = fixed_raw
                    break
                except Exception as fix_err:
                    print(f"[WARN] LLM fix also failed: {fix_err}")
            
            if attempt == max_retries - 1:
                print("[FATAL] Could not generate valid JSON after retries.")
                debug_dump_path = out_dir / "step2_failed_response_dump.txt"
                debug_dump_path.write_text(f"Error: {json_err}\n\n{step2_raw}", encoding="utf-8")
                sys.exit(1)

    # Post-processing
    try:
        plan_obj = sanitize_plan(plan_obj, df)
        plan_path.write_text(json.dumps(plan_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Wrote plan.json to: {plan_path}")
    except Exception as e:
        print(f"[WARN] Failed to sanitize or write plan.json: {e}")

    print("\n[INFO] Done. Output files:")
    print(f"  - {step1_raw_path}")
    print(f"  - {step2_raw_path}")
    print(f"  - {plan_path}")


if __name__ == "__main__":
    main()