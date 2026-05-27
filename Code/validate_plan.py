"""
Plan Validator for WikiTQ Pipeline
Validates Step 2 output against operation_schema.json
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional

# Load schema on module import
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "operation_schema.json")
with open(SCHEMA_PATH) as f:
    SCHEMA = json.load(f)

OPERATIONS = SCHEMA["operations"]


class ValidationError:
    """Represents a single validation error."""
    def __init__(self, step_id: str, error_type: str, message: str):
        self.step_id = step_id
        self.error_type = error_type
        self.message = message
    
    def __repr__(self):
        return f"[{self.step_id}] {self.error_type}: {self.message}"


class PlanValidationResult:
    """Result of plan validation."""
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def add_error(self, step_id: str, error_type: str, message: str):
        self.errors.append(ValidationError(step_id, error_type, message))
    
    def add_warning(self, step_id: str, error_type: str, message: str):
        self.warnings.append(ValidationError(step_id, error_type, message))
    
    def summary(self) -> str:
        lines = []
        if self.is_valid:
            lines.append("✅ Plan is VALID")
        else:
            lines.append(f"❌ Plan is INVALID ({len(self.errors)} errors)")
        
        if self.errors:
            lines.append("\nERRORS:")
            for err in self.errors:
                lines.append(f"  • {err}")
        
        if self.warnings:
            lines.append("\nWARNINGS:")
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")
        
        return "\n".join(lines)


def validate_plan(plan: Dict) -> PlanValidationResult:
    """
    Validate a transformation plan against the schema.
    
    Args:
        plan: The plan dictionary (loaded from plan.json)
    
    Returns:
        PlanValidationResult with errors and warnings
    """
    result = PlanValidationResult()
    
    # Check top-level structure
    if "steps" not in plan:
        result.add_error("plan", "MISSING_FIELD", "Plan must have 'steps' array")
        return result
    
    steps = plan["steps"]
    if not isinstance(steps, list):
        result.add_error("plan", "INVALID_TYPE", "'steps' must be an array")
        return result
    
    if len(steps) == 0:
        result.add_error("plan", "EMPTY_PLAN", "Plan has no steps")
        return result
    
    # Track columns created so far (for dependency checking)
    available_columns = set()
    
    # Add original columns if provided
    if "original_columns" in plan:
        available_columns.update(plan["original_columns"])
    
    # Validate each step
    for i, step in enumerate(steps):
        step_id = step.get("step_id", f"step_{i}")
        
        # Check required step fields
        _validate_step_structure(step, step_id, result)
        
        if result.errors:
            continue  # Skip detailed validation if structure is wrong
        
        # Check operation is known
        op = step.get("op")
        if op not in OPERATIONS:
            # Unknown operations are warnings, not errors - they'll be handled by LLM fallback
            result.add_warning(step_id, "UNKNOWN_OPERATION", 
                             f"Operation '{op}' is not in schema. Will use LLM fallback.")
            continue  # Skip param validation for unknown ops
        
        # Validate parameters
        _validate_params(step, step_id, OPERATIONS[op], result)
        
        # Check reads exist (columns must be available)
        reads = step.get("reads", [])
        for col in reads:
            if col not in available_columns and available_columns:
                result.add_warning(step_id, "UNKNOWN_READ", 
                                 f"Reading column '{col}' which hasn't been created yet (might be original column)")
        
        # Add writes to available columns
        writes = step.get("writes", [])
        available_columns.update(writes)
    
    return result


def _validate_step_structure(step: Dict, step_id: str, result: PlanValidationResult):
    """Validate that step has required fields."""
    required = ["step_id", "op", "reads", "writes", "params"]
    
    for field in required:
        if field not in step:
            result.add_error(step_id, "MISSING_FIELD", f"Step missing required field: '{field}'")
    
    # Type checks
    if "reads" in step and not isinstance(step["reads"], list):
        result.add_error(step_id, "INVALID_TYPE", "'reads' must be an array")
    
    if "writes" in step and not isinstance(step["writes"], list):
        result.add_error(step_id, "INVALID_TYPE", "'writes' must be an array")
    
    if "params" in step and not isinstance(step["params"], dict):
        result.add_error(step_id, "INVALID_TYPE", "'params' must be an object")


# Parameter aliases - maps legacy/alternative param names to canonical names
PARAM_ALIASES = {
    "rename": {"col_map": "mapping"},
    "derive_conditional": {"cases": "conditions"},
    "map_values": {"cols": "col"},
    "extract_regex": {
        "column": "col", "input_col": "col",
        "regex": "pattern",
        "output_groups": "out_groups", "out": "out_groups", "outputs": "out_groups",
        "out_cols": "out_groups", "output_cols": "out_groups"
    },
    "parse_number": {
        "column": "col", "input_col": "col",
        "output_value": "out_value", "output_unit": "out_unit",
        "out": "out_value"
    },
    "parse_date_text": {
        "column": "col", "input_col": "col",
        "out_col": "out_date", "output_date": "out_date", "out": "out_date"
    },
    "replace_value": {"column": "col", "input_col": "col", "columns": "col"},  # columns is batch mode
    "replace_string": {"column": "col", "input_col": "col", "out": "col"},
    "cast_column": {"column": "col", "input_col": "col"},
    "derive_math": {"output": "out", "output_col": "out", "result": "out"},
    "fillna_static": {"column": "col", "columns": "col"},
    "fillna_dynamic": {"column": "col", "columns": "col"},
}

# Operations that have alternative modes where required params differ
ALTERNATIVE_MODES = {
    "replace_value": {
        # Batch mode: columns + mapping instead of col + old_value + new_value
        "batch": {"required": ["columns", "mapping"], "replaces": ["col", "old_value", "new_value"]}
    }
}


def _validate_params(step: Dict, step_id: str, op_schema: Dict, result: PlanValidationResult):
    """Validate parameters against operation schema."""
    params = step.get("params", {})
    op = step.get("op", "")
    required_params = op_schema.get("required", [])
    optional_params = op_schema.get("optional", [])
    all_known_params = set(required_params) | set(optional_params)
    
    # Get aliases for this operation
    aliases = PARAM_ALIASES.get(op, {})
    
    # Check for alternative modes (e.g., batch mode for replace_value)
    alt_modes = ALTERNATIVE_MODES.get(op, {})
    using_alt_mode = False
    
    for mode_name, mode_def in alt_modes.items():
        alt_required = mode_def.get("required", [])
        # Check if all alternative required params are present
        if all(p in params for p in alt_required):
            using_alt_mode = True
            # Add alternative params to known params
            all_known_params.update(alt_required)
            # Remove the params that this mode replaces from required
            required_params = [p for p in required_params if p not in mode_def.get("replaces", [])]
            break
    
    # Check required params are present (considering aliases)
    for param in required_params:
        # Check if param exists directly or via alias
        alias_names = [k for k, v in aliases.items() if v == param]
        has_param = param in params or any(alias in params for alias in alias_names)
        
        if not has_param:
            result.add_error(step_id, "MISSING_PARAM", 
                           f"Required parameter '{param}' missing for operation '{op}'")
    
    # Warn about unknown params (might be typos) - but allow known aliases
    all_aliases = set(aliases.keys())
    for param in params:
        if param not in all_known_params and param not in all_aliases:
            result.add_warning(step_id, "UNKNOWN_PARAM", 
                             f"Unknown parameter '{param}' for operation '{op}'. "
                             f"Known params: {list(all_known_params)}")


def validate_plan_file(plan_path: str) -> PlanValidationResult:
    """Validate a plan from a JSON file."""
    with open(plan_path) as f:
        plan = json.load(f)
    return validate_plan(plan)


def get_expected_columns(plan: Dict) -> set:
    """Get all columns that should exist in final table."""
    expected = set()
    for step in plan.get("steps", []):
        expected.update(step.get("writes", []))
    return expected


def get_schema_prompt() -> str:
    """
    Generate a prompt section describing the operation schema.
    This is used in Step 2 to constrain the LLM output.
    """
    lines = [
        "## OPERATION SCHEMA (STRICT - FOLLOW EXACTLY)",
        "",
        "You MUST use ONLY these operations with EXACT parameter names.",
        "Any deviation will cause the pipeline to fail.",
        "",
    ]
    
    for op_name, op_def in OPERATIONS.items():
        required = op_def.get("required", [])
        optional = op_def.get("optional", [])
        
        lines.append(f"### {op_name}")
        lines.append(f"**Description:** {op_def.get('description', 'No description')}")
        
        if required:
            lines.append(f"**Required params:** {', '.join(required)}")
        if optional:
            lines.append(f"**Optional params:** {', '.join(optional)}")
        
        # Add param types if available
        param_types = op_def.get("param_types", {})
        if param_types:
            lines.append("**Param types:**")
            for p, t in param_types.items():
                lines.append(f"  - `{p}`: {t}")
        
        lines.append("")
    
    lines.extend([
        "## STEP STRUCTURE (REQUIRED)",
        "",
        "Each step MUST have these EXACT fields:",
        "```json",
        "{",
        '  "step_id": "s01_description",',
        '  "op": "operation_name",',
        '  "reads": ["columns", "used", "as", "input"],',
        '  "writes": ["columns", "created", "or", "modified"],',
        '  "params": { ... operation parameters ... }',
        "}",
        "```",
        "",
        "## CRITICAL RULES",
        "1. Use ONLY operations listed above",
        "2. Use EXACT parameter names (no variations like 'out_col' instead of 'out')",
        "3. 'writes' must list ALL columns this step creates",
        "4. 'reads' must list ALL columns this step uses as input",
        "5. Column names: lowercase_with_underscores, no spaces, no special characters",
    ])
    
    return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_plan.py <plan.json>")
        print("\nOr use --schema to print the schema prompt:")
        print("  python validate_plan.py --schema")
        sys.exit(1)
    
    if sys.argv[1] == "--schema":
        print(get_schema_prompt())
    else:
        result = validate_plan_file(sys.argv[1])
        print(result.summary())
        sys.exit(0 if result.is_valid else 1)

