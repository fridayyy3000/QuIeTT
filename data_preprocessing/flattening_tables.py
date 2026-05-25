import argparse
import glob
import json
import os

import pandas as pd

def extract_leaves(node, path=None):
    if path is None:
        path = []
    leaves = {}
    new_path = path + [node["name"]]

    if node.get("line_idx") is not None:
        leaves[node["line_idx"]] = " / ".join(new_path[1:])  
    for child in node.get("children_dict", []):
        leaves.update(extract_leaves(child, new_path))
    return leaves

def flatten_table(table_json):
    """Flatten a hierarchical table JSON into a row-wise DataFrame.

    Left-hierarchy levels become row_level_0, row_level_1, ... columns.
    Top-hierarchy paths become column headers.
    Works for any number of hierarchy levels.
    """
    top_map = extract_leaves(table_json["top_root"])
    left_map = extract_leaves(table_json["left_root"])

    rows = []
    for row_idx, row in enumerate(table_json["data"]):
        row_dict = {}

        # Row identifiers from left hierarchy — generic for any depth
        if row_idx in left_map:
            parts = left_map[row_idx].split(" / ")
            for i, part in enumerate(parts):
                row_dict[f"row_level_{i}"] = part

        # Column values from top hierarchy
        for col_idx, cell in enumerate(row):
            if col_idx in top_map:
                row_dict[top_map[col_idx]] = cell["value"]

        rows.append(row_dict)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten hierarchical HiTab-format JSON tables into CSVs for QUIETT."
    )
    parser.add_argument(
        "--input", required=True,
        help="Folder containing hierarchical table JSON files."
    )
    parser.add_argument(
        "--output", required=True,
        help="Folder to write flattened CSV files into."
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = glob.glob(os.path.join(args.input, "*.json"))
    if not files:
        print(f"No JSON files found in {args.input}")
    else:
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                table_json = json.load(f)

            df = flatten_table(table_json)

            table_id = os.path.splitext(os.path.basename(file))[0]
            out_file = os.path.join(args.output, f"{table_id}.csv")
            df.to_csv(out_file, index=False, encoding="utf-8-sig")
            print(f"Saved {out_file}")

        print(f"\nDone. {len(files)} table(s) flattened → {args.output}")
