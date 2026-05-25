#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Text normalisation (mirrors run_pipeline.py Appendix J)
# ---------------------------------------------------------------------------

MONTH_ABBREVS = {
    'jan': 'january', 'feb': 'february', 'mar': 'march', 'apr': 'april',
    'jun': 'june', 'jul': 'july', 'aug': 'august', 'sep': 'september',
    'oct': 'october', 'nov': 'november', 'dec': 'december'
}

ORDINALS = {
    '1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth',
    '5th': 'fifth', '6th': 'sixth', '7th': 'seventh', '8th': 'eighth',
    '9th': 'ninth', '10th': 'tenth'
}

ROMAN_NUMERALS = {
    'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
    'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10'
}

_MONTH_NUM_TO_NAME = {
    '01': 'january', '02': 'february', '03': 'march', '04': 'april',
    '05': 'may', '06': 'june', '07': 'july', '08': 'august',
    '09': 'september', '10': 'october', '11': 'november', '12': 'december',
    '1': 'january', '2': 'february', '3': 'march', '4': 'april',
    '5': 'may', '6': 'june', '7': 'july', '8': 'august',
    '9': 'september',
}


def _iso_date_to_text(text: str) -> str:
    """'2018-01-05' -> '5 january 2018'."""
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', text.strip())
    if m:
        year, month, day = m.group(1), m.group(2), m.group(3)
        month_name = _MONTH_NUM_TO_NAME.get(month, month)
        return f"{int(day)} {month_name} {year}"
    return text


def _normalize_number(text: str) -> str:
    text = str(text).strip().replace(',', '')
    if re.match(r'^-?\d+\.0+$', text):
        text = text.split('.')[0]
    return text


def normalize_text(text: str) -> str:
    """Full normalisation pipeline (Appendix J)."""
    text = str(text).lower().strip()
    text = _iso_date_to_text(text)
    text_num = _normalize_number(text)
    if text_num != text:
        text = text_num.lower()
    for abbrev, full in MONTH_ABBREVS.items():
        text = re.sub(rf'\b{abbrev}\b', full, text)
    for ordinal, word in ORDINALS.items():
        text = re.sub(rf'\b{ordinal}\b', word, text)
    for roman, arabic in ROMAN_NUMERALS.items():
        text = re.sub(rf'\b{roman}\b', arabic, text)
    text = re.sub(r'\byes\b', 'true', text)
    text = re.sub(r'\bno\b', 'false', text)
    text = re.sub(r'[@;|]', ' ', text)
    text = re.sub(r'\s*%\s*', ' percent ', text)
    text = re.sub(r'\$\s*', '', text)
    text = re.sub(r'\s*°[cfCF]?\b', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_answer(text: str) -> str:
    """Strip common LLM preambles before scoring."""
    text = str(text).strip()
    for prefix in ("the answer is", "answer:", "answer is", "final answer:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text.strip("'\"").strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def token_f1(predicted: str, target: str) -> float:
    """Token-level F1 between a single predicted string and a single target."""
    pred_n = normalize_text(predicted)
    tgt_n = normalize_text(target)
    if pred_n == tgt_n:
        return 1.0
    # Numeric equality
    try:
        if abs(float(_normalize_number(str(predicted))) - float(_normalize_number(str(target)))) < 0.01:
            return 1.0
    except (ValueError, TypeError):
        pass
    pred_tok = set(pred_n.split())
    tgt_tok = set(tgt_n.split())
    if not pred_tok or not tgt_tok:
        return 0.0
    common = pred_tok & tgt_tok
    if not common:
        return 0.0
    p = len(common) / len(pred_tok)
    r = len(common) / len(tgt_tok)
    return 2 * p * r / (p + r)


def score_prediction(predicted: str, targets: list) -> dict:

    predicted = clean_answer(predicted)
    pred_n = normalize_text(predicted)

    # Exact match against any single target
    for t in targets:
        if pred_n == normalize_text(str(t)):
            return {"exact_match": True, "f1": 1.0}

    # Exact match against combined targets
    for sep in (', ', ' and ', ' '):
        combined = sep.join(str(t) for t in targets)
        if pred_n == normalize_text(combined):
            return {"exact_match": True, "f1": 1.0}

    # Containment: predicted contains all targets
    if len(targets) > 1:
        if all(normalize_text(str(t)) in pred_n for t in targets):
            return {"exact_match": True, "f1": 1.0}

    # Not an exact match — compute best token-F1 for soft scoring only
    best_f1 = max(token_f1(predicted, str(t)) for t in targets)
    combined_str = ' '.join(str(t) for t in targets)
    best_f1 = max(best_f1, token_f1(predicted, combined_str))

    return {"exact_match": False, "f1": best_f1}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> list:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Predictions file must be a JSON list, got {type(data)}")
    return data


def load_gold(path: Path) -> dict:
    """Returns a dict mapping id -> list[str]."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    index = {}
    for item in data:
        qid = str(item.get("id", item.get("qid", "")))
        ans = item.get("answer", item.get("answers", []))
        if isinstance(ans, str):
            ans = [ans]
        index[qid] = [str(a) for a in ans]
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QUIETT pipeline predictions against gold answers."
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="Path to predictions JSON (qa_results.json or simple predictions file).",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Path to gold answers JSON (required when predictions file has no 'targets' field).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write per-question results + summary to this JSON file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary line.",
    )
    args = parser.parse_args()

    preds = load_predictions(args.predictions)

    # Detect format: do records already have 'targets'?
    has_targets = preds and ("targets" in preds[0] or "correct" in preds[0])

    gold_index = {}
    if not has_targets:
        if args.gold is None:
            print(
                "ERROR: Predictions file does not contain 'targets'. "
                "Please supply --gold <gold.json>.",
                file=sys.stderr,
            )
            sys.exit(1)
        gold_index = load_gold(args.gold)

    per_question = []
    total_em = 0
    total_f1 = 0.0

    for rec in preds:
        qid = str(rec.get("qid", rec.get("id", "")))
        question = rec.get("question", "")

        # Predicted answer
        predicted = rec.get("predicted", rec.get("predicted_answer", ""))

        # Targets
        if has_targets:
            targets = rec.get("targets", [])
            if not targets:
                targets = []
        else:
            targets = gold_index.get(qid, [])
            if not targets:
                if not args.quiet:
                    print(f"WARNING: no gold answer found for id={qid!r}; skipping.")
                continue

        scores = score_prediction(str(predicted), [str(t) for t in targets])
        total_em += int(scores["exact_match"])
        total_f1 += scores["f1"]

        per_question.append({
            "qid": qid,
            "question": question,
            "predicted": predicted,
            "targets": targets,
            "exact_match": scores["exact_match"],
            "f1": round(scores["f1"], 4),
        })

    n = len(per_question)
    if n == 0:
        print("No questions evaluated.", file=sys.stderr)
        sys.exit(1)

    em_pct = total_em / n * 100
    avg_f1 = total_f1 / n

    summary = {
        "n_questions": n,
        "exact_match": total_em,
        "exact_match_pct": round(em_pct, 2),
        "avg_f1": round(avg_f1, 4),
    }

    # Print per-question breakdown unless --quiet
    if not args.quiet:
        for r in per_question:
            mark = "\u2713" if r["exact_match"] else "\u2717"
            print(f"[{mark}] {r['qid']:<30} F1={r['f1']:.2f}  pred={r['predicted']!r}  gold={r['targets']}")

    print(
        f"\n{'='*60}\n"
        f"  Questions      : {n}\n"
        f"  Exact Match    : {total_em} / {n}  ({em_pct:.1f}%)\n"
        f"  Avg token-F1   : {avg_f1:.4f}\n"
        f"{'='*60}"
    )

    if args.output:
        out = {"summary": summary, "results": per_question}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Detailed results written to: {args.output}")


if __name__ == "__main__":
    main()
