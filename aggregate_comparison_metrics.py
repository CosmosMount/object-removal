#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


COLUMNS = [
    "method",
    "part_label",
    "experiment_name",
    "mask_jm",
    "mask_jr",
    "video_psnr",
    "video_ssim",
    "mask_source",
    "video_source",
    "metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-method metric JSONs into one CSV/Markdown table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="List of metrics_summary.json paths")
    parser.add_argument("--output_csv", required=True, help="Output combined CSV path")
    parser.add_argument("--output_md", required=True, help="Output combined markdown table path")
    return parser.parse_args()


def _safe_float(v: Any):
    if v is None:
        return ""
    try:
        return float(v)
    except Exception:
        return ""


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def load_row(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    experiment = str(data.get("experiment_name", "")).strip() or "unknown"
    row = {
        "method": experiment,
        "part_label": data.get("part_label", ""),
        "experiment_name": experiment,
        "mask_jm": _safe_float(data.get("mask_jm")),
        "mask_jr": _safe_float(data.get("mask_jr")),
        "video_psnr": _safe_float(data.get("video_psnr")),
        "video_ssim": _safe_float(data.get("video_ssim")),
        "mask_source": data.get("mask_source", ""),
        "video_source": data.get("video_source", ""),
        "metrics_path": str(path),
    }
    return row


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(COLUMNS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in COLUMNS) + " |")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    rows: List[Dict[str, Any]] = []
    missing = []
    for item in args.inputs:
        p = Path(item)
        if not p.exists():
            missing.append(str(p))
            continue
        rows.append(load_row(p))

    rows.sort(key=lambda x: (str(x.get("part_label", "")), str(x.get("method", ""))))

    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    write_csv(rows, out_csv)
    write_md(rows, out_md)

    print("Aggregate summary")
    print(f"- rows_written: {len(rows)}")
    if missing:
        print(f"- missing_inputs: {len(missing)}")
        for x in missing:
            print(f"  - {x}")
    print(f"- csv: {out_csv}")
    print(f"- md: {out_md}")


if __name__ == "__main__":
    main()
