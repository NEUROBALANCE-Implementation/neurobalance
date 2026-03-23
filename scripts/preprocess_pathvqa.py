# scripts/preprocess_pathvqa.py
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -------------------------
# Shared utilities
# -------------------------
REQUIRED_KEYS = ("id", "image_path", "question", "answer", "domain")


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def write_jsonl(records: Iterable[Dict[str, Any]], out_path: str | Path) -> int:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            validate_record(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def validate_record(r: Dict[str, Any]) -> None:
    for k in REQUIRED_KEYS:
        if k not in r:
            raise ValueError(f"Missing key '{k}' in record: {r}")
    if not isinstance(r["question"], str) or not isinstance(r["answer"], str):
        raise ValueError(f"question/answer must be strings: {r}")


def read_json_any(path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
      - JSON list of dicts
      - JSON dict with a list under common keys
      - JSONL (one dict per line)
    """
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    # JSONL heuristic
    if "\n" in txt and txt.lstrip().startswith("{") and txt.rstrip().endswith("}"):
        rows: List[Dict[str, Any]] = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    obj = json.loads(txt)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "annotations", "qa_pairs", "questions", "samples", "items"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    raise ValueError(f"Unrecognized JSON structure in {path}")


def read_csv_any(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def pick_images_dir(raw_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    # Common candidates
    candidates = [
        raw_dir / "images",
        raw_dir / "Images",
        raw_dir / "img",
        raw_dir / "imgs",
        raw_dir / "data" / "images",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.*")):
            return c
    # fallback: raw_dir itself if it contains images
    if any(raw_dir.glob("*.png")) or any(raw_dir.glob("*.jpg")) or any(raw_dir.glob("*.jpeg")):
        return raw_dir
    return raw_dir  # still return; later validation will warn


def normalize_image_path(images_dir: Path, img_ref: str) -> Path:
    """
    img_ref can be:
      - filename
      - relative path
      - absolute path
    """
    p = Path(img_ref)
    if p.is_absolute() and p.exists():
        return p
    # try relative to images_dir
    p2 = images_dir / img_ref
    if p2.exists():
        return p2
    # try by basename
    p3 = images_dir / Path(img_ref).name
    if p3.exists():
        return p3
    return p2  # best guess


def maybe_copy_image(src: Path, dst_images_dir: Path, copy_images: bool) -> Path:
    if not copy_images:
        return src
    ensure_dir(dst_images_dir)
    dst = dst_images_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def find_split_files(raw_dir: Path) -> Dict[str, Path]:
    """
    Auto-detect train/val/test annotation files by filename patterns.
    Supports json/jsonl/csv.
    """
    all_files = list(raw_dir.rglob("*"))
    ann_files = [p for p in all_files if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".csv")]
    def score(p: Path) -> int:
        name = p.name.lower()
        s = 0
        if "pathvqa" in name:
            s += 2
        if "train" in name:
            s += 1
        if "val" in name or "valid" in name:
            s += 1
        if "test" in name:
            s += 1
        if "qa" in name or "annotation" in name or "questions" in name:
            s += 1
        return s

    ann_files.sort(key=score, reverse=True)

    out: Dict[str, Path] = {}
    for p in ann_files:
        n = p.name.lower()
        if "train" in n and "train" not in out:
            out["train"] = p
        elif ("val" in n or "valid" in n) and "val" not in out:
            out["val"] = p
        elif "test" in n and "test" not in out:
            out["test"] = p
    return out


def extract_qa_fields(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Try common keys for image/question/answer.
    """
    # image
    img = None
    for k in ("image", "image_path", "img", "img_path", "image_name", "filename", "file_name", "pic"):
        if k in row and row[k]:
            img = str(row[k])
            break

    # question
    q = None
    for k in ("question", "Question", "query", "Q", "ques"):
        if k in row and row[k]:
            q = str(row[k])
            break

    # answer
    a = None
    for k in ("answer", "Answer", "A", "ans", "label"):
        if k in row and row[k] is not None:
            a = str(row[k])
            break

    return img, q, a


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() in (".json", ".jsonl"):
        return read_json_any(path)
    if path.suffix.lower() == ".csv":
        return read_csv_any(path)
    raise ValueError(f"Unsupported annotation format: {path}")


def build_records(
    split_name: str,
    ann_rows: List[Dict[str, Any]],
    images_dir: Path,
    out_images_dir: Path,
    copy_images: bool,
    domain: str,
    limit: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    n_take = len(ann_rows) if limit <= 0 else min(limit, len(ann_rows))
    for idx in range(n_take):
        row = ann_rows[idx]
        img_ref, q, a = extract_qa_fields(row)
        if q is None or a is None:
            continue

        img_path = ""
        if img_ref is not None:
            src = normalize_image_path(images_dir, img_ref)
            if src.exists():
                dst = maybe_copy_image(src, out_images_dir, copy_images)
                img_path = str(dst)
            else:
                # keep best guess; training loader can warn later
                img_path = str(src)

        rec_id = f"pathvqa_{split_name}_{idx+1:06d}"
        records.append(
            {
                "id": rec_id,
                "image_path": img_path,
                "question": q.strip(),
                "answer": a.strip(),
                "domain": domain,
            }
        )
    return records


# -------------------------
# Main script
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess PathVQA into unified JSONL format.")
    p.add_argument("--raw_dir", type=str, required=True, help="Raw dataset root directory.")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for unified dataset.")
    p.add_argument("--copy_images", type=str, default="false", help="true/false. Copy images into out_dir/images/.")
    p.add_argument("--limit", type=int, default=0, help="Optional limit per split for debugging.")
    p.add_argument("--domain", type=str, default="pathology", help="Domain label saved into JSONL.")
    # Optional explicit paths
    p.add_argument("--images_dir", type=str, default="", help="Explicit images directory (if auto-detect fails).")
    p.add_argument("--train_ann", type=str, default="", help="Explicit train annotation file.")
    p.add_argument("--val_ann", type=str, default="", help="Explicit val annotation file.")
    p.add_argument("--test_ann", type=str, default="", help="Explicit test annotation file.")
    return p.parse_args()


def str2bool(x: str) -> bool:
    return x.strip().lower() in ("1", "true", "yes", "y")


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    copy_images = str2bool(args.copy_images)

    ensure_dir(out_dir)
    dataset_dir = out_dir  # already passed as dataset-specific out_dir per your layout
    out_images_dir = dataset_dir / "images"

    images_dir = pick_images_dir(raw_dir, args.images_dir or None)

    # Determine annotation files
    split_files = {}
    if args.train_ann:
        split_files["train"] = Path(args.train_ann)
    if args.val_ann:
        split_files["val"] = Path(args.val_ann)
    if args.test_ann:
        split_files["test"] = Path(args.test_ann)

    if not split_files:
        split_files = find_split_files(raw_dir)

    if not split_files:
        raise SystemExit(
            f"[PathVQA] Could not auto-detect annotation files in {raw_dir}.\n"
            f"Provide --train_ann/--val_ann/--test_ann explicitly."
        )

    print("[PathVQA] raw_dir:", raw_dir)
    print("[PathVQA] images_dir:", images_dir)
    print("[PathVQA] out_dir:", out_dir)
    print("[PathVQA] split_files:", {k: str(v) for k, v in split_files.items()})

    for split_name, ann_path in split_files.items():
        rows = load_annotations(ann_path)
        recs = build_records(
            split_name=split_name,
            ann_rows=rows,
            images_dir=images_dir,
            out_images_dir=out_images_dir,
            copy_images=copy_images,
            domain=args.domain,
            limit=args.limit,
        )
        out_path = dataset_dir / f"{split_name}.jsonl"
        n = write_jsonl(recs, out_path)
        print(f"[PathVQA] wrote {n} records -> {out_path}")

        # Print a couple sample lines
        for s in recs[:2]:
            print("  sample:", s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
