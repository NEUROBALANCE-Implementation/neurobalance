# scripts/preprocess_pmc_vqa.py
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REQUIRED_KEYS = ("id", "image_path", "question", "answer", "domain")


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def validate_record(r: Dict[str, Any]) -> None:
    for k in REQUIRED_KEYS:
        if k not in r:
            raise ValueError(f"Missing key '{k}' in record: {r}")


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


def read_json_any(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    # JSONL
    if "\n" in txt and txt.lstrip().startswith("{") and txt.rstrip().endswith("}"):
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
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


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    suf = path.suffix.lower()
    if suf in (".json", ".jsonl"):
        return read_json_any(path)
    if suf == ".csv":
        return read_csv_any(path)
    raise ValueError(f"Unsupported format: {path}")


def str2bool(x: str) -> bool:
    return x.strip().lower() in ("1", "true", "yes", "y")


def pick_images_dir(raw_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    candidates = [
        raw_dir / "images",
        raw_dir / "Images",
        raw_dir / "figures",
        raw_dir / "Figures",
        raw_dir / "imgs",
        raw_dir / "img",
        raw_dir / "data" / "images",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.*")):
            return c
    return raw_dir


def normalize_image_path(images_dir: Path, img_ref: str) -> Path:
    p = Path(img_ref)
    if p.is_absolute() and p.exists():
        return p
    p2 = images_dir / img_ref
    if p2.exists():
        return p2
    return images_dir / Path(img_ref).name


def maybe_copy_image(src: Path, dst_images_dir: Path, copy_images: bool) -> Path:
    if not copy_images:
        return src
    ensure_dir(dst_images_dir)
    dst = dst_images_dir / src.name
    if not dst.exists() and src.exists():
        shutil.copy2(src, dst)
    return dst


def detect_split_files(raw_dir: Path) -> Dict[str, Path]:
    ann_files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".csv")]
    def score(p: Path) -> int:
        n = p.name.lower()
        s = 0
        if "pmc" in n:
            s += 2
        if "train" in n:
            s += 1
        if "val" in n or "valid" in n:
            s += 1
        if "test" in n:
            s += 1
        if "qa" in n or "annotation" in n or "questions" in n:
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


def extract_fields(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    img = None
    for k in ("image", "image_path", "img", "img_path", "figure", "figure_path", "file_name", "filename"):
        if k in row and row[k]:
            img = str(row[k])
            break

    q = None
    for k in ("question", "Question", "query", "Q", "prompt"):
        if k in row and row[k]:
            q = str(row[k])
            break

    a = None
    for k in ("answer", "Answer", "A", "ans", "label"):
        if k in row and row[k] is not None:
            a = str(row[k])
            break

    return img, q, a


def build_records(
    split: str,
    rows: List[Dict[str, Any]],
    images_dir: Path,
    out_images_dir: Path,
    copy_images: bool,
    domain: str,
    limit: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    n_take = len(rows) if limit <= 0 else min(limit, len(rows))
    for i in range(n_take):
        img_ref, q, a = extract_fields(rows[i])
        if q is None or a is None:
            continue

        img_path = ""
        if img_ref:
            src = normalize_image_path(images_dir, img_ref)
            if src.exists():
                src = maybe_copy_image(src, out_images_dir, copy_images)
            img_path = str(src)

        records.append(
            {
                "id": f"pmc_vqa_{split}_{i+1:06d}",
                "image_path": img_path,
                "question": q.strip(),
                "answer": a.strip(),
                "domain": domain,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess PMC-VQA into unified JSONL format.")
    p.add_argument("--raw_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--copy_images", type=str, default="false")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--domain", type=str, default="medical")  # or "biomed"
    # explicit overrides
    p.add_argument("--images_dir", type=str, default="")
    p.add_argument("--train_ann", type=str, default="")
    p.add_argument("--val_ann", type=str, default="")
    p.add_argument("--test_ann", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    copy_images = str2bool(args.copy_images)
    images_dir = pick_images_dir(raw_dir, args.images_dir or None)
    out_images_dir = out_dir / "images"

    split_files: Dict[str, Path] = {}
    if args.train_ann:
        split_files["train"] = Path(args.train_ann)
    if args.val_ann:
        split_files["val"] = Path(args.val_ann)
    if args.test_ann:
        split_files["test"] = Path(args.test_ann)

    if not split_files:
        split_files = detect_split_files(raw_dir)

    if not split_files:
        raise SystemExit(
            f"[PMC-VQA] Could not auto-detect annotation files in {raw_dir}.\n"
            f"Provide --train_ann/--val_ann/--test_ann explicitly."
        )

    print("[PMC-VQA] raw_dir:", raw_dir)
    print("[PMC-VQA] images_dir:", images_dir)
    print("[PMC-VQA] out_dir:", out_dir)
    print("[PMC-VQA] split_files:", {k: str(v) for k, v in split_files.items()})

    for split, ann_path in split_files.items():
        rows = load_annotations(ann_path)
        recs = build_records(
            split=split,
            rows=rows,
            images_dir=images_dir,
            out_images_dir=out_images_dir,
            copy_images=copy_images,
            domain=args.domain,
            limit=args.limit,
        )
        out_path = out_dir / f"{split}.jsonl"
        n = write_jsonl(recs, out_path)
        print(f"[PMC-VQA] wrote {n} records -> {out_path}")
        for s in recs[:2]:
            print("  sample:", s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
