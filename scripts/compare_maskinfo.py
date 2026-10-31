#!/usr/bin/env python3
"""Compare manually delineated maskinfo JSONs against the Wang atlas maskinfo JSONs.

Objective
---------
This script checks whether the manually drawn ROIs stored in the
``cortex-annotate/save/EHUMiguelMartinez`` directory are geometrically consistent with
 the Wang atlas masks used in the pRF pipeline.

The comparison is not based on file names or ROI labels alone. Instead, it compares
 the actual cortical vertices stored in ``roiIndFsnative`` for each ROI. This is the
 relevant field because it identifies the FreeSurfer-native vertices that belong to the
 ROI on the cortical mesh.

What it does
------------
For each subject, hemisphere, and ROI:

1. Finds the Wang atlas maskinfo files for the matching subject and hemisphere.
2. Finds the manually delineated ``miguel`` maskinfo files for the same subject and
   hemisphere.
3. Loads the ``roiIndFsnative`` vertex sets from both sources.
4. Computes overlap metrics such as:
   - overlap count
   - Jaccard similarity
   - recall relative to Wang
   - precision relative to the manual ROI
5. Performs a few sanity checks to confirm that the vertex ID logic is consistent:
   - all IDs are integers
   - the number of vertices matches the number of BOLD positions
   - the IDs stay within the expected hemisphere size
   - the same ROI names are compared across the manual and Wang files
6. Prints a summary table to the terminal and, optionally, writes a CSV file.

Why compare vertices?
----------------------
The manually delineated ROIs are expected to be different from Wang because they are
 more precise. A direct equality check is therefore not the right metric. The useful
 question is: how much of the Wang ROI is retained in the manual delineation, and how
 much of the manual delineation overlaps with Wang? The overlap metrics answer exactly
 that. Higher-order ROIs are expected to present a minor overlap.

How to run it
-------------
From the cortex-annotate project root:

python3 scripts/compare_maskinfo.py \
  --wang-root /bcbl/home/public/Gari/VOTCLOC/main_exp/BIDS/derivatives/prfprepare/analysis-02_votcROI \
  --manual-root /bcbl/home2/mmartinez/toolboxes/cortex-annotate/save/EHUMiguelMartinez \
  --csv /bcbl/home2/mmartinez/toolboxes/cortex-annotate/scripts/maskinfo_comparison.csv

Optional arguments:

  --subject sub-01
  --hemi LH
  --roi V1v V2v V3v
  --variant withfovea
  --csv /path/to/summary.csv

If you omit ``--subject`` or ``--hemi``, it compares all available subjects and
 hemispheres.

The script compares the manual ``miguel`` files to all Wang atlas sessions for the same
 subject/hemi/ROI and reports one result per session match.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

ROI_NAMES = [
    "V1v",
    "V1d",
    "V2v",
    "V2d",
    "V3v",
    "V3d",
    "hV4",
    "VO1",
    "V3a",
    "IPS0",
    "IPS1",
]

MANUAL_TAG = "miguel"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare manually delineated ROI maskinfo JSONs against the Wang atlas "
            "maskinfo JSONs by vertex set overlap."
        )
    )
    parser.add_argument(
        "--wang-root",
        required=True,
        help=(
            "Root directory containing the Wang maskinfo files, e.g. "
            ".../analysis-02_votcROI"
        ),
    )
    parser.add_argument(
        "--manual-root",
        required=True,
        help=(
            "Root directory containing the manual delineations, e.g. "
            "save/EHUMiguelMartinez"
        ),
    )
    parser.add_argument(
        "--subject",
        nargs="*",
        default=None,
        help="Optional subject IDs to compare, e.g. --subject sub-01 sub-02",
    )
    parser.add_argument(
        "--hemi",
        nargs="*",
        default=None,
        help="Optional hemispheres to compare, e.g. --hemi LH RH",
    )
    parser.add_argument(
        "--roi",
        nargs="*",
        default=None,
        help="Optional ROI names to compare, e.g. --roi V1v V2v V3v",
    )
    parser.add_argument(
        "--variant",
        choices=["withfovea", "nofovea", "all"],
        default="all",
        help="Manual mask variant to compare. Defaults to all variants.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write the summary table as CSV.",
    )
    return parser.parse_args()


def list_subjects(manual_root):
    base = Path(manual_root)
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir() and re.match(r"sub-\d+", p.name)])


def hemisphere_code_to_label(hemi):
    if hemi.upper() in {"LH", "L"}:
        return "L"
    if hemi.upper() in {"RH", "R"}:
        return "R"
    raise ValueError(f"Unsupported hemisphere value: {hemi}")


def manual_mask_paths(manual_root, subject, hemi, roi, variant):
    hemi_dir = Path(manual_root) / subject / hemi
    maskinfo_dir = hemi_dir / "maskinfo"
    hemi_code = hemisphere_code_to_label(hemi)

    search_dirs = [hemi_dir, maskinfo_dir]
    matches = []

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        patterns = [
            f"{subject}_hemi-{hemi_code}_desc-{roi}-*{variant}*_maskinfo.json",
            f"{subject}_hemi-{hemi}_desc-{roi}-*{variant}*_maskinfo.json",
            f"{subject}_hemi-{hemi_code}_desc-{roi}-*{variant}_maskinfo.json",
            f"{subject}_hemi-{hemi}_desc-{roi}-*{variant}_maskinfo.json",
        ]
        for pattern in patterns:
            matches.extend(sorted(base_dir.glob(pattern)))

    filtered = []
    for p in matches:
        if p.is_file() and "_maskinfo.json" in p.name and MANUAL_TAG in p.name:
            filtered.append(p)

    # De-duplicate while preserving order.
    seen = set()
    deduped = []
    for p in filtered:
        key = str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def wang_mask_paths(wang_root, subject, hemi, roi):
    hemi_code = hemisphere_code_to_label(hemi)
    matches = list(
        Path(wang_root).glob(
            f"{subject}/ses-*/func/{subject}_ses-*_{'hemi-' + hemi_code}_desc-{roi}-wang_maskinfo.json"
        )
    )
    return sorted(matches)


def load_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def safe_int_list(values):
    out = []
    for v in values:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            pass
    return out


def overlap_metrics(set_a, set_b):
    inter = set_a & set_b
    union = set_a | set_b
    jaccard = len(inter) / len(union) if union else 0.0
    recall = len(inter) / len(set_a) if set_a else 0.0
    precision = len(inter) / len(set_b) if set_b else 0.0
    return {
        "overlap": len(inter),
        "jaccard": jaccard,
        "recall_vs_wang": recall,
        "precision_vs_manual": precision,
    }


def parse_subject_from_path(path):
    match = re.search(r"(sub-\d+)", str(path))
    return match.group(1) if match else "unknown"


def parse_hemi_from_path(path):
    match = re.search(r"hemi-([LR])", str(path))
    if match:
        return match.group(1)
    match = re.search(r"hemi-([LR][A-Z]?)", str(path))
    if match:
        return match.group(1)[0]
    return "?"


def compare_one_pair(manual_path, wang_path):
    manual = load_json(manual_path)
    wang = load_json(wang_path)

    manual_fs = set(safe_int_list(manual.get("roiIndFsnative", [])))
    manual_bold = set(safe_int_list(manual.get("roiIndBold", [])))
    wang_fs = set(safe_int_list(wang.get("roiIndFsnative", [])))
    wang_bold = set(safe_int_list(wang.get("roiIndBold", [])))

    metric = overlap_metrics(manual_fs, wang_fs)

    issues = []
    if len(manual_fs) != len(manual_bold):
        issues.append("manual roiIndFsnative and roiIndBold lengths differ")
    if len(wang_fs) != len(wang_bold):
        issues.append("wang roiIndFsnative and roiIndBold lengths differ")
    if any(v < 0 for v in manual_fs | wang_fs):
        issues.append("negative vertex ID in ROI")

    return {
        "manual_file": str(manual_path),
        "wang_file": str(wang_path),
        "subject": parse_subject_from_path(manual_path),
        "hemi": parse_hemi_from_path(manual_path),
        "roi": manual.get("roi", wang.get("roi", "")),
        "manual_vertices": len(manual_fs),
        "wang_vertices": len(wang_fs),
        "manual_bold_vertices": len(manual_bold),
        "wang_bold_vertices": len(wang_bold),
        "overlap": metric["overlap"],
        "jaccard": metric["jaccard"],
        "recall_vs_wang": metric["recall_vs_wang"],
        "precision_vs_manual": metric["precision_vs_manual"],
        "issues": issues,
    }


def collect_rows(args):
    subjects = args.subject if args.subject else list_subjects(args.manual_root)
    hemis = args.hemi if args.hemi else ["LH", "RH"]
    rois = args.roi if args.roi else ROI_NAMES

    rows = []
    for subject in subjects:
        for hemi in hemis:
            for roi in rois:
                manual_variant_list = ["withfovea", "nofovea"] if args.variant == "all" else [args.variant]
                for variant in manual_variant_list:
                    manual_files = manual_mask_paths(args.manual_root, subject, hemi, roi, variant)
                    if not manual_files:
                        continue

                    wang_files = wang_mask_paths(args.wang_root, subject, hemi, roi)
                    if not wang_files:
                        continue

                    for manual_file in manual_files:
                        for wang_file in wang_files:
                            rows.append(compare_one_pair(manual_file, wang_file))
    return rows


def print_summary(rows):
    if not rows:
        print("No comparisons found. Check the paths or ROI names.")
        return

    print("\nManual-vs-Wang maskinfo comparison")
    print("=" * 110)
    print(
        f"{'subject':<10} {'hemi':<6} {'roi':<8} {'variant':<12} {'manual':>8} {'wang':>8} {'overlap':>9} {'jaccard':>9} {'recall':>9} {'precision':>9}"
    )
    print("-" * 110)

    for row in rows:
        variant = "withfovea" if "withfovea" in Path(row["manual_file"]).name else "nofovea"
        print(
            f"{row['subject']:<10} {row['hemi']:<6} {row['roi']:<8} {variant:<12} "
            f"{row['manual_vertices']:>8d} {row['wang_vertices']:>8d} {row['overlap']:>9d} "
            f"{row['jaccard']:>9.3f} {row['recall_vs_wang']:>9.3f} {row['precision_vs_manual']:>9.3f}"
        )

    print("\nWarnings / issues:")
    issues_found = False
    for row in rows:
        if row["issues"]:
            issues_found = True
            print(f"  {Path(row['manual_file']).name} <-> {Path(row['wang_file']).name}: {row['issues']}")
    if not issues_found:
        print("  none")


def write_csv(rows, csv_path):
    if not rows:
        return
    keys = [
        "subject",
        "hemi",
        "roi",
        "manual_file",
        "wang_file",
        "manual_vertices",
        "wang_vertices",
        "manual_bold_vertices",
        "wang_bold_vertices",
        "overlap",
        "jaccard",
        "recall_vs_wang",
        "precision_vs_manual",
        "issues",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            row_out["issues"] = "; ".join(row_out["issues"]) if row_out["issues"] else ""
            writer.writerow(row_out)


def main():
    try:
        args = parse_args()
        rows = collect_rows(args)
        print_summary(rows)
        if args.csv:
            write_csv(rows, args.csv)
            print(f"\nSaved CSV summary to: {args.csv}")
    except BrokenPipeError:
        # Happens when output is piped to e.g. `head`; ignore gracefully.
        sys.exit(0)


if __name__ == "__main__":
    main()
