from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_drug_annotations(drugbank_path: Path | None, ctd_path: Path | None) -> dict[str, dict]:
    records: dict[str, dict] = {}

    def _ensure_record(drug_id: str) -> dict:
        key = drug_id.upper()
        if key not in records:
            records[key] = {
                "preferred_name": "",
                "indications": set(),
                "contraindications": set(),
                "sources": [],
            }
        return records[key]

    def _append_source(entry: dict, label: str, url: str | None) -> None:
        if not label:
            return
        for source in entry["sources"]:
            if source["label"] == label and source.get("url") == url:
                return
        entry["sources"].append({"label": label, "url": url})

    if drugbank_path and drugbank_path.exists():
        with drugbank_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                drug_id = (row.get("drug_id") or "").strip()
                if not drug_id:
                    continue
                entry = _ensure_record(drug_id)
                preferred = row.get("preferred_name") or ""
                if preferred and not entry["preferred_name"]:
                    entry["preferred_name"] = preferred
                indication = (row.get("indication") or "").strip()
                contraindication = (row.get("contraindication") or "").strip()
                if indication:
                    entry["indications"].add(indication)
                if contraindication:
                    entry["contraindications"].add(contraindication)
                _append_source(entry, "DrugBank", row.get("source_url"))

    if ctd_path and ctd_path.exists():
        with ctd_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                drug_id = (row.get("drug_id") or "").strip()
                if not drug_id:
                    continue
                entry = _ensure_record(drug_id)
                preferred = row.get("preferred_name") or ""
                if preferred and not entry["preferred_name"]:
                    entry["preferred_name"] = preferred
                indication = (row.get("indication") or "").strip()
                contraindication = (row.get("contraindication") or "").strip()
                if indication:
                    entry["indications"].add(indication)
                if contraindication:
                    entry["contraindications"].add(contraindication)
                _append_source(entry, "CTD", row.get("source_url"))

    normalized: dict[str, dict] = {}
    for key, entry in records.items():
        normalized[key] = {
            "preferred_name": entry["preferred_name"],
            "indications": sorted(entry["indications"]),
            "contraindications": sorted(entry["contraindications"]),
            "sources": entry["sources"],
        }
    return normalized


def load_pathways(pathway_path: Path | None) -> dict[str, list[dict]]:
    if not pathway_path or not pathway_path.exists():
        return {}
    mapping: dict[str, list[dict]] = defaultdict(list)
    with pathway_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gene = (row.get("gene_symbol") or "").strip()
            pathway = (row.get("pathway_name") or "").strip()
            if not gene or not pathway:
                continue
            entry = {
                "name": pathway,
                "source": (row.get("source") or "").strip() or None,
                "url": (row.get("pathway_url") or "").strip() or None,
            }
            mapping[gene.upper()].append(entry)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate static drug annotation and pathway lookup tables."
    )
    parser.add_argument("--drugbank", type=Path, help="Path to DrugBank-derived CSV file")
    parser.add_argument("--ctd", type=Path, help="Path to CTD-derived CSV file")
    parser.add_argument(
        "--pathways",
        type=Path,
        help="Path to pathway CSV file (columns: gene_symbol,pathway_name,source,pathway_url)",
    )
    parser.add_argument(
        "--annotations-output",
        type=Path,
        default=Path("app/data/drug_annotations.json"),
        help="Where to write the consolidated drug annotations JSON",
    )
    parser.add_argument(
        "--pathways-output",
        type=Path,
        default=Path("app/data/pathways.json"),
        help="Where to write the consolidated pathway JSON",
    )
    args = parser.parse_args()

    annotations = load_drug_annotations(args.drugbank, args.ctd)
    args.annotations_output.parent.mkdir(parents=True, exist_ok=True)
    args.annotations_output.write_text(json.dumps(annotations, indent=2, sort_keys=True), encoding="utf-8")

    pathways = load_pathways(args.pathways)
    args.pathways_output.parent.mkdir(parents=True, exist_ok=True)
    args.pathways_output.write_text(json.dumps(pathways, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote drug annotations to {args.annotations_output}")
    print(f"Wrote pathways to {args.pathways_output}")


if __name__ == "__main__":
    main()
