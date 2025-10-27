from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_DISEASE_PREFIXES = {"EFO", "MONDO", "ORPHANET"}
_DRUG_PREFIXES = {"CHEMBL", "CHEBI", "DRUGBANK"}
_GENE_PREFIXES = {"ENSG", "HGNC"}

_CHEMBL_PATTERN = re.compile(r"^CHEMBL\d+$")
_CHEBI_PATTERN = re.compile(r"^CHEBI[:_]\d+$", re.IGNORECASE)
_MONDO_PATTERN = re.compile(r"^MONDO[:_]\d+$", re.IGNORECASE)
_EFO_PATTERN = re.compile(r"^EFO[:_]\d+$", re.IGNORECASE)


def _normalize_prefix(identifier: str, prefixes: set[str]) -> Optional[str]:
    if not identifier:
        return None
    value = identifier.strip()
    if ":" in value:
        prefix, suffix = value.split(":", 1)
        prefix = prefix.upper()
        if suffix and prefix in prefixes:
            return f"{prefix}_{suffix}"
    upper = value.upper()
    for prefix in prefixes:
        if upper.startswith(f"{prefix}_"):
            return upper
    return None


def normalize_disease_id(identifier: str) -> Optional[str]:
    normalized = _normalize_prefix(identifier, _DISEASE_PREFIXES)
    if normalized:
        return normalized
    # Some APIs return MONDO IDs embedded in URLs.
    if "obo/mondo#" in identifier.lower():
        _, _, suffix = identifier.lower().partition("mondo#")
        if suffix:
            return f"MONDO_{suffix}".upper()
    return None


def normalize_drug_id(identifier: str) -> Optional[str]:
    normalized = _normalize_prefix(identifier, _DRUG_PREFIXES)
    if normalized:
        if _CHEBI_PATTERN.match(normalized):
            return normalized.replace(":", "_")
        return normalized

    if _CHEMBL_PATTERN.match(identifier.upper()):
        return identifier.upper()

    return None


def normalize_gene_id(identifier: str) -> Optional[str]:
    normalized = _normalize_prefix(identifier, _GENE_PREFIXES)
    if normalized:
        return normalized
    if identifier.upper().startswith("ENSG"):
        return identifier.upper()
    return None


@dataclass(frozen=True, slots=True)
class HarmonizedEntity:
    raw: str
    normalized: Optional[str]
    namespace: Optional[str]


def harmonize_identifier(identifier: str) -> HarmonizedEntity:
    for ns, normalizer in (
        ("disease", normalize_disease_id),
        ("drug", normalize_drug_id),
        ("gene", normalize_gene_id),
    ):
        normalized = normalizer(identifier)
        if normalized:
            return HarmonizedEntity(raw=identifier, normalized=normalized, namespace=ns)
    return HarmonizedEntity(raw=identifier, normalized=None, namespace=None)

