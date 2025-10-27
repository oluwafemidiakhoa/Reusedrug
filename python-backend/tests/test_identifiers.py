from app.utils.identifiers import (
    harmonize_identifier,
    normalize_disease_id,
    normalize_drug_id,
    normalize_gene_id,
)


def test_normalize_disease_id():
    assert normalize_disease_id("EFO:0000400") == "EFO_0000400"
    assert normalize_disease_id("mondo:0003538") == "MONDO_0003538"
    assert normalize_disease_id("http://purl.obolibrary.org/obo/mondo#0003538") == "MONDO_0003538"
    assert normalize_disease_id("unknown") is None


def test_normalize_drug_id():
    assert normalize_drug_id("CHEMBL25") == "CHEMBL25"
    assert normalize_drug_id("CHEBI:1234") == "CHEBI_1234"
    assert normalize_drug_id("CHEMBL:120") == "CHEMBL_120"
    assert normalize_drug_id("foo") is None


def test_normalize_gene_id():
    assert normalize_gene_id("ENSG000001") == "ENSG000001"
    assert normalize_gene_id("hgnc:5") == "HGNC_5"
    assert normalize_gene_id("x") is None


def test_harmonize_identifier():
    result = harmonize_identifier("EFO:0000400")
    assert result.namespace == "disease"
    assert result.normalized == "EFO_0000400"

    result = harmonize_identifier("CHEMBL25")
    assert result.namespace == "drug"

