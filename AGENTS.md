# AGENTS.md — VietLawBERT operating contract

## Mission

Build a trustworthy Vietnamese legal corpus and legal AI pipeline for VietLawBERT. Priority order: correctness, legal-data quality, reproducibility, observability, performance, corpus size. Never optimize document count at expense of corpus quality.

## Before modifying code

Read `README.md` if present, `AGENTS.md`, relevant files under `docs/`, `law_dataset/VBPL_CONTENT_ADAPTER.md`, `law_dataset/VBPL_VERIFICATION_FINDINGS.md`, and latest pilot report. Inspect existing implementation, tests, expected inputs/outputs, and `git status` before editing.

## Change discipline

Make smallest reasonable change. Preserve working behavior. Avoid unrelated refactoring, massive rewrites, silent schema changes, deleting data without explicit justification, secrets in source control, and hardcoded user-specific absolute paths.

## Data rules

Every accepted legal document must keep provenance. Preserve source URL and stable document ID when available. Rejected pages must be logged with explicit reason. Crawler errors must not silently become corpus documents. Malformed JSON/JSONL must fail validation.

## VBPL crawling rules

Use official VBPL sitemap for discovery. Preserve IDs as strings. Preserve source scope: central/local. Keep official document group and document form separate. Do not infer ministries, agencies, sectors, or legal categories from URL/free text. HTTP 200 is not enough: distinguish `HTML_VALID`, `CONTENT_TOO_SHORT`, `PDF_ONLY`, `METADATA_ONLY`, `SECURITY_CHALLENGE`, `HTTP_ERROR`, `CONTRACT_ERROR`, and `PARSE_ERROR`. Stop batches on security challenge, contract change, or parser incompatibility.

## Validation rules

After relevant changes run tests, harness validation, and safe pilot/report checks when applicable. If validation fails, do not declare success. Diagnose, fix, and rerun. Never claim unexecuted tests passed.

## Git rules

Work only on the active safe project branch unless user gives explicit approval. Never push directly to `main`, `master`, or `ubuntu`. Commits must be focused. Do not commit secrets, raw crawled legal text, checkpoints, large datasets, or generated garbage files. Push only when explicitly requested.

## Developer commands

```bash
python -m unittest discover -s law_dataset/tests -v
python -m unittest discover -s tests -v
python -m harness.checks.validate_pipeline --input harness/fixtures/sample_documents.jsonl
python -m py_compile law_dataset/src/crawler/*.py
git diff --check
```

PowerShell equivalents:

```powershell
python -m unittest discover -s law_dataset/tests -v
python -m unittest discover -s tests -v
python -m harness.checks.validate_pipeline --input harness/fixtures/sample_documents.jsonl
python -m py_compile law_dataset/src/crawler/*.py
git diff --check
```
