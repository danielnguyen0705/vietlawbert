# VBPL controlled full-text pilot

- Technical retrieval decision: **READY_FOR_LARGER_PILOT**
- Corpus readiness decision: **REVIEW_REQUIRED**
- Completed: **120/120**
- Stopped early: **False**
- Sampling: central/local + identifier family + observed document group

## Technical quality gates

| Gate | Passed |
|---|---:|
| `completed_requested_sample` | True |
| `hard_failure_ratio_lte_2pct` | True |
| `content_or_pdf_ratio_gte_95pct` | True |
| `missing_official_group_ratio_lte_5pct` | True |
| `duplicate_content_ratio_lte_1pct` | True |
| `classification_review_ratio_lte_20pct` | True |

## Technical metrics

| Metric | Value |
|---|---:|
| `records` | 120 |
| `hard_failures` | 0 |
| `hard_failure_ratio` | 0.0 |
| `content_or_pdf` | 120 |
| `content_or_pdf_ratio` | 1.0 |
| `missing_official_group` | 2 |
| `missing_official_group_ratio` | 0.0167 |
| `duplicate_content_records` | 0 |
| `duplicate_content_ratio` | 0.0 |
| `classification_review_records` | 10 |
| `classification_review_ratio` | 0.0833 |
| `corpus_ready_records` | 114 |
| `corpus_ready_ratio` | 0.95 |
| `corpus_review_required_records` | 3 |
| `corpus_review_required_ratio` | 0.025 |
| `recovered_quality_review_records` | 3 |
| `recovered_quality_review_ratio` | 0.025 |

## Corpus readiness gates

| Gate | Passed |
|---|---:|
| `completed_requested_sample` | True |
| `content_or_pdf_ratio_gte_95pct` | True |
| `corpus_review_required_records_eq_0` | False |
| `recovered_quality_review_records_eq_0` | False |

## Corpus readiness metrics

| Metric | Value |
|---|---:|
| `records` | 120 |
| `content_or_pdf` | 120 |
| `content_or_pdf_ratio` | 1.0 |
| `corpus_ready_records` | 114 |
| `corpus_ready_ratio` | 0.95 |
| `corpus_review_required_records` | 3 |
| `corpus_review_required_ratio` | 0.025 |
| `recovered_quality_review_records` | 3 |
| `recovered_quality_review_ratio` | 0.025 |

## PDF/OCR fallback

| Metric | Value |
|---|---:|
| `recovered_from_pdf` | 3 |
| `recovered_by_ocr` | 2 |
| `pdf_fetch_failed` | 0 |
| `pdf_extraction_insufficient` | 2 |

## Status

| Value | Count |
|---|---:|
| CONTENT_TOO_SHORT | 2 |
| HTML_VALID | 117 |
| PDF_ONLY | 1 |

## Scope

| Value | Count |
|---|---:|
| central | 41 |
| local | 79 |

## Identifier type

| Value | Count |
|---|---:|
| legacy_prefixed | 10 |
| numeric | 105 |
| uuid | 5 |

## Observed group

| Value | Count |
|---|---:|
| CONSOLIDATED | 2 |
| LEGAL_FORM_CANDIDATE | 108 |
| SYSTEMATIZED | 3 |
| TRANSLATION | 7 |

## Official group

| Value | Count |
|---|---:|
| BD | 7 |
| UNKNOWN | 2 |
| VBHN | 2 |
| VBHTH | 3 |
| VBQPPL | 106 |

## Warnings

| Value | Count |
|---|---:|
| CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED | 2 |
| CONTENT_RECOVERED_FROM_PDF | 3 |
| DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED | 8 |
| MISSING_OFFICIAL_PARENT_GROUP | 2 |
| OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED | 1 |
| PDF_EXTRACTION_INSUFFICIENT | 2 |
| RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED | 1 |

## Limits

- This pilot is stratified and deterministic, not a simple random sample.
- Observed groups come from JSON-LD; official groups/forms come from VBPL detail metadata.
- PDF availability is not proof that PDF text extraction or OCR will succeed.
- Decision-number warnings are triage heuristics, not legal conclusions.
- A larger crawl still needs rate limiting, checkpointing and periodic contract verification.

## Recommended next step

- Keep Phase 2 open: review and repair recovered records flagged for OCR or encoding quality.
- Do not treat recovered HTML_VALID records as corpus-ready until corpus review warnings return to zero.
- Rebuild the pilot report after fixes, then re-evaluate Phase 3 entry criteria.
- Do not start a 500–1,000 document pilot or full-sitemap crawl from this report alone.
