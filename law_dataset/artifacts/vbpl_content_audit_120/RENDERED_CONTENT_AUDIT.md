# VBPL rendered-content audit

Verification date: 2026-08-10 (Asia/Ho_Chi_Minh)

This second audit opened ten stratified records in a real rendered browser after
the 120-page public-metadata audit. It measured the active `Nội dung` panel after
hydration.

## Results

| Record | Intended group | Result | Active content characters | Paragraphs | `prov-content` nodes | Article-like lines | Tables |
|---|---|---|---:|---:|---:|---:|---:|
| `121624` | Consolidated | Security check page | 0 | 0 | 0 | 0 | 0 |
| `138353` | Consolidated | Navigation aborted | — | — | — | — | — |
| `vbpqta_3750` | Translation | Rendered | 1,331 | 13 | 0 | 2 | 0 |
| `vbpqta_1964` | Translation | Rendered | 679 | 2 | 0 | 0 | 0 |
| `hhtquyetdinh_416` | Systematized | Rendered | 3,161 | 38 | 0 | 3 | 0 |
| `hhtquyetdinh_228` | Systematized | Rendered | 80,839 | 1,615 | 0 | 4 | 1 |
| `42125` | Decision, possibly normative | Rendered | 8,904 | 65 | 0 | 8 | 1 |
| `90831` | Individual administrative decision | Rendered | 5,553 | 96 | 79 | 3 | 3 |
| `3dedecb0-9239-11f1-889f-513c5fa29d01` | New central decree | Rendered | 52,577 | 369 | 16 | 42 | 2 |
| `83c7c0b0-78eb-11f1-ad24-f9eab024fad3` | New local decision | Rendered | 6,638 | 118 | 0 | 5 | 4 |

All eight successfully rendered records exposed the tabs `Nội dung`, `Thuộc tính`,
`Lược đồ`, `Văn bản gốc`, and `Tải về`.

## What was disproved

1. A sitemap URL is not guaranteed to render successfully on every attempt.
2. `.prov-content` is not a universal selector: six valid rendered samples had zero
   such nodes.
3. A long body does not prove that a document belongs in the normative corpus. The
   systematized record `hhtquyetdinh_228` was the longest sample.
4. The form `Quyết định` does not prove normative status. Record `90831` is an
   individual planning approval but still has structured paragraphs and article-like
   lines.
5. Article counts alone cannot classify legal status or document group.

## Safe classification consequence

Public metadata can identify legal form, but not always legal group. The audit now
uses `LEGAL_FORM_CANDIDATE`, not `NORMATIVE_CANDIDATE`. These records remain
`GROUP_AND_CONTENT_CHECK_REQUIRED` until a verified source field or reviewed rule
confirms their group and the rendered body or attachment parses successfully.

## Remaining requirement

A production crawler needs a content adapter with retries and explicit outcomes:

```text
RENDERED_HTML_VALID
PDF_ONLY
METADATA_ONLY
SECURITY_CHALLENGE
NAVIGATION_ERROR
CONTENT_TOO_SHORT
STRUCTURE_PARSE_FAILED
```

The two unsuccessful rendered samples should be retried later according to policy;
they must not be silently deleted or counted as empty legal documents.
