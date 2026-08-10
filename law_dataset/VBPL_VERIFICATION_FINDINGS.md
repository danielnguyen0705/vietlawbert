# VBPL verification findings

Verification date: 2026-08-10 (Asia/Ho_Chi_Minh)

## Outcome

The existing crawler should not use the ministry/agency filter or a hard-coded
Next.js action as its primary discovery mechanism. VBPL publishes a public sitemap
that separates central and local documents. The safest verified architecture is:

1. sitemap discovery;
2. public page and JSON-LD metadata extraction;
3. a separate, replaceable adapter for full legal content and attachments;
4. raw-first storage, validation, then agency normalization.

## Evidence gathered

The full probe inspected all 36 sitemap entries advertised by
`https://vbpl.vn/sitemap.xml`: one static sitemap, 12 central-document sitemaps,
and 23 local-document sitemaps.

| Measurement | Result |
|---|---:|
| Unique public document URLs | 171,536 |
| Central documents | 56,834 |
| Local documents | 114,702 |
| Duplicate URLs | 0 |
| Numeric IDs | 155,546 |
| UUID IDs | 4,103 |
| Legacy-prefixed IDs | 11,887 |

Legacy URLs have at least two shapes:

```text
/van-ban/chi-tiet/<slug>--<document-id>
/van-ban/chi-tiet/<document-id>
```

Examples of valid IDs include:

```text
175440
3dedecb0-9239-11f1-889f-513c5fa29d01
hhtquyetdinh_228
vbpqdinhchinh_75
vbpqta_6368
```

Therefore, IDs must be stored as strings and must not be restricted to integers or
UUIDs.

## Page sample

The probe selected 18 central/local pages across new, old, numeric, UUID, and legacy
identifier groups. All 18 returned HTTP 200 and all exposed a Schema.org
`Legislation` JSON-LD object.

| JSON-LD field | Coverage |
|---|---:|
| `legislationType` | 18/18 |
| `legislationDate` | 18/18 |
| `legislationLegalForce` | 18/18 |
| `legislationPassedBy` | 18/18 |
| `legislationIdentifier` | 16/18 |
| `name` | 13/18 |

The missing names/identifiers occurred mainly in legacy administrative-related,
consolidated, or unidentified records. These must remain valid records with explicit
missing-field warnings; they must not be silently dropped.

None of the 18 raw HTML responses contained server-rendered `prov-content`, although
all contained the application marker `documentContent`. The rendered browser does
show the full text after hydration. This proves that discovery and metadata can be
crawled from public pages, while full text requires a separate dynamic-content path.

## robots.txt boundary

At verification time, `https://vbpl.vn/robots.txt` contained:

```text
User-Agent: *
Allow: /
Disallow: /api/
Disallow: /Pages/
Sitemap: https://vbpl.vn/sitemap.xml
```

The supplied probe follows this boundary and does not call API paths. The separate
gateway hostname used by the frontend must be assessed independently before a
production crawl. The current frontend bundle identifies its API base as:

```text
https://vbpl-bientap-gateway.moj.gov.vn/api
```

It also constructs attachment URLs shaped like:

```text
/qtdc/public/doc/minio/buckets/vbpl/{document_id}/{file_name}/download
```

The repository's current detail and diagram routes are consistent with the rendered
application, but the list discovery route still depends on a hard-coded Next.js
action hash. A frontend rebuild can invalidate that hash.

## Required follow-up before production full-text crawling

The sitemap and metadata layers are now verified. Full-text extraction still needs a
fresh browser Network capture covering:

1. one numeric, one UUID, and one legacy-prefixed record;
2. the `Nội dung`, `Thuộc tính`, `Lược đồ`, `Văn bản gốc`, and `Tải về` tabs;
3. request URL, method, payload, minimum headers, response schema, and status;
4. replay from a new session without cookies and without Next.js-only headers;
5. a second replay after a frontend deployment or at a later time.

Do not start a whole-site full-text crawl until that adapter passes fixtures for all
three identifier families and handles records with metadata but no HTML body.

## Recommended production boundaries

- `SitemapDiscovery`: yields URL, scope, last modified time, raw document ID.
- `PublicMetadataExtractor`: reads canonical URL, meta tags, and Legislation JSON-LD.
- `ContentAdapter`: obtains HTML/PDF using a separately verified contract.
- `RawStore`: saves immutable source payloads before parsing.
- `LegalStructureParser`: parses chapters/articles/clauses with plain-text fallback.
- `AgencyNormalizer`: preserves `agency_raw` and maps it using document date.
- `ValidationQueue`: holds missing content, malformed metadata, and uncertain agencies.

The generated evidence is under `artifacts/vbpl_probe_full/`.

## Follow-up audit results

A second deterministic, stratified audit inspected 120 public pages drawn from the
171,536-URL inventory:

- 41 central and 79 local records;
- 105 numeric, 10 legacy-prefixed, and 5 UUID records;
- 120/120 returned HTTP 200 and exposed Legislation JSON-LD;
- 108 were legal-form candidates, 2 consolidated records, 3 systematized records,
  and 7 translations;
- no URL or legal-key duplicate occurred inside this sample;
- none of the 120 raw HTML responses server-rendered the legal body.

The label is deliberately `LEGAL_FORM_CANDIDATE`, not `NORMATIVE_CANDIDATE`.
Inspection found Decisions that are individual administrative approvals even though
their metadata form is `Quyết định`.

A rendered-browser follow-up opened ten representative records. Eight rendered
legal/document content, one returned a security-check page, and one navigation was
aborted. The eight rendered content panels ranged from 679 to 80,839 characters.
Only two used `.prov-content`, proving that this class cannot be a universal content
selector. Details are in
`artifacts/vbpl_content_audit_120/RENDERED_CONTENT_AUDIT.md`.

These audits do not prove that all 171,536 URLs are usable. They establish the
validation states and show why each record needs a group check, content outcome, and
parser result before entering the RAG corpus.
