# VBPL structural probe

Generated: `2026-08-13T01:58:15.657916+00:00`

## Verified facts

- robots.txt allows: `/`
- robots.txt disallows: `/api/, /Pages/`
- Sitemap files inspected: **36**
- Public document URLs discovered: **171598**
- Duplicate URLs: **1**
- Sample pages inspected: **0**; successful: **0**

## Inventory by scope

| Scope | Count |
|---|---:|
| central | 56862 |
| local | 114736 |

## Identifier formats

| ID type | Count |
|---|---:|
| legacy_prefixed | 11887 |
| numeric | 155521 |
| uuid | 4190 |

## JSON-LD field coverage

| Field | Present | Coverage |
|---|---:|---:|
| name | 0/0 | 0.0% |
| legislationIdentifier | 0/0 | 0.0% |
| legislationType | 0/0 | 0.0% |
| legislationDate | 0/0 | 0.0% |
| legislationLegalForce | 0/0 | 0.0% |
| legislationPassedBy | 0/0 | 0.0% |
| legislationJurisdiction | 0/0 | 0.0% |
| url | 0/0 | 0.0% |
| inLanguage | 0/0 | 0.0% |
| keywords | 0/0 | 0.0% |

## Sample page structure

- Pages with Legislation JSON-LD: **0/0**
- Pages with server-rendered `prov-content`: **0/0**
- Pages containing a `documentContent` application marker: **0/0**

## Design implications

- Use the advertised sitemap for discovery, not an agency-filter enumeration.
- Preserve numeric, UUID, and legacy-prefixed identifiers as strings.
- Use Legislation JSON-LD as the stable public metadata layer.
- Treat full legal text as a separate extraction layer; it is not guaranteed to be server-rendered.
- Preserve raw agency names and normalize them only after considering the document date.
- Do not call paths disallowed by robots.txt in this probe.
