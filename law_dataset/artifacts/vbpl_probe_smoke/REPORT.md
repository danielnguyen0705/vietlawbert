# VBPL structural probe

Generated: `2026-08-10T15:32:34.717981+00:00`

## Verified facts

- robots.txt allows: `/`
- robots.txt disallows: `/api/, /Pages/`
- Sitemap files inspected: **3**
- Public document URLs discovered: **10000**
- Duplicate URLs: **0**
- Sample pages inspected: **6**; successful: **6**

## Inventory by scope

| Scope | Count |
|---|---:|
| central | 10000 |

## Identifier formats

| ID type | Count |
|---|---:|
| legacy_prefixed | 60 |
| numeric | 9018 |
| uuid | 922 |

## JSON-LD field coverage

| Field | Present | Coverage |
|---|---:|---:|
| name | 6/6 | 100.0% |
| legislationIdentifier | 6/6 | 100.0% |
| legislationType | 6/6 | 100.0% |
| legislationDate | 6/6 | 100.0% |
| legislationLegalForce | 6/6 | 100.0% |
| legislationPassedBy | 6/6 | 100.0% |
| legislationJurisdiction | 6/6 | 100.0% |
| url | 6/6 | 100.0% |
| inLanguage | 6/6 | 100.0% |
| keywords | 6/6 | 100.0% |

## Design implications

- Use the advertised sitemap for discovery, not an agency-filter enumeration.
- Preserve numeric, UUID, and legacy-prefixed identifiers as strings.
- Use Legislation JSON-LD as the stable public metadata layer.
- Treat full legal text as a separate extraction layer; it is not guaranteed to be server-rendered.
- Preserve raw agency names and normalize them only after considering the document date.
- Do not call paths disallowed by robots.txt in this probe.
