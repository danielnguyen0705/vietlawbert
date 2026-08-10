# VBPL structural probe

Generated: `2026-08-10T15:38:40.495789+00:00`

## Verified facts

- robots.txt allows: `/`
- robots.txt disallows: `/api/, /Pages/`
- Sitemap files inspected: **36**
- Public document URLs discovered: **171536**
- Duplicate URLs: **0**
- Sample pages inspected: **18**; successful: **18**

## Inventory by scope

| Scope | Count |
|---|---:|
| central | 56834 |
| local | 114702 |

## Identifier formats

| ID type | Count |
|---|---:|
| legacy_prefixed | 11887 |
| numeric | 155546 |
| uuid | 4103 |

## JSON-LD field coverage

| Field | Present | Coverage |
|---|---:|---:|
| name | 13/18 | 72.2% |
| legislationIdentifier | 16/18 | 88.9% |
| legislationType | 18/18 | 100.0% |
| legislationDate | 18/18 | 100.0% |
| legislationLegalForce | 18/18 | 100.0% |
| legislationPassedBy | 18/18 | 100.0% |
| legislationJurisdiction | 18/18 | 100.0% |
| url | 18/18 | 100.0% |
| inLanguage | 18/18 | 100.0% |
| keywords | 18/18 | 100.0% |

## Sample page structure

- Pages with Legislation JSON-LD: **18/18**
- Pages with server-rendered `prov-content`: **0/18**
- Pages containing a `documentContent` application marker: **18/18**

## Design implications

- Use the advertised sitemap for discovery, not an agency-filter enumeration.
- Preserve numeric, UUID, and legacy-prefixed identifiers as strings.
- Use Legislation JSON-LD as the stable public metadata layer.
- Treat full legal text as a separate extraction layer; it is not guaranteed to be server-rendered.
- Preserve raw agency names and normalize them only after considering the document date.
- Do not call paths disallowed by robots.txt in this probe.
