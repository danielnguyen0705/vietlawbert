#!/usr/bin/env python3
"""Read-only structural probe for the public VBPL website.

This tool deliberately separates document discovery from document-content APIs:

* Discovery uses the sitemap advertised by https://vbpl.vn/robots.txt.
* Core metadata comes from public document pages and their Legislation JSON-LD.
* It does not call paths disallowed by VBPL's robots.txt.

The output is evidence for designing a crawler; it is not the production crawler.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
import urllib.error
import urllib.request
from urllib.parse import unquote, urlparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


BASE_URL = "https://vbpl.vn"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
DEFAULT_USER_AGENT = "VietLawBERT-Research-Probe/0.1 (+read-only; respectful-rate)"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SitemapRef:
    url: str
    scope: str


@dataclass(frozen=True)
class DocumentRef:
    url: str
    lastmod: str | None
    scope: str
    sitemap_url: str
    document_id: str
    id_type: str


class PageMetadataParser(HTMLParser):
    """Extract title, selected meta tags, and JSON-LD without third-party packages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.meta: dict[str, str] = {}
        self.canonical: str | None = None
        self.json_ld: list[Any] = []
        self.json_ld_errors = 0
        self._in_title = False
        self._in_json_ld = False
        self._script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = values.get("name") or values.get("property")
            if key and "content" in values:
                self.meta[key] = values["content"]
        elif tag == "link" and values.get("rel", "").lower() == "canonical":
            self.canonical = values.get("href") or None
        elif tag == "script" and values.get("type", "").lower() == "application/ld+json":
            self._in_json_ld = True
            self._script_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        elif tag == "script" and self._in_json_ld:
            payload = "".join(self._script_parts).strip()
            if payload:
                try:
                    self.json_ld.append(json.loads(payload))
                except json.JSONDecodeError:
                    self.json_ld_errors += 1
            self._in_json_ld = False
            self._script_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._in_json_ld:
            self._script_parts.append(data)

    @property
    def title(self) -> str:
        return "".join(self.title_parts).strip()


class PoliteFetcher:
    def __init__(self, user_agent: str, delay: float, timeout: float, retries: int) -> None:
        self.user_agent = user_agent
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.retries = max(0, retries)
        self._last_request_at = 0.0

    def fetch(self, url: str, accept: str = "*/*") -> tuple[bytes, dict[str, str], int]:
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent, "Accept": accept},
            )
            try:
                self._last_request_at = time.monotonic()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    headers = {key.lower(): value for key, value in response.headers.items()}
                    return body, headers, response.status
            except (urllib.error.URLError, TimeoutError) as exc:
                error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
        assert error is not None
        raise error


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_robots(text: str) -> dict[str, Any]:
    allow: list[str] = []
    disallow: list[str] = []
    sitemaps: list[str] = []
    current_applies = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        key_lower = key.lower()
        if key_lower == "user-agent":
            current_applies = value == "*"
        elif key_lower == "sitemap":
            sitemaps.append(value)
        elif current_applies and key_lower == "allow":
            allow.append(value)
        elif current_applies and key_lower == "disallow":
            disallow.append(value)
    return {"allow": allow, "disallow": disallow, "sitemaps": sitemaps}


def parse_sitemap_index(xml_bytes: bytes) -> list[SitemapRef]:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(xml_bytes, parser=parser)
    scope = "unclassified"
    refs: list[SitemapRef] = []
    for child in root:
        if child.tag is ET.Comment:
            comment = (child.text or "").strip().lower()
            if "trung ương" in comment:
                scope = "central"
            elif "địa phương" in comment:
                scope = "local"
            elif "tĩnh" in comment or "trang" in comment:
                scope = "static"
            continue
        if local_name(str(child.tag)) != "sitemap":
            continue
        location = next(
            ((node.text or "").strip() for node in child if local_name(str(node.tag)) == "loc"),
            "",
        )
        if location:
            refs.append(SitemapRef(location, scope))
    return refs


def extract_document_id(url: str) -> str:
    path_segment = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1])
    return path_segment.rsplit("--", 1)[-1]


def classify_document_id(document_id: str) -> str:
    if document_id.isdigit():
        return "numeric"
    if UUID_RE.fullmatch(document_id):
        return "uuid"
    if re.fullmatch(r"[A-Za-z0-9_-]+", document_id):
        return "legacy_prefixed"
    return "other"


def parse_urlset(xml_bytes: bytes, sitemap: SitemapRef) -> list[DocumentRef]:
    root = ET.fromstring(xml_bytes)
    documents: list[DocumentRef] = []
    for url_node in root:
        if local_name(str(url_node.tag)) != "url":
            continue
        values = {
            local_name(str(node.tag)): (node.text or "").strip()
            for node in url_node
        }
        url = values.get("loc", "")
        if "/van-ban/chi-tiet/" not in url:
            continue
        document_id = extract_document_id(url)
        documents.append(
            DocumentRef(
                url=url,
                lastmod=values.get("lastmod") or None,
                scope=sitemap.scope,
                sitemap_url=sitemap.url,
                document_id=document_id,
                id_type=classify_document_id(document_id),
            )
        )
    return documents


def flatten_json_ld(items: Iterable[Any]) -> Iterable[dict[str, Any]]:
    for item in items:
        if isinstance(item, list):
            yield from flatten_json_ld(item)
        elif isinstance(item, dict):
            graph = item.get("@graph")
            if isinstance(graph, list):
                yield from flatten_json_ld(graph)
            yield item


def legislation_from(parser: PageMetadataParser) -> dict[str, Any] | None:
    for item in flatten_json_ld(parser.json_ld):
        value = item.get("@type")
        types = value if isinstance(value, list) else [value]
        if "Legislation" in types:
            return item
    return None


def evenly_spaced(items: list[DocumentRef], count: int) -> list[DocumentRef]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return items[:]
    if count == 1:
        return [items[len(items) // 2]]
    indexes = {round(i * (len(items) - 1) / (count - 1)) for i in range(count)}
    return [items[index] for index in sorted(indexes)]


def select_samples(documents: list[DocumentRef], sample_size: int) -> list[DocumentRef]:
    groups: dict[tuple[str, str], list[DocumentRef]] = defaultdict(list)
    for document in documents:
        groups[(document.scope, document.id_type)].append(document)
    if not groups or sample_size <= 0:
        return []
    per_group = max(1, math.ceil(sample_size / len(groups)))
    candidates: list[DocumentRef] = []
    for key in sorted(groups):
        candidates.extend(evenly_spaced(groups[key], per_group))
    if len(candidates) <= sample_size:
        return candidates
    return evenly_spaced(candidates, sample_size)


def inspect_page(fetcher: PoliteFetcher, document: DocumentRef) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "url": document.url,
        "document_id": document.document_id,
        "id_type": document.id_type,
        "scope": document.scope,
        "lastmod": document.lastmod,
    }
    try:
        body, headers, status = fetcher.fetch(document.url, "text/html,application/xhtml+xml")
        text = body.decode("utf-8", "replace")
        parser = PageMetadataParser()
        parser.feed(text)
        legislation = legislation_from(parser)
        result.update(
            {
                "ok": status == 200,
                "status": status,
                "content_type": headers.get("content-type"),
                "bytes": len(body),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "title": parser.title,
                "canonical": parser.canonical,
                "description": parser.meta.get("description"),
                "json_ld_blocks": len(parser.json_ld),
                "json_ld_errors": parser.json_ld_errors,
                "legislation": legislation,
                "has_legislation_json_ld": legislation is not None,
                "has_next_flight_data": "self.__next_f" in text,
                "has_ssr_provision_content": "prov-content" in text,
                "has_document_content_marker": "documentContent" in text,
            }
        )
    except Exception as exc:  # probe must report failures instead of aborting the run
        result.update(
            {
                "ok": False,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return result


def coverage(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fields = [
        "name",
        "legislationIdentifier",
        "legislationType",
        "legislationDate",
        "legislationLegalForce",
        "legislationPassedBy",
        "legislationJurisdiction",
        "url",
        "inLanguage",
        "keywords",
    ]
    successful = [sample for sample in samples if sample.get("ok")]
    denominator = len(successful)
    output: dict[str, dict[str, Any]] = {}
    for field in fields:
        present = sum(
            1
            for sample in successful
            if isinstance(sample.get("legislation"), dict)
            and sample["legislation"].get(field) not in (None, "", [], {})
        )
        output[field] = {
            "present": present,
            "successful_pages": denominator,
            "ratio": round(present / denominator, 4) if denominator else 0.0,
        }
    return output


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    inventory = report["inventory"]
    sample = report["sample"]
    lines = [
        "# VBPL structural probe",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Verified facts",
        "",
        f"- robots.txt allows: `{', '.join(report['robots']['parsed']['allow']) or '(none)'}`",
        f"- robots.txt disallows: `{', '.join(report['robots']['parsed']['disallow']) or '(none)'}`",
        f"- Sitemap files inspected: **{report['sitemaps']['inspected']}**",
        f"- Public document URLs discovered: **{inventory['total_urls']}**",
        f"- Duplicate URLs: **{inventory['duplicate_urls']}**",
        f"- Sample pages inspected: **{sample['attempted']}**; successful: **{sample['successful']}**",
        "",
        "## Inventory by scope",
        "",
        "| Scope | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in inventory["by_scope"].items())
    lines.extend(["", "## Identifier formats", "", "| ID type | Count |", "|---|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in inventory["by_id_type"].items())
    lines.extend(["", "## JSON-LD field coverage", "", "| Field | Present | Coverage |", "|---|---:|---:|"])
    for field, stats in sample["legislation_json_ld_coverage"].items():
        lines.append(f"| {field} | {stats['present']}/{stats['successful_pages']} | {stats['ratio']:.1%} |")
    lines.extend(
        [
            "",
            "## Sample page structure",
            "",
            f"- Pages with Legislation JSON-LD: **{sample['has_legislation_json_ld']}/{sample['attempted']}**",
            f"- Pages with server-rendered `prov-content`: **{sample['has_ssr_provision_content']}/{sample['attempted']}**",
            f"- Pages containing a `documentContent` application marker: **{sample['has_document_content_marker']}/{sample['attempted']}**",
            "",
            "## Design implications",
            "",
            "- Use the advertised sitemap for discovery, not an agency-filter enumeration.",
            "- Preserve numeric, UUID, and legacy-prefixed identifiers as strings.",
            "- Use Legislation JSON-LD as the stable public metadata layer.",
            "- Treat full legal text as a separate extraction layer; it is not guaranteed to be server-rendered.",
            "- Preserve raw agency names and normalize them only after considering the document date.",
            "- Do not call paths disallowed by robots.txt in this probe.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/vbpl_probe")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument(
        "--max-sitemaps",
        type=int,
        default=0,
        help="0 means inspect every sitemap advertised by the sitemap index.",
    )
    parser.add_argument("--delay", type=float, default=0.35, help="Minimum delay between requests.")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--skip-pages", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fetcher = PoliteFetcher(args.user_agent, args.delay, args.timeout, args.retries)

    robots_body, robots_headers, robots_status = fetcher.fetch(ROBOTS_URL, "text/plain")
    robots_text = robots_body.decode("utf-8", "replace")
    robots = parse_robots(robots_text)
    (output_dir / "robots.txt").write_text(robots_text, encoding="utf-8")

    sitemap_body, sitemap_headers, sitemap_status = fetcher.fetch(SITEMAP_URL, "application/xml,text/xml")
    sitemap_refs = parse_sitemap_index(sitemap_body)
    selected_sitemaps = sitemap_refs[: args.max_sitemaps] if args.max_sitemaps > 0 else sitemap_refs

    documents: list[DocumentRef] = []
    sitemap_errors: list[dict[str, str]] = []
    sitemap_counts: list[dict[str, Any]] = []
    for index, sitemap in enumerate(selected_sitemaps, start=1):
        print(f"[{index}/{len(selected_sitemaps)}] {sitemap.scope}: {sitemap.url}", flush=True)
        try:
            body, _, status = fetcher.fetch(sitemap.url, "application/xml,text/xml")
            parsed = parse_urlset(body, sitemap)
            documents.extend(parsed)
            sitemap_counts.append(
                {"url": sitemap.url, "scope": sitemap.scope, "status": status, "documents": len(parsed)}
            )
        except Exception as exc:
            sitemap_errors.append(
                {"url": sitemap.url, "scope": sitemap.scope, "error_type": type(exc).__name__, "error": str(exc)}
            )

    url_counts = Counter(document.url for document in documents)
    unique_documents = list({document.url: document for document in documents}.values())
    by_scope = Counter(document.scope for document in unique_documents)
    by_id_type = Counter(document.id_type for document in unique_documents)
    by_scope_and_id = Counter(f"{document.scope}:{document.id_type}" for document in unique_documents)

    inventory_path = output_dir / "sitemap_inventory.jsonl.gz"
    with gzip.open(inventory_path, "wt", encoding="utf-8") as stream:
        for document in unique_documents:
            stream.write(json.dumps(document.__dict__, ensure_ascii=False) + "\n")

    selected_samples = [] if args.skip_pages else select_samples(unique_documents, args.sample_size)
    sample_results: list[dict[str, Any]] = []
    for index, document in enumerate(selected_samples, start=1):
        print(f"[sample {index}/{len(selected_samples)}] {document.scope}/{document.id_type}: {document.url}", flush=True)
        sample_results.append(inspect_page(fetcher, document))

    with (output_dir / "sample_pages.jsonl").open("w", encoding="utf-8") as stream:
        for sample in sample_results:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probe_policy": {
            "read_only": True,
            "respects_robots": True,
            "gateway_api_called": False,
            "minimum_delay_seconds": args.delay,
            "user_agent": args.user_agent,
        },
        "robots": {
            "url": ROBOTS_URL,
            "status": robots_status,
            "content_type": robots_headers.get("content-type"),
            "parsed": robots,
        },
        "sitemaps": {
            "index_url": SITEMAP_URL,
            "index_status": sitemap_status,
            "index_content_type": sitemap_headers.get("content-type"),
            "advertised": len(sitemap_refs),
            "inspected": len(selected_sitemaps),
            "counts": sitemap_counts,
            "errors": sitemap_errors,
        },
        "inventory": {
            "total_rows": len(documents),
            "total_urls": len(unique_documents),
            "duplicate_urls": sum(count - 1 for count in url_counts.values() if count > 1),
            "by_scope": dict(sorted(by_scope.items())),
            "by_id_type": dict(sorted(by_id_type.items())),
            "by_scope_and_id_type": dict(sorted(by_scope_and_id.items())),
        },
        "sample": {
            "attempted": len(sample_results),
            "successful": sum(bool(sample.get("ok")) for sample in sample_results),
            "failed": sum(not bool(sample.get("ok")) for sample in sample_results),
            "has_legislation_json_ld": sum(bool(sample.get("has_legislation_json_ld")) for sample in sample_results),
            "has_ssr_provision_content": sum(bool(sample.get("has_ssr_provision_content")) for sample in sample_results),
            "has_document_content_marker": sum(bool(sample.get("has_document_content_marker")) for sample in sample_results),
            "legislation_json_ld_coverage": coverage(sample_results),
        },
        "artifacts": {
            "inventory": inventory_path.name,
            "sample_pages": "sample_pages.jsonl",
            "robots": "robots.txt",
        },
    }
    write_json(output_dir / "report.json", report)
    write_report_markdown(output_dir / "REPORT.md", report)
    print(f"Report written to {output_dir / 'REPORT.md'}")
    return 0 if not sitemap_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
