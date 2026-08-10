"""Verified, read-only VBPL content adapter with explicit outcomes.

The server-action contract is configuration, not an assumed permanent API. A caller
must refresh and re-verify the hashes when VBPL deploys a new frontend build.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from crawler.rsc_parser import RSCParseError, parse_action_result
except ModuleNotFoundError:  # direct execution support
    from rsc_parser import RSCParseError, parse_action_result  # type: ignore[no-redef]


DEFAULT_HOME_URL = "https://vbpl.vn/"
DEFAULT_GATEWAY_URL = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc"
GROUP_NAMES = {
    "VBQPPL": "Văn bản quy phạm pháp luật",
    "VBHN": "Văn bản hợp nhất",
    "VBHTH": "Văn bản hệ thống hóa",
    "BD": "Bản dịch văn bản",
}
TOP_LEVEL_GROUP_CODES = set(GROUP_NAMES)
NORMATIVE_DECISION_NUMBER_RE = re.compile(r"/(?:19|20)\d{2}/Q[ĐD]", re.IGNORECASE)


class ContentStatus(str, Enum):
    HTML_VALID = "HTML_VALID"
    PDF_ONLY = "PDF_ONLY"
    METADATA_ONLY = "METADATA_ONLY"
    CONTENT_TOO_SHORT = "CONTENT_TOO_SHORT"
    SECURITY_CHALLENGE = "SECURITY_CHALLENGE"
    HTTP_ERROR = "HTTP_ERROR"
    CONTRACT_ERROR = "CONTRACT_ERROR"
    PARSE_ERROR = "PARSE_ERROR"


@dataclass(frozen=True)
class ActionContract:
    detail: str
    diagram: str
    files: str
    search: str
    verified_at: str
    source_chunk: str
    requires_cookie: bool = False
    requires_router_state: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "ActionContract":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class AdapterResult:
    document_id: str
    status: ContentStatus
    metadata: dict[str, Any] | None = None
    content_html: str = ""
    diagram: Any = None
    files: list[dict[str, Any]] = field(default_factory=list)
    official_group_code: str | None = None
    official_group_name: str | None = None
    official_form_code: str | None = None
    official_form_name: str | None = None
    warnings: list[str] = field(default_factory=list)
    backend: str = "server_action"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


class VBPLContractError(RuntimeError):
    pass


class VBPLHTTPError(VBPLContractError):
    pass


class VBPLResponseParseError(VBPLContractError):
    pass


def classify_official_doc_type(
    metadata: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None, list[str]] | None:
    """Read the official form/group embedded in a detail response.

    A missing ``parentCode`` must not turn a form code into a made-up top-level
    group. Only codes that are themselves known top-level groups may fall back.
    """

    doc_type = metadata.get("docType")
    if not isinstance(doc_type, dict) or not doc_type.get("code"):
        return None
    warnings: list[str] = []
    form_code = str(doc_type["code"])
    form_name = str(doc_type.get("name")) if doc_type.get("name") else None
    parent_code = doc_type.get("parentCode")
    group_code = str(parent_code) if parent_code else (
        form_code if form_code in TOP_LEVEL_GROUP_CODES else None
    )
    group_name = None
    if group_code:
        group_name = str(doc_type.get("parentName") or GROUP_NAMES.get(group_code) or form_name)
    else:
        warnings.append("MISSING_OFFICIAL_PARENT_GROUP")
    if group_code == "VBQPPL" and form_code == "QĐ":
        number = str(metadata.get("docNum") or "").upper().replace(" ", "")
        if not NORMATIVE_DECISION_NUMBER_RE.search(number):
            warnings.append("DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED")
    if group_code == "VBQPPL" and metadata.get("isAdministrativeDocument") is True:
        warnings.append("OFFICIAL_GROUP_ADMINISTRATIVE_FLAG_CONFLICT")
    return group_code, group_name, form_code, form_name, warnings


class NextActionClient:
    def __init__(
        self,
        contract: ActionContract,
        home_url: str = DEFAULT_HOME_URL,
        user_agent: str = "VietLawBERT-ContentAdapter/0.1",
        timeout: float = 90.0,
        delay: float = 0.75,
        retries: int = 2,
    ) -> None:
        self.contract = contract
        self.home_url = home_url
        self.user_agent = user_agent
        self.timeout = timeout
        self.delay = max(0.0, delay)
        self.retries = max(0, retries)
        self._last_request_at = 0.0

    def _call(self, action_hash: str, args: list[Any]) -> Any:
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            request = urllib.request.Request(
                self.home_url,
                data=json.dumps(args, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                method="POST",
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/x-component",
                    "Content-Type": "text/plain;charset=UTF-8",
                    "next-action": action_hash,
                    "Origin": "https://vbpl.vn",
                    "Referer": self.home_url,
                },
            )
            try:
                self._last_request_at = time.monotonic()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                    content_type = response.headers.get("content-type", "")
                    if "text/x-component" not in content_type:
                        title = payload[:500].decode("utf-8", "replace").lower()
                        if "security check" in title or "captcha" in title:
                            raise VBPLContractError("SECURITY_CHALLENGE")
                        raise VBPLContractError(f"Unexpected content type: {content_type}")
                    return parse_action_result(payload)
            except (urllib.error.URLError, TimeoutError, RSCParseError, VBPLContractError) as exc:
                error = exc
                if isinstance(exc, VBPLContractError) and str(exc) == "SECURITY_CHALLENGE":
                    break
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
        message = str(error) if error else "Unknown action failure"
        if isinstance(error, RSCParseError):
            raise VBPLResponseParseError(message) from error
        if isinstance(error, (urllib.error.URLError, TimeoutError)):
            raise VBPLHTTPError(message) from error
        if isinstance(error, VBPLContractError):
            raise error
        raise VBPLContractError(message)

    def detail(self, document_id: str) -> dict[str, Any]:
        value = self._call(self.contract.detail, [str(document_id)])
        if not isinstance(value, dict):
            raise VBPLContractError("Detail action did not return an object")
        return value

    def diagram(self, document_id: str) -> Any:
        return self._call(self.contract.diagram, [str(document_id)])

    def files(self, document_id: str, parts: list[int] | None = None) -> list[dict[str, Any]]:
        value = self._call(self.contract.files, [str(document_id), parts or [1, 2, 4]])
        if value is None:
            return []
        if not isinstance(value, list):
            raise VBPLContractError("Files action did not return a list")
        return value

    def search_number(self, number: str, page_size: int = 50) -> dict[str, Any]:
        params = {
            "pageNumber": 1,
            "pageSize": page_size,
            "keyword": number,
            "optionDoc": "number",
            "matchMode": "exact",
        }
        value = self._call(self.contract.search, [params])
        if not isinstance(value, dict):
            raise VBPLContractError("Search action did not return an object")
        return value


class VBPLContentAdapter:
    def __init__(self, client: NextActionClient, minimum_html_chars: int = 100) -> None:
        self.client = client
        self.minimum_html_chars = minimum_html_chars

    @staticmethod
    def _choose_content(metadata: dict[str, Any]) -> str:
        default_language = str(metadata.get("defaultLanguage") or "VN").upper()
        primary = metadata.get("documentContent") or {}
        english = metadata.get("documentContentEn") or {}
        candidates = [english, primary] if default_language == "EN" else [primary, english]
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("content"), str):
                if candidate["content"].strip():
                    return candidate["content"]
        return ""

    @staticmethod
    def _official_classification(
        document_id: str,
        metadata: dict[str, Any],
        client: NextActionClient,
    ) -> tuple[str | None, str | None, str | None, str | None, list[str]]:
        detail_classification = classify_official_doc_type(metadata)
        if detail_classification is not None:
            return detail_classification
        warnings: list[str] = []
        number = metadata.get("docNum")
        if not number:
            return None, None, None, None, ["MISSING_DOCUMENT_NUMBER_FOR_GROUP_LOOKUP"]
        search = client.search_number(str(number))
        matches = [item for item in search.get("items", []) if str(item.get("id")) == str(document_id)]
        if not matches:
            return None, None, None, None, ["OFFICIAL_GROUP_NOT_FOUND"]
        if len(matches) > 1:
            warnings.append("DUPLICATE_ID_IN_GROUP_LOOKUP")
        doc_type = matches[0].get("docType") or {}
        form_code = doc_type.get("code")
        form_name = doc_type.get("name")
        group_code = doc_type.get("parentCode") or form_code
        group_name = doc_type.get("parentName") or GROUP_NAMES.get(str(group_code)) or form_name
        if group_code == "VBQPPL" and form_code == "QĐ":
            normalized_number = str(number).upper().replace(" ", "")
            if not NORMATIVE_DECISION_NUMBER_RE.search(normalized_number):
                warnings.append("DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED")
        if group_code == "VBQPPL" and metadata.get("isAdministrativeDocument") is True:
            warnings.append("OFFICIAL_GROUP_ADMINISTRATIVE_FLAG_CONFLICT")
        return (
            str(group_code) if group_code else None,
            str(group_name) if group_name else None,
            str(form_code) if form_code else None,
            str(form_name) if form_name else None,
            warnings,
        )

    def fetch(self, document_id: str, include_diagram: bool = False) -> AdapterResult:
        document_id = str(document_id)
        try:
            metadata = self.client.detail(document_id)
            files = self.client.files(document_id)
            html = self._choose_content(metadata)
            group_code, group_name, form_code, form_name, warnings = self._official_classification(
                document_id, metadata, self.client
            )
            diagram = self.client.diagram(document_id) if include_diagram else None
            if len(html.strip()) >= self.minimum_html_chars:
                status = ContentStatus.HTML_VALID
            elif html.strip():
                status = ContentStatus.CONTENT_TOO_SHORT
            elif any("pdf" in str(item.get("fileName", "")).lower() for item in files):
                status = ContentStatus.PDF_ONLY
            else:
                status = ContentStatus.METADATA_ONLY
            return AdapterResult(
                document_id=document_id,
                status=status,
                metadata=metadata,
                content_html=html,
                diagram=diagram,
                files=files,
                official_group_code=group_code,
                official_group_name=group_name,
                official_form_code=form_code,
                official_form_name=form_name,
                warnings=warnings,
            )
        except VBPLHTTPError as exc:
            return AdapterResult(document_id=document_id, status=ContentStatus.HTTP_ERROR, error=str(exc))
        except VBPLResponseParseError as exc:
            return AdapterResult(document_id=document_id, status=ContentStatus.PARSE_ERROR, error=str(exc))
        except VBPLContractError as exc:
            message = str(exc)
            status = ContentStatus.SECURITY_CHALLENGE if "SECURITY_CHALLENGE" in message else ContentStatus.CONTRACT_ERROR
            return AdapterResult(document_id=document_id, status=status, error=message)
        except Exception as exc:  # retain unexpected failures as explicit records
            return AdapterResult(
                document_id=document_id,
                status=ContentStatus.PARSE_ERROR,
                error=f"{type(exc).__name__}: {exc}",
            )
