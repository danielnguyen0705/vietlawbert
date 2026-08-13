"""Verified, read-only VBPL content adapter with explicit outcomes.

The server-action contract is configuration, not an assumed permanent API. A caller
must refresh and re-verify the hashes when VBPL deploys a new frontend build.
"""

from __future__ import annotations

import importlib
import io
import json
import re
import shutil
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
MIN_PDF_TEXT_CHARS = 120
MIN_OCR_REVIEW_CHARS = 3000
RECOVERED_TEXT_QUALITY_WARNING_CODES = {
    "CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
    "OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED",
}
SUSPECT_CHAR_TOKENS = ("(cid:", "\x00", "�", "FOO CR")
TRUNCATION_END_PATTERNS = (
    " h6 t",
    " h6i t",
    " h6",
    " h0 t",
    " h0i t",
    " h0",
    " hô t",
    " hôi t",
    " hô",
)
MIN_VIETNAMESE_CHAR_RATIO = 0.05
MAX_SUSPECT_CHAR_DENSITY = 0.01
OCR_SELECTION_MIN_VIETNAMESE_RATIO = 0.03
OCR_SELECTION_MAX_SUSPECT_DENSITY = 0.002
OCR_SELECTION_TEXT_LAYER_WARNING_CODES = {
    "RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED",
    "RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED",
}


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
    extraction_method: str | None = None
    extraction_source_file: str | None = None
    extraction_confidence: float | None = None
    extraction_note: str | None = None
    ocr_attempted: bool = False
    pdf_url: str | None = None
    pdf_text_chars: int | None = None
    pdf_ocr_chars: int | None = None
    pdf_fetch_error: str | None = None
    recovered_text_normalized: str | None = None
    recovered_text_normalization: dict[str, Any] | None = None
    recovered_text_metrics: dict[str, Any] | None = None
    recovered_text_selection: dict[str, Any] | None = None
    manual_corpus_review: dict[str, Any] | None = None

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


@dataclass(frozen=True)
class PDFExtractionResult:
    text: str
    method: str
    source_file: str | None = None
    confidence: float = 0.0
    note: str | None = None
    ocr_attempted: bool = False
    text_chars: int = 0
    ocr_chars: int = 0

    @property
    def usable(self) -> bool:
        return len(self.text.strip()) >= MIN_PDF_TEXT_CHARS


@dataclass(frozen=True)
class RecoveredTextQualityMetrics:
    total_chars: int
    suspect_token_count: int
    suspect_char_density: float
    vietnamese_char_ratio: float
    structure_marker_count: int
    trailing_fragment_suspected: bool


def _count_suspect_tokens(text: str) -> int:
    return sum(text.count(token) for token in SUSPECT_CHAR_TOKENS)


def _vietnamese_char_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    vietnamese_diacritics = 0
    for char in letters:
        normalized = unicodedata.normalize("NFD", char)
        if char in {"đ", "Đ"} or any(unicodedata.combining(component) for component in normalized):
            vietnamese_diacritics += 1
    return vietnamese_diacritics / len(letters)


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return stripped.replace("đ", "d").replace("Đ", "D")


def _structure_marker_count(text: str) -> int:
    lowered = _strip_diacritics(text.lower())
    markers = ("dieu", "khoan", "muc", "chuong", "quyet nghi", "can cu")
    return sum(lowered.count(marker) for marker in markers)


def _trailing_fragment_suspected(text: str) -> bool:
    compact = " ".join(text.strip().lower().split())
    if not compact:
        return False
    if compact.endswith(("...", "…", ":", ";", ",", "(", "[", "{", "/")):
        return True
    return any(compact.endswith(pattern) for pattern in TRUNCATION_END_PATTERNS)


def recovered_text_quality_metrics(text: str) -> RecoveredTextQualityMetrics:
    stripped = text.strip()
    total_chars = len(stripped)
    suspect_token_count = _count_suspect_tokens(stripped)
    suspect_char_density = suspect_token_count / total_chars if total_chars else 0.0
    return RecoveredTextQualityMetrics(
        total_chars=total_chars,
        suspect_token_count=suspect_token_count,
        suspect_char_density=suspect_char_density,
        vietnamese_char_ratio=_vietnamese_char_ratio(stripped),
        structure_marker_count=_structure_marker_count(stripped),
        trailing_fragment_suspected=_trailing_fragment_suspected(stripped),
    )


def remediate_recovered_text(text: str, extraction: PDFExtractionResult) -> tuple[str, dict[str, Any]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.strip()
    flags: list[str] = []
    if normalized != text.strip():
        flags.append("whitespace_normalized")
    return normalized, {
        "method": extraction.method,
        "flags": flags,
        "normalized_chars": len(normalized),
    }


def recovered_text_quality_warnings(text: str, extraction: PDFExtractionResult) -> list[str]:
    warnings: list[str] = []
    stripped = text.strip()
    if not stripped:
        return warnings
    metrics = recovered_text_quality_metrics(stripped)
    if extraction.method == "PDF_OCR" and len(stripped) < MIN_OCR_REVIEW_CHARS:
        warnings.append("OCR_RECOVERED_TEXT_SHORT_REVIEW_RECOMMENDED")
    if extraction.method in {"PDF_TEXT_LAYER", "PDF_OCR"} and metrics.suspect_token_count > 0:
        warnings.append("RECOVERED_TEXT_ENCODING_REVIEW_RECOMMENDED")
    if extraction.method in {"PDF_TEXT_LAYER", "PDF_OCR"} and metrics.suspect_char_density > MAX_SUSPECT_CHAR_DENSITY:
        warnings.append("RECOVERED_TEXT_SUSPECT_CHAR_DENSITY_REVIEW_RECOMMENDED")
    if extraction.method in {"PDF_TEXT_LAYER", "PDF_OCR"} and metrics.vietnamese_char_ratio < MIN_VIETNAMESE_CHAR_RATIO:
        warnings.append("RECOVERED_TEXT_LOW_VIETNAMESE_RATIO_REVIEW_RECOMMENDED")
    if extraction.method in {"PDF_TEXT_LAYER", "PDF_OCR"} and metrics.trailing_fragment_suspected:
        warnings.append("RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED")
    return list(dict.fromkeys(warnings))


def recovered_text_quality_score(metrics: RecoveredTextQualityMetrics) -> float:
    score = float(metrics.total_chars)
    score += float(metrics.structure_marker_count) * 25.0
    score -= float(metrics.suspect_token_count) * 250.0
    score -= float(metrics.suspect_char_density) * 100_000.0
    if metrics.vietnamese_char_ratio < MIN_VIETNAMESE_CHAR_RATIO:
        score -= (MIN_VIETNAMESE_CHAR_RATIO - metrics.vietnamese_char_ratio) * 10_000.0
    if metrics.trailing_fragment_suspected:
        score -= 1_000.0
    return score


def _should_try_ocr_over_text_layer(text_layer: PDFExtractionResult) -> tuple[bool, dict[str, Any]]:
    text_metrics = recovered_text_quality_metrics(text_layer.text)
    text_warnings = recovered_text_quality_warnings(text_layer.text, text_layer)
    text_warning_set = set(text_warnings)
    decision = {
        "text_layer_method": text_layer.method,
        "text_layer_chars": len(text_layer.text.strip()),
        "text_layer_metrics": asdict(text_metrics),
        "text_layer_warnings": text_warnings,
        "text_layer_score": recovered_text_quality_score(text_metrics),
        "ocr_attempted_for_selection": False,
        "selected_method": text_layer.method,
        "reason": "text_layer_accepted",
    }
    severe_text_layer_noise = OCR_SELECTION_TEXT_LAYER_WARNING_CODES.issubset(text_warning_set)
    return severe_text_layer_noise, decision


def choose_recovered_pdf_extraction(
    text_layer: PDFExtractionResult,
    ocr_result: PDFExtractionResult,
    source_file: str | None = None,
) -> tuple[PDFExtractionResult, dict[str, Any]]:
    should_try_ocr, decision = _should_try_ocr_over_text_layer(text_layer)
    if not should_try_ocr:
        return text_layer, decision
    ocr_metrics = recovered_text_quality_metrics(ocr_result.text)
    ocr_warnings = recovered_text_quality_warnings(ocr_result.text, ocr_result)
    ocr_score = recovered_text_quality_score(ocr_metrics)
    decision.update({
        "ocr_attempted_for_selection": True,
        "ocr_method": ocr_result.method,
        "ocr_chars": len(ocr_result.text.strip()),
        "ocr_metrics": asdict(ocr_metrics),
        "ocr_warnings": ocr_warnings,
        "ocr_score": ocr_score,
    })
    ocr_warning_set = set(ocr_warnings)
    ocr_is_usable = (
        len(ocr_result.text.strip()) >= MIN_PDF_TEXT_CHARS
        and ocr_result.method == "PDF_OCR"
        and ocr_metrics.suspect_char_density <= OCR_SELECTION_MAX_SUSPECT_DENSITY
        and ocr_metrics.vietnamese_char_ratio >= OCR_SELECTION_MIN_VIETNAMESE_RATIO
        and "RECOVERED_TEXT_TRUNCATION_REVIEW_RECOMMENDED" not in ocr_warning_set
    )
    if ocr_is_usable and ocr_score > decision["text_layer_score"]:
        decision.update({
            "selected_method": "PDF_OCR",
            "reason": "ocr_selected_over_noisy_text_layer",
        })
        return PDFExtractionResult(
            text=ocr_result.text,
            method="PDF_OCR",
            source_file=source_file,
            confidence=ocr_result.confidence,
            note="ocr_selected_over_noisy_text_layer",
            ocr_attempted=True,
            text_chars=len(text_layer.text),
            ocr_chars=len(ocr_result.text),
        ), decision
    decision.update({
        "selected_method": text_layer.method,
        "reason": "text_layer_kept_after_ocr_comparison",
    })
    return PDFExtractionResult(
        text=text_layer.text,
        method=text_layer.method,
        source_file=source_file,
        confidence=text_layer.confidence,
        note=text_layer.note,
        ocr_attempted=True,
        text_chars=len(text_layer.text),
        ocr_chars=len(ocr_result.text),
    ), decision


class PDFExtractor:
    def __init__(self, minimum_text_chars: int = MIN_PDF_TEXT_CHARS, enable_ocr: bool = True) -> None:
        self.minimum_text_chars = minimum_text_chars
        self.enable_ocr = enable_ocr

    @staticmethod
    def _module(name: str) -> Any | None:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            return None

    @staticmethod
    def _clean_text(parts: list[str]) -> str:
        return "\n\n".join(part.strip() for part in parts if part and part.strip()).strip()

    def extract_text_layer(self, pdf_bytes: bytes, source_file: str | None = None) -> PDFExtractionResult:
        pdfplumber = self._module("pdfplumber")
        if pdfplumber is None:
            return PDFExtractionResult("", "PDF_TEXT_UNAVAILABLE", source_file, 0.0, "pdfplumber_missing")
        try:
            parts: list[str] = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    parts.append(page.extract_text() or "")
            text = self._clean_text(parts)
            confidence = 0.95 if len(text) >= self.minimum_text_chars else 0.25
            return PDFExtractionResult(
                text=text,
                method="PDF_TEXT_LAYER",
                source_file=source_file,
                confidence=confidence,
                note=None if text else "empty_text_layer",
                text_chars=len(text),
            )
        except Exception as exc:
            return PDFExtractionResult(
                "",
                "PDF_TEXT_ERROR",
                source_file,
                0.0,
                f"{type(exc).__name__}: {exc}",
            )

    def extract_ocr(self, pdf_bytes: bytes, source_file: str | None = None) -> PDFExtractionResult:
        if not self.enable_ocr:
            return PDFExtractionResult("", "OCR_DISABLED", source_file, 0.0, "ocr_disabled", ocr_attempted=False)
        pdf2image = self._module("pdf2image")
        pytesseract = self._module("pytesseract")
        if pdf2image is None or pytesseract is None or not shutil.which("tesseract"):
            return PDFExtractionResult(
                "",
                "OCR_UNAVAILABLE",
                source_file,
                0.0,
                "missing_pdf2image_pytesseract_or_tesseract_binary",
                ocr_attempted=True,
            )
        try:
            images = pdf2image.convert_from_bytes(pdf_bytes, dpi=200)
            parts = [pytesseract.image_to_string(image, lang="vie+eng") for image in images]
            text = self._clean_text(parts)
            confidence = 0.70 if len(text) >= self.minimum_text_chars else 0.20
            return PDFExtractionResult(
                text=text,
                method="PDF_OCR",
                source_file=source_file,
                confidence=confidence,
                note=None if text else "empty_ocr_text",
                ocr_attempted=True,
                ocr_chars=len(text),
            )
        except Exception as exc:
            return PDFExtractionResult(
                "",
                "OCR_ERROR",
                source_file,
                0.0,
                f"{type(exc).__name__}: {exc}",
                ocr_attempted=True,
            )

    def extract(self, pdf_bytes: bytes, source_file: str | None = None) -> PDFExtractionResult:
        text_layer = self.extract_text_layer(pdf_bytes, source_file)
        if len(text_layer.text.strip()) >= self.minimum_text_chars:
            should_try_ocr, _ = _should_try_ocr_over_text_layer(text_layer)
            if not should_try_ocr:
                return text_layer
            ocr_result = self.extract_ocr(pdf_bytes, source_file)
            selected, _ = choose_recovered_pdf_extraction(text_layer, ocr_result, source_file)
            return selected
        ocr_result = self.extract_ocr(pdf_bytes, source_file)
        if len(ocr_result.text.strip()) >= self.minimum_text_chars:
            return PDFExtractionResult(
                text=ocr_result.text,
                method=ocr_result.method,
                source_file=source_file,
                confidence=ocr_result.confidence,
                note=ocr_result.note,
                ocr_attempted=True,
                text_chars=len(text_layer.text),
                ocr_chars=len(ocr_result.text),
            )
        note_parts = [part for part in (text_layer.note, ocr_result.note) if part]
        return PDFExtractionResult(
            text=text_layer.text or ocr_result.text,
            method="PDF_EXTRACTION_INSUFFICIENT",
            source_file=source_file,
            confidence=max(text_layer.confidence, ocr_result.confidence),
            note=";".join(note_parts) or "insufficient_pdf_text",
            ocr_attempted=ocr_result.ocr_attempted,
            text_chars=len(text_layer.text),
            ocr_chars=len(ocr_result.text),
        )

    def extract_with_selection(self, pdf_bytes: bytes, source_file: str | None = None) -> tuple[PDFExtractionResult, dict[str, Any] | None]:
        text_layer = self.extract_text_layer(pdf_bytes, source_file)
        if len(text_layer.text.strip()) >= self.minimum_text_chars:
            should_try_ocr, decision = _should_try_ocr_over_text_layer(text_layer)
            if not should_try_ocr:
                return text_layer, decision
            ocr_result = self.extract_ocr(pdf_bytes, source_file)
            selected, decision = choose_recovered_pdf_extraction(text_layer, ocr_result, source_file)
            return selected, decision
        result = self.extract(pdf_bytes, source_file)
        if result.method == "PDF_OCR":
            return result, {
                "selected_method": "PDF_OCR",
                "reason": "text_layer_too_short_ocr_selected",
                "text_layer_chars": len(text_layer.text.strip()),
                "ocr_chars": len(result.text.strip()),
            }
        return result, None


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
        gateway_url: str = DEFAULT_GATEWAY_URL,
        user_agent: str = "VietLawBERT-ContentAdapter/0.1",
        timeout: float = 90.0,
        delay: float = 0.75,
        retries: int = 2,
    ) -> None:
        self.contract = contract
        self.home_url = home_url
        self.gateway_url = gateway_url.rstrip("/")
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

    def download_pdf(self, document_id: str, file_name: str | None = None) -> tuple[bytes, str]:
        """Download one PDF attachment without treating HTTP 200 as valid content."""
        if file_name:
            url = (
                f"{self.gateway_url}/minio/buckets/vbpl/"
                f"{quote(str(document_id), safe='')}/{quote(file_name, safe='')}/download"
            )
        else:
            url = f"{self.gateway_url}/{quote(str(document_id), safe='')}/download"
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/pdf,*/*",
                "Origin": "https://vbpl.vn",
                "Referer": self.home_url,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                content_type = response.headers.get("content-type", "").lower()
                if response.status != 200:
                    raise VBPLHTTPError(f"PDF HTTP status: {response.status}")
                if not body.startswith(b"%PDF") and "application/pdf" not in content_type:
                    preview = body[:300].decode("utf-8", "replace").lower()
                    if any(token in preview for token in ("captcha", "cloudflare", "security verification")):
                        raise VBPLContractError("SECURITY_CHALLENGE")
                    raise VBPLContractError(f"PDF response is not PDF: {content_type or 'unknown'}")
                if not body:
                    raise VBPLHTTPError("PDF response is empty")
                return body, url
        except urllib.error.HTTPError as exc:
            raise VBPLHTTPError(f"PDF HTTP status: {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise VBPLHTTPError(str(exc)) from exc

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
    def __init__(
        self,
        client: NextActionClient,
        minimum_html_chars: int = 100,
        pdf_extractor: PDFExtractor | None = None,
        enable_pdf_fallback: bool = False,
    ) -> None:
        self.client = client
        self.minimum_html_chars = minimum_html_chars
        self.pdf_extractor = pdf_extractor or PDFExtractor(minimum_text_chars=minimum_html_chars)
        self.enable_pdf_fallback = enable_pdf_fallback

    @staticmethod
    def _pdf_file_name(files: list[dict[str, Any]]) -> str | None:
        for item in files:
            if not isinstance(item, dict):
                continue
            file_name = item.get("fileName") or item.get("objectName") or item.get("name")
            if isinstance(file_name, str) and file_name.lower().endswith(".pdf"):
                return file_name
        return None

    def _extract_pdf_fallback(
        self,
        document_id: str,
        files: list[dict[str, Any]],
    ) -> tuple[PDFExtractionResult | None, str | None, str | None, dict[str, Any] | None]:
        file_name = self._pdf_file_name(files)
        if not file_name:
            return None, None, None, None
        downloader = getattr(self.client, "download_pdf", None)
        if not callable(downloader):
            return None, None, "PDF_DOWNLOADER_UNAVAILABLE", None
        try:
            pdf_bytes, pdf_url = downloader(document_id, file_name)
            extractor = getattr(self.pdf_extractor, "extract_with_selection", None)
            if callable(extractor):
                extraction, selection = extractor(pdf_bytes, file_name)
                return extraction, pdf_url, None, selection
            return self.pdf_extractor.extract(pdf_bytes, file_name), pdf_url, None, None
        except VBPLContractError as exc:
            if "SECURITY_CHALLENGE" in str(exc):
                raise
            return None, None, f"{type(exc).__name__}: {exc}", None
        except Exception as exc:
            return None, None, f"{type(exc).__name__}: {exc}", None

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
            extraction: PDFExtractionResult | None = None
            pdf_url: str | None = None
            pdf_fetch_error: str | None = None
            recovered_text_normalized: str | None = None
            recovered_text_normalization: dict[str, Any] | None = None
            recovered_text_metrics: dict[str, Any] | None = None
            recovered_text_selection: dict[str, Any] | None = None
            has_pdf = self._pdf_file_name(files) is not None
            if len(html.strip()) >= self.minimum_html_chars:
                status = ContentStatus.HTML_VALID
            else:
                if self.enable_pdf_fallback and has_pdf:
                    extraction, pdf_url, pdf_fetch_error, recovered_text_selection = self._extract_pdf_fallback(document_id, files)
                if extraction and extraction.usable:
                    html = extraction.text
                    recovered_text_normalized, recovered_text_normalization = remediate_recovered_text(html, extraction)
                    recovered_text_metrics = asdict(recovered_text_quality_metrics(html))
                    status = ContentStatus.HTML_VALID
                    warnings.append("CONTENT_RECOVERED_FROM_PDF")
                    if extraction.method == "PDF_OCR":
                        warnings.append("CONTENT_RECOVERED_BY_OCR_REVIEW_RECOMMENDED")
                    warnings.extend(recovered_text_quality_warnings(html, extraction))
                elif html.strip():
                    status = ContentStatus.CONTENT_TOO_SHORT
                    if extraction:
                        warnings.append("PDF_EXTRACTION_INSUFFICIENT")
                elif has_pdf:
                    status = ContentStatus.PDF_ONLY
                    if extraction:
                        warnings.append("PDF_EXTRACTION_INSUFFICIENT")
                    if pdf_fetch_error:
                        warnings.append("PDF_FETCH_FAILED")
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
                extraction_method=extraction.method if extraction else ("HTML" if status == ContentStatus.HTML_VALID else None),
                extraction_source_file=extraction.source_file if extraction else None,
                extraction_confidence=extraction.confidence if extraction else (1.0 if status == ContentStatus.HTML_VALID else None),
                extraction_note=extraction.note if extraction else None,
                ocr_attempted=extraction.ocr_attempted if extraction else False,
                pdf_url=pdf_url,
                pdf_text_chars=extraction.text_chars if extraction else None,
                pdf_ocr_chars=extraction.ocr_chars if extraction else None,
                pdf_fetch_error=pdf_fetch_error,
                recovered_text_normalized=recovered_text_normalized,
                recovered_text_normalization=recovered_text_normalization,
                recovered_text_metrics=recovered_text_metrics,
                recovered_text_selection=recovered_text_selection,
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
