from __future__ import annotations

import re
import time
import unicodedata
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


DOC_TYPE_PATTERNS = [
    "BỘ LUẬT",
    "NGHỊ QUYẾT",
    "QUYẾT ĐỊNH",
    "NGHỊ ĐỊNH",
    "THÔNG TƯ",
    "CHỈ THỊ",
    "QUY CHẾ",
    "QUY ĐỊNH",
    "HƯỚNG DẪN",
    "LUẬT",
]

DOC_TYPE_ABBR_MAP = {
    "NĐ": "Nghị định",
    "ND": "Nghị định",
    "NQ": "Nghị quyết",
    "QĐ": "Quyết định",
    "QD": "Quyết định",
    "CT": "Chỉ thị",
    "QC": "Quy chế",
    "QYĐ": "Quy định",
    "QYD": "Quy định",
    "TC": "Thông cáo",
    "TB": "Thông báo",
    "HD": "Hướng dẫn",
    "CTR": "Chương trình",
    "KH": "Kế hoạch",
    "PA": "Phương án",
    "ĐA": "Đề án",
    "DA": "Dự án",
    "BC": "Báo cáo",
    "BB": "Biên bản",
    "TTR": "Tờ trình",
    "HĐ": "Hợp đồng",
    "CĐ": "Công điện",
    "BGN": "Bản ghi nhớ",
    "BTT": "Bản thỏa thuận",
    "GUQ": "Giấy ủy quyền",
    "GM": "Giấy mời",
    "GGT": "Giấy giới thiệu",
    "GNP": "Giấy nghỉ phép",
    "PG": "Phiếu gửi",
    "PC": "Phiếu chuyển",
    "PB": "Phiếu báo",
    "TT": "Thông tư",
    "TTLT": "Thông tư liên tịch",
    "TTLTT": "Thông tư liên tịch",
    "TT-LT": "Thông tư liên tịch",
}

AGENCY_REGISTRY = {
    "CP": {"name": "Chính phủ", "aliases": ["CHINHPHU", "CP"]},
    "VPCP": {"name": "Văn phòng Chính phủ", "aliases": ["VPCP", "VANPHONGCHINHPHU"]},
    "VPQH": {"name": "Văn phòng Quốc hội", "aliases": ["VPQH", "VANPHONGQUOCHOI"]},
    "VPCTN": {"name": "Văn phòng Chủ tịch nước", "aliases": ["VPCTN", "VANPHONGCHUTICHNUOC"]},
    "TANDTC": {"name": "Tòa án nhân dân tối cao", "aliases": ["TANDTC", "TOAANNHANDANTOICAO"]},
    "VKSNDTC": {"name": "Viện Kiểm sát nhân dân tối cao", "aliases": ["VKSNDTC", "VIENKIEMSATNhandANTOICAO", "VIENKIEMSATNHANDANTOICAO"]},
    "BNG": {"name": "Bộ Ngoại giao", "aliases": ["BNG", "BONGOAIGIAO"]},
    "BTP": {"name": "Bộ Tư pháp", "aliases": ["BTP", "BOTUPHAP"]},
    "KTNN": {"name": "Kiểm toán Nhà nước", "aliases": ["KTNN", "KIEMTOANNHANUOC"]},
    "BKHĐT": {"name": "Bộ Kế hoạch và Đầu tư", "aliases": ["BKHDT", "BOKEHOACHVADAUTU"]},
    "TTCP": {"name": "Thanh tra Chính phủ", "aliases": ["TTCP", "THANHTRACHINHPHU"]},
    "BTTTT": {"name": "Bộ Thông tin và Truyền thông", "aliases": ["BTTTT", "BOTHONGTINVATRUYENTHONG"]},
    "BNV": {"name": "Bộ Nội vụ", "aliases": ["BNV", "BONOIVU", "NV"]},
    "BTC": {"name": "Bộ Tài chính", "aliases": ["BTC", "BOTAICHINH"]},
    "BVHTTDL": {"name": "Bộ Văn hóa, Thể thao và Du lịch", "aliases": ["BVHTTDL", "BOVANHOATHETHAOVADULICH"]},
    "BGDĐT": {"name": "Bộ Giáo dục và Đào tạo", "aliases": ["BGDDT", "BOGIAODUCVADAOTAO"]},
    "BKHCN": {"name": "Bộ Khoa học và Công nghệ", "aliases": ["BKHCN", "BOKHOAHOCVACONGNGHE"]},
    "BYT": {"name": "Bộ Y tế", "aliases": ["BYT", "BOYTE"]},
    "BLĐTBXH": {"name": "Bộ Lao động - Thương binh và Xã hội", "aliases": ["BLDTBXH", "BOLAODONGTHUONGBINHVAXAHOI"]},
    "UBDT": {"name": "Ủy ban Dân tộc", "aliases": ["UBDT", "UYBANDANTOC"]},
    "BNN": {"name": "Bộ Nông nghiệp và Phát triển nông thôn", "aliases": ["BNN", "BONONGNGHIEPVAPHATTRIENNONGTHON", "BNNPTNT"]},
    "BCT": {"name": "Bộ Công Thương", "aliases": ["BCT", "BOCONGTHUONG"]},
    "BTNMT": {"name": "Bộ Tài nguyên và Môi trường", "aliases": ["BTNMT", "BOTAINGUYENVAMOITRUONG"]},
    "BXD": {"name": "Bộ Xây dựng", "aliases": ["BXD", "BOXAYDUNG"]},
    "BGTVT": {"name": "Bộ Giao thông vận tải", "aliases": ["BGTVT", "GTVT", "BOGIAOTHONGVANTAI"]},
    "BCA": {"name": "Bộ Công an", "aliases": ["BCA", "BOCONGAN"]},
    "BQP": {"name": "Bộ Quốc phòng", "aliases": ["BQP", "BOQUOCPHONG"]},
}

ALIASES_TO_CODES = {}
for code, meta in AGENCY_REGISTRY.items():
    ALIASES_TO_CODES[code] = code
    ALIASES_TO_CODES[re.sub(r'[^A-Z0-9]+', '', code)] = code
    for alias in meta["aliases"]:
        ALIASES_TO_CODES[re.sub(r'[^A-Z0-9]+', '', alias.upper())] = code

TITLE_STOP_PATTERNS = [
    r"^Căn cứ\b",
    r"^Theo đề nghị\b",
    r"^Điều\s+1\.?\b",
    r"^CHƯƠNG\s+",
    r"^MỤC\s+",
    r"^PHẦN\s+",
    r"^Nơi nhận:\s*",
    r"^KT\.\s*",
    r"^TM\.\s*",
    r"^Ban hành kèm theo\b",
    r"^Kèm theo\b",
    r"^(BỘ TRƯỞNG|THỨ TRƯỞNG|CỤC TRƯỞNG|TỔNG CỤC TRƯỞNG|CHỦ TỊCH)\b",
    r"^CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\b",
]

DECORATIVE_LINE_PATTERNS = [r"^[_\-=—\s]{5,}$"]

TITLE_LEAD_PATTERNS = [
    r"VỀ\s+VIỆC\b",
    r"BAN\s+HÀNH\b",
    r"QUY\s+ĐỊNH\b",
    r"QUY\s+CHẾ\b",
    r"HƯỚNG\s+DẪN\b",
    r"PHÊ\s+DUYỆT\b",
    r"CÔNG\s+BỐ\b",
    r"SỬA\s+ĐỔI\b",
    r"BỔ\s+SUNG\b",
]

ISSUER_LINE_PATTERNS = [
    r"^CỦA\s+(BỘ TRƯỞNG|THỦ TƯỚNG|CHỦ TỊCH|BỘ|ỦY BAN|UỶ BAN|TỔNG CỤC|CỤC)\b",
    r"^(BỘ TRƯỞNG|THỨ TRƯỞNG|CHỦ TỊCH|TỔNG CỤC TRƯỞNG|CỤC TRƯỞNG)\b",
]


def _norm_text(s: str) -> str:
    s = (s or "").replace("\xa0", " ").replace("\u200b", " ").replace("\u00ad", " ")
    return " ".join(s.split()).strip()


def _norm_preserve_newlines(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\u200b", " ").replace("\u00ad", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_norm_text(line) for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _normalize_code(text: str) -> str:
    text = _norm_text(text).upper().replace("Đ", "D")
    text = _strip_accents(text).replace("Đ", "D")
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return text


def canonicalize_agency_code(value: str) -> str:
    key = _normalize_code(value)
    return ALIASES_TO_CODES.get(key, "")


def canonicalize_agency_name(value: str) -> str:
    code = canonicalize_agency_code(value)
    if not code:
        return _norm_text(value)
    return AGENCY_REGISTRY[code]["name"]


def _join_agency_names(codes: List[str]) -> str:
    names = [AGENCY_REGISTRY[c]["name"] for c in codes if c in AGENCY_REGISTRY]
    return " - ".join(dict.fromkeys(names))


def _extract_agency_mentions(text: str) -> List[str]:
    s = _normalize_code(text)
    found: List[str] = []
    for alias, code in ALIASES_TO_CODES.items():
        if len(alias) < 3:
            continue
        if alias and alias in s:
            found.append(code)
    return list(dict.fromkeys(found))


def _is_decorative_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if any(re.match(pat, s) for pat in DECORATIVE_LINE_PATTERNS):
        return True
    return bool(re.fullmatch(r"[_\-=—–·•.]{3,}", s))


def _is_title_stop_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    return any(re.match(pat, s, flags=re.IGNORECASE) for pat in TITLE_STOP_PATTERNS)


def _is_issuer_like_line(line: str) -> bool:
    s = _norm_text(line).upper()
    if not s:
        return False
    if any(re.match(pat, s, flags=re.IGNORECASE) for pat in ISSUER_LINE_PATTERNS):
        return True
    codes = _extract_agency_mentions(s)
    return len(codes) == 1 and len(s.split()) <= 8


def extract_links_from_result_html(html: str, base_url: str, type_value: str, ministry: str) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    items: List[Dict] = []
    for a in soup.select("ul.listLaw p.title a[href*='vbpq-toanvan.aspx']"):
        href = (a.get("href") or "").strip()
        doc_id = _norm_text(a.get_text(" ", strip=True))
        if not href:
            continue
        items.append({
            "type": type_value,
            "doc_id": doc_id,
            "url": urljoin(base_url, href),
            "ministry": canonicalize_agency_name(ministry),
        })
    seen = set()
    dedup: List[Dict] = []
    for it in items:
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        dedup.append(it)
    return dedup


def fetch_html(url: str, timeout: int = 60, max_retries: int = 3, sleep_sec: float = 1.0) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries:
                print(f"[Retry {attempt}/{max_retries}] {url} -> {e}")
                time.sleep(sleep_sec)
    assert last_error is not None
    raise last_error


def extract_item_id(url: str) -> Optional[str]:
    m = re.search(r"ItemID=(\d+)", url)
    return m.group(1) if m else None


def _extract_raw_text(soup: BeautifulSoup) -> str:
    candidate_selectors = [
        "div#toanvancontent",
        "div.toanvan",
        "div#contentdetail",
        "div.content1",
        "div.content",
        "td.ToanVan",
        "body",
    ]
    for sel in candidate_selectors:
        tag = soup.select_one(sel)
        if not tag:
            continue
        text = tag.get_text("\n", strip=True)
        text = _norm_preserve_newlines(text)
        if text and len(text) > 200:
            return text
    body_text = soup.get_text("\n", strip=True)
    return _norm_preserve_newlines(body_text)


def _split_doc_id(doc_id: str) -> Tuple[str, str]:
    doc_id = _norm_text(doc_id)
    if not doc_id:
        return "", ""
    m = re.match(
        r"^(Quyết định|Nghị định|Thông tư|Chỉ thị|Luật|Bộ luật|Nghị quyết|Quy chế|Quy định|Hướng dẫn)\s+(.+)$",
        doc_id,
        flags=re.IGNORECASE,
    )
    if m:
        return _norm_text(m.group(1)), _norm_text(m.group(2))
    return "", doc_id


def _first_nonempty_lines(text: str, limit: int = 80) -> List[str]:
    lines = [_norm_text(line) for line in text.split("\n")]
    return [line for line in lines if line][:limit]


def _extract_agency_lines(lines: List[str]) -> List[str]:
    agency_lines: List[str] = []
    for line in lines[:15]:
        s = _norm_text(line)
        if not s or _is_decorative_line(s):
            continue
        if re.search(r"Số\s*[:.]", s, flags=re.IGNORECASE):
            break
        if re.match(r"^(CỘNG\s+HÒA|Độc lập|Hà Nội|TP\.|Thành phố)", s, flags=re.IGNORECASE):
            continue
        if re.match(r"^(QUYẾT ĐỊNH|NGHỊ ĐỊNH|THÔNG TƯ|CHỈ THỊ|LUẬT|BỘ LUẬT|NGHỊ QUYẾT)\b", s, flags=re.IGNORECASE):
            break
        agency_lines.append(s)
    return agency_lines


def _extract_doc_number_only(doc_id: str) -> str:
    _, doc_number = _split_doc_id(doc_id)
    return doc_number or _norm_text(doc_id)


def _parse_doc_type_from_number(doc_number: str) -> str:
    if not doc_number:
        return ""
    parts = [p.strip() for p in re.split(r"[-/]", doc_number) if p.strip()]
    normalized_parts = [_normalize_code(p) for p in parts]
    for i in range(1, len(parts)):
        original = parts[i].upper()
        normalized = normalized_parts[i]
        if original == "TT-LT" or normalized == "TTLT":
            return "Thông tư liên tịch"
        if normalized in DOC_TYPE_ABBR_MAP:
            return DOC_TYPE_ABBR_MAP[normalized]
    return ""


def _parse_agency_codes_from_doc_number(doc_number: str) -> List[str]:
    if not doc_number:
        return []
    parts = [p.strip() for p in re.split(r"[-/]", doc_number) if p.strip()]
    found: List[str] = []
    for part in parts[1:]:
        norm = _normalize_code(part)
        if norm in {_normalize_code(k) for k in DOC_TYPE_ABBR_MAP.keys()}:
            continue
        if norm in {"LT", "LB", "LN"}:
            continue
        if norm.startswith("LB") and len(norm) > 2:
            tail_codes = _extract_agency_mentions(norm[2:])
            found.extend(tail_codes)
            continue
        code = canonicalize_agency_code(norm)
        if code:
            found.append(code)
            continue
        found.extend(_extract_agency_mentions(norm))
    return list(dict.fromkeys(found))


def _parse_agency_codes_from_agency_lines(lines: List[str]) -> List[str]:
    found: List[str] = []
    for line in lines:
        found.extend(_extract_agency_mentions(line))
    return list(dict.fromkeys(found))


def _clean_title_text(text: str) -> str:
    original_text = _norm_text(text)
    if not original_text:
        return ""
    lead_match = re.search("|".join(TITLE_LEAD_PATTERNS), original_text, flags=re.IGNORECASE)
    text = original_text[lead_match.start():] if lead_match else original_text
    text = text.replace("\\", " ").replace("/", " ")
    text = text.replace('"', '“')
    text = re.sub(r"[_\-=—–]{3,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^SỐ\s*[:.]?\s*[^\s]+\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^NGÀY\s+\d{1,2}\s+THÁNG\s+\d{1,2}\s+NĂM\s+\d{4}\s*", "", text, flags=re.IGNORECASE)
    return _norm_text(text)


def _extract_doc_type_and_title_from_lines(lines: List[str], fallback_doc_type: str = "") -> Tuple[str, str]:
    if not lines:
        return fallback_doc_type, ""

    for i, line in enumerate(lines[:25]):
        upper = line.upper().strip()
        for doc_type_upper in DOC_TYPE_PATTERNS:
            if upper == doc_type_upper:
                title_parts: List[str] = []
                for next_line in lines[i + 1 : i + 8]:
                    next_line = next_line.strip()
                    if not next_line:
                        break
                    if _is_decorative_line(next_line) or _is_title_stop_line(next_line) or _is_issuer_like_line(next_line):
                        continue
                    title_parts.append(next_line)
                title = _clean_title_text(" ".join(title_parts))
                return fallback_doc_type or doc_type_upper.title(), title

    for line in lines[:25]:
        upper = line.upper().strip()
        for doc_type_upper in DOC_TYPE_PATTERNS:
            if upper.startswith(doc_type_upper + " "):
                title = _clean_title_text(line[len(doc_type_upper):])
                return fallback_doc_type or doc_type_upper.title(), title

    probable_title_parts: List[str] = []
    for line in lines[:10]:
        if _is_decorative_line(line) or _is_title_stop_line(line) or _is_issuer_like_line(line):
            continue
        probable_title_parts.append(line)
    return fallback_doc_type, _clean_title_text(" ".join(probable_title_parts[:3]))


def extract_document_from_html(html: str, url: str, fallback_doc_id: str = "") -> Dict:
    soup = BeautifulSoup(html, "lxml")
    raw_text = _extract_raw_text(soup)
    item_id = extract_item_id(url) or ""
    lines = _first_nonempty_lines(raw_text, limit=80)
    agency_lines = _extract_agency_lines(lines)

    _, fallback_doc_number = _split_doc_id(fallback_doc_id)
    fallback_doc_number = fallback_doc_number or _extract_doc_number_only(fallback_doc_id)

    doc_type_from_number = _parse_doc_type_from_number(fallback_doc_number)
    codes_from_number = _parse_agency_codes_from_doc_number(fallback_doc_number)
    codes_from_lines = _parse_agency_codes_from_agency_lines(agency_lines)
    agency_codes = list(dict.fromkeys(codes_from_number + codes_from_lines))

    parsed_doc_type, parsed_title = _extract_doc_type_and_title_from_lines(
        lines,
        fallback_doc_type=doc_type_from_number,
    )

    doc_type = doc_type_from_number or parsed_doc_type
    title = _clean_title_text(parsed_title)

    return {
        "id": item_id,
        "doc_type": doc_type,
        "doc_number": fallback_doc_number,
        "title": title,
        "ministry": _join_agency_names(agency_codes),
        "agency_codes": agency_codes,
        "agencies": [AGENCY_REGISTRY[c]["name"] for c in agency_codes if c in AGENCY_REGISTRY],
        "raw_text": raw_text,
        "url": url,
    }