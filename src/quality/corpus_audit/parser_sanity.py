# -*- coding: utf-8 -*-
"""
corpus_audit/parser_sanity.py
=============================
KIỂM BẤT BIẾN LỚP 1 cho AST parser trên DỮ LIỆU THẬT (chỉ đọc) — xác nhận parser ổn
định TRƯỚC khi chạy xuyên suốt pipeline trên cloud (CLAUDE.md §E, §H.2).

QUAN TRỌNG: văn bản pháp luật KHÔNG đồng nhất cấu trúc. Hiến pháp (Phần/Chương/Điều),
Bộ luật (Phần thứ.../Chương/Mục/Điều/Khoản), Luật/Nghị định/Thông tư (Điều phẳng),
và RẤT nhiều văn bản hành chính (Quyết định/Công văn) KHÔNG có Chương/Mục, có khi
không có cả Điều. Vì vậy mỗi văn bản được PHÂN TẦNG theo cấp cao nhất xuất hiện, và
bất biến được báo RIÊNG từng tầng — không áp một khuôn Điều-centric.

4 bất biến cấp-văn-bản (unit-test synthetic không phủ được ở quy mô kho thật):
  1) KHỚP SỐ ĐIỀU   : tập "Điều N" đếm bằng regex ĐỘC LẬP (trên văn bản parser thấy
                      sau _resplit_if_flat) == tập số Điều parser phát. (Văn bản không
                      có Điều: 0==0 → ĐẠT, hợp lệ.)
  2) PHỦ VĂN BẢN    : ký tự (bỏ trắng) trong [CONTENT] các chunk / ký tự nguồn ≥ 0.95
                      → không mất chữ (bắt lỗi parser bỏ dòng ngắn/phân mảnh).
  3) ĐƠN ĐIỆU ĐIỀU  : số Điều (phần nguyên) theo thứ tự chunk không giảm.
  4) KHÔNG NODE RỖNG : không chunk nào có thân [CONTENT] < 10 ký tự.

Tổng hợp tỉ lệ ĐẠT kèm Wilson CI 95% (tái dùng audit_corpus.wilson_ci), BỔ theo
content_source VÀ theo tầng cấu trúc; xuất ≤20 doc_id ví dụ mỗi loại vi phạm.

Tái sử dụng: preprocess.ast_parser.HybridASTParser, quality.crawl_audit.safe_read_jsonl,
audit_corpus.wilson_ci.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_corpus import wilson_ci  # noqa: E402

_RE_SRC_ARTICLE = re.compile(r"(?:^|\n)\s*(?:ĐIỀU|Điều)\s+(\d+[a-zA-Z]?)\b")
_RE_PART = re.compile(r"(?:^|\n)\s*(?:PHẦN|Phần)\s+(?:thứ\s+|[IVXLCDM\d])")
_RE_CHAPTER = re.compile(r"(?:^|\n)\s*(?:CHƯƠNG|Chương)\s+[IVXLCDM\d]")
_RE_SECTION = re.compile(r"(?:^|\n)\s*(?:MỤC|Mục)\s+\d")
_COVERAGE_MIN = 0.95
_EMPTY_BODY_CHARS = 10
_INVARIANTS = ("article_match", "coverage", "monotonic", "no_empty_chunk")

# Boilerplate hành chính ở đầu/cuối văn bản mà parser cố ý bỏ (text_cleaner.py / crawl_audit.py):
# quốc hiệu, tiêu ngữ, số hiệu, dòng ngày-tháng-năm, tên cơ quan đứng riêng, nơi nhận, chữ ký, gạch ngăn.
_RE_BOILERPLATE = re.compile(
    r"(?:CỘNG\s*H[ÒO]A\s*XÃ\s*HỘI\s*CHỦ\s*NGHĨA\s*VIỆT\s*NAM"
    r"|Độc\s*lập\s*[-–—]\s*Tự\s*do\s*[-–—]\s*Hạnh\s*phúc"
    r"|^\s*(?:Số|SỐ)\s*:\s*\S+"
    r"|ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}"
    r"|^\s*(?:Nơi\s*nhận|NƠI\s*NHẬN)\s*[:.]?"
    r"|^\s*(?:TM\.|KT\.|TL\.|T\s*/\s*M|K\s*/\s*T|PP\.|Q\.)\s*"
    r"|^\s*[-_*\s]*\(Đã\s*ký[^)]*\)"
    r"|^\s*(?:CHỦ\s*TỊCH|PHÓ\s*CHỦ\s*TỊCH|BỘ\s*TRƯỞNG|THỨ\s*TRƯỞNG|THỦ\s*TƯỚNG|PHÓ\s*THỦ\s*TƯỚNG"
    r"|CHÁNH\s*ÁN|VIỆN\s*TRƯỞNG|TỔNG\s*GIÁM\s*ĐỐC|GIÁM\s*ĐỐC|CỤC\s*TRƯỞNG|VỤ\s*TRƯỞNG)\b"
    r"|^\s*(?:ỦY\s*BAN\s*NHÂN\s*DÂN|UỶ\s*BAN\s*NHÂN\s*DÂN|HỘI\s*ĐỒNG\s*NHÂN\s*DÂN|CHÍNH\s*PHỦ"
    r"|QUỐC\s*HỘI|ỦY\s*BAN\s*THƯỜNG\s*VỤ\s*QUỐC\s*HỘI|THỦ\s*TƯỚNG\s*CHÍNH\s*PHỦ"
    r"|BỘ\s+[A-ZĐÀ-Ỹ\s,–-]+|TÒA\s*ÁN\s*NHÂN\s*DÂN|VIỆN\s*KIỂM\s*SÁT\s*NHÂN\s*DÂN|NGÂN\s*HÀNG\s*NHÀ\s*NƯỚC)\b"
    r"|^\s*[-_*=]{3,}\s*$"
    r"|^\s*(?:-\s*)?Lưu\s*:\s*VT"
    r")",
    re.IGNORECASE,
)
_RE_ART_LINE = re.compile(r"^\s*(?:ĐIỀU|Điều)\s+\d+[a-zA-Z]?\b")
_RE_HIER_IN_CONTENT = re.compile(r"\[HIERARCHY\](.*?)(?=\[CONTENT\]|$)", re.S)


def structural_class(resplit_text: str) -> str:
    """Tầng cấu trúc theo CẤP CAO NHẤT xuất hiện (phân tầng kiểm định, §E Lớp 2)."""
    if _RE_PART.search(resplit_text):
        return "co_phan"          # Bộ luật / Hiến pháp nhiều phần
    if _RE_SECTION.search(resplit_text):
        return "co_muc"           # Luật lớn: Chương + Mục
    if _RE_CHAPTER.search(resplit_text):
        return "co_chuong"        # có Chương, không Mục
    if _RE_SRC_ARTICLE.search(resplit_text):
        return "dieu_phang"       # Điều phẳng (Nghị định/Thông tư/Quyết định)
    return "khong_dieu"           # hành chính/prose: không có Điều


def _body_of(chunk_content: str) -> str:
    return chunk_content.split("[CONTENT]", 1)[-1] if "[CONTENT]" in chunk_content else chunk_content


def _hier_and_body_of(chunk: dict) -> str:
    """Ghép [HIERARCHY] + [CONTENT] của chunk để các tiêu đề Phần/Chương/Mục/Điều ngắn
    (được parser đưa vào HIERARCHY thay vì lặp lại ở CONTENT) không bị tính nhầm là 'mất chữ'."""
    raw = chunk.get("content", "") or ""
    h_explicit = chunk.get("hierarchy_path", "") or ""
    m_h = _RE_HIER_IN_CONTENT.search(raw)
    h_in = m_h.group(1).strip() if m_h else ""
    return f"{h_explicit} {h_in} {_body_of(raw)}"


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _norm_alnum(s: str) -> str:
    """Chuẩn hóa chỉ giữ chữ cái và chữ số (bỏ dấu câu/ngoặc) để khớp tiêu đề 'Điều 2. Tiêu đề'
    với '[HIERARCHY] Điều 2 (Tiêu đề)'."""
    return re.sub(r"[\W_]+", "", s or "", flags=re.UNICODE).lower()


def _int_prefix(num: str) -> int:
    m = re.match(r"\d+", str(num) or "")
    return int(m.group(0)) if m else -1


def _block_of(chunk: dict) -> str:
    """Khối phụ lục đính kèm của chunk (vd '_b2'); '' nếu thuộc thân chính. Phụ lục
    'ban hành kèm theo' đánh lại số Điều → đơn điệu phải xét TRONG từng khối."""
    m = re.search(r"(_b\d+)_art_", str(chunk.get("article_id") or chunk.get("chunk_id") or ""))
    return m.group(1) if m else ""


def is_boilerplate_line(line: str) -> bool:
    """Kiểm tra dòng có phải boilerplate hành chính (quốc hiệu, tiêu ngữ, ngày ký, nơi nhận, chữ ký)."""
    return bool(_RE_BOILERPLATE.search((line or "").strip()))


def _coverage_raw(resplit_text: str, body: str, min_line: int = 25) -> tuple[float, int]:
    """Sàn thận trọng (chưa lọc boilerplate): tính mọi dòng ≥min_line ký tự trên chỉ [CONTENT]."""
    lines = [_norm(l) for l in resplit_text.splitlines() if len(l.strip()) >= min_line]
    lines = [l for l in lines if l]
    if not lines:
        return 1.0, 0
    tot = sum(len(l) for l in lines)
    cov = sum(len(l) for l in lines if l in body)
    missing = sum(1 for l in lines if l not in body)
    return round(cov / tot, 4), missing


def _coverage(resplit_text: str, full_chunk_text: str, min_line: int = 25) -> tuple[float, int]:
    """Line-recall thực chất: trong các DÒNG nguồn nội dung (≥min_line ký tự, đã loại
    boilerplate quốc hiệu/tiêu ngữ/ngày ký/nơi nhận/chữ ký và các mảnh tiền văn ≤40 ký tự
    trước Điều 1 mà parser chủ đích bỏ theo ngưỡng >40), bao nhiêu phần xuất hiện trong
    chunk ([HIERARCHY] + [CONTENT]). Trả (tỉ lệ, số dòng mất)."""
    raw_lines = [l.strip() for l in resplit_text.splitlines() if l.strip()]
    if not raw_lines:
        return 1.0, 0
    has_long_line = any(len(l) > 40 for l in raw_lines)
    seen_art = False
    substantive: list[str] = []
    for s in raw_lines:
        if _RE_ART_LINE.match(s):
            seen_art = True
        if is_boilerplate_line(s):
            continue
        # Trước Điều đầu tiên: HybridASTParser (ast_parser.py:190) chỉ giữ dòng > 40 ký tự
        if not seen_art and has_long_line and len(s) <= 40:
            continue
        if len(s) < min_line:
            continue
        nl = _norm_alnum(s)
        if nl:
            substantive.append(nl)
    if not substantive:
        return 1.0, 0
    blob = _norm_alnum(full_chunk_text)
    tot = sum(len(nl) for nl in substantive)
    cov = sum(len(nl) for nl in substantive if nl in blob)
    missing = sum(1 for nl in substantive if nl not in blob)
    return round(cov / tot, 4), missing


def check_doc(chunks: list[dict], resplit_text: str) -> dict:
    """Đo 4 bất biến Lớp 1 cho MỘT văn bản (parser đã chạy ở caller). HÀM THUẦN.
    resplit_text = văn bản parser thực sự thấy (sau _resplit_if_flat)."""
    src_arts = set(_RE_SRC_ARTICLE.findall(resplit_text))
    parser_arts = {str(c["article_num"]) for c in chunks if c.get("article_num")}

    body = _norm(" ".join(_body_of(c.get("content", "")) for c in chunks))
    full_chunk_text = " ".join(_hier_and_body_of(c) for c in chunks)
    coverage, missing_lines = _coverage(resplit_text, full_chunk_text)
    coverage_raw, missing_lines_raw = _coverage_raw(resplit_text, body)

    seq_by_block: dict[str, list[int]] = {}
    for c in chunks:
        if c.get("article_num"):
            seq_by_block.setdefault(_block_of(c), []).append(_int_prefix(c["article_num"]))
    monotonic = all(s[i] <= s[i + 1] for s in seq_by_block.values() for i in range(len(s) - 1))

    empty = any(len(_norm(_body_of(c.get("content", "")))) < _EMPTY_BODY_CHARS for c in chunks)
    return {
        "n_src_articles": len(src_arts),
        "n_parser_articles": len(parser_arts),
        "article_match": src_arts == parser_arts,
        "article_delta": len(src_arts) - len(parser_arts),
        "coverage": coverage,
        "coverage_ok": coverage >= _COVERAGE_MIN,
        "missing_lines": missing_lines,
        "coverage_raw": coverage_raw,
        "coverage_raw_ok": coverage_raw >= _COVERAGE_MIN,
        "missing_lines_raw": missing_lines_raw,
        "monotonic": monotonic,
        "no_empty_chunk": not empty,
        "structural_class": structural_class(resplit_text),
        "n_chunks": len(chunks),
    }


def _ok(r: dict, inv: str) -> bool:
    return {"article_match": r["article_match"], "coverage": r["coverage_ok"],
            "monotonic": r["monotonic"], "no_empty_chunk": r["no_empty_chunk"]}[inv]


def _load_ids(ids_path: Optional[str]) -> Optional[set]:
    if not ids_path:
        return None
    ids = set()
    with open(ids_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line).get("doc_id"))
    return ids


def run(shards_dir: str, out_dir: str, ids_path: Optional[str] = None,
        limit: Optional[int] = None, git_head: str = "?") -> dict:
    from quality.crawl_audit import safe_read_jsonl
    from preprocess.ast_parser import HybridASTParser
    os.makedirs(out_dir, exist_ok=True)
    parser = HybridASTParser()
    ids = _load_ids(ids_path)
    shards = sorted(glob.glob(os.path.join(shards_dir, "crawl_pages_*.jsonl.gz")))
    if limit:
        shards = shards[:limit]

    n = 0
    no_chunk = 0
    no_chunk_with_text = 0
    cov_sum = 0.0
    cov_raw_sum = 0.0
    cov_raw_pass = 0
    passed = Counter()
    by_cs = defaultdict(Counter); cs_total = Counter()
    by_sc = defaultdict(Counter); sc_total = Counter()
    cov_sum_sc = defaultdict(float)
    fail = defaultdict(list)

    for sh in shards:
        for raw in safe_read_jsonl(sh):
            did = raw.get("doc_id") or raw.get("item_id")
            if ids is not None and did not in ids:
                continue
            text = raw.get("full_text") or raw.get("text") or ""
            cs = raw.get("content_source") or "(null)"
            resplit = parser._resplit_if_flat(text)
            meta = {k: raw.get(k) for k in ("doc_id", "item_id", "doc_number", "title",
                    "effective_date", "status", "co_quan", "org")}
            chunks = parser.parse_document(text, meta)
            n += 1
            cs_total[cs] += 1
            if not chunks:
                no_chunk += 1
                if len(_norm(text)) >= 200:        # có nội dung thật mà parser không sinh chunk = lỗi
                    no_chunk_with_text += 1
            r = check_doc(chunks, resplit)
            sc = r["structural_class"]
            sc_total[sc] += 1
            cov_sum += r["coverage"]; cov_sum_sc[sc] += r["coverage"]
            cov_raw_sum += r["coverage_raw"]
            if r["coverage_raw_ok"]:
                cov_raw_pass += 1
            for inv in _INVARIANTS:
                if _ok(r, inv):
                    passed[inv] += 1; by_cs[cs][inv] += 1; by_sc[sc][inv] += 1
                elif len(fail[inv]) < 20:
                    fail[inv].append(did)

    summary = {
        "total_docs": n,
        "docs_without_chunks": no_chunk,
        "docs_without_chunks_with_text": no_chunk_with_text,
        "avg_coverage": round(cov_sum / n, 4) if n else 0.0,
        "avg_coverage_raw": round(cov_raw_sum / n, 4) if n else 0.0,
        "coverage_raw_invariant": {
            "pass": cov_raw_pass,
            **dict(zip(("rate_pct", "ci95_lo_pct", "ci95_hi_pct"), wilson_ci(cov_raw_pass, n))),
        },
        "invariants": {inv: {"pass": passed[inv],
                       **dict(zip(("rate_pct", "ci95_lo_pct", "ci95_hi_pct"), wilson_ci(passed[inv], n)))}
                       for inv in _INVARIANTS},
        "by_structural_class": {sc: {"docs": sc_total[sc],
                                "avg_coverage": round(cov_sum_sc[sc] / sc_total[sc], 4),
                                **{inv: by_sc[sc][inv] for inv in _INVARIANTS}}
                                for sc in sc_total},
        "by_content_source": {cs: {"docs": cs_total[cs],
                              **{inv: by_cs[cs][inv] for inv in _INVARIANTS}} for cs in cs_total},
        "fail_sample_doc_ids": {inv: fail[inv] for inv in _INVARIANTS},
    }
    env = {"date": date.today().isoformat(), "shards_dir": shards_dir,
           "shards": len(shards), "ids_path": ids_path, "git_head": git_head}
    stamp = date.today().strftime("%Y%m%d")
    with open(os.path.join(out_dir, f"parser_sanity_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "env": env}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"parser_sanity_{stamp}.md"), "w", encoding="utf-8") as f:
        f.write(_md(summary, env))
    return {"summary": summary, "env": env}


_LABEL = {"article_match": "Khớp số Điều", "coverage": "Phủ VB ≥0.95",
          "monotonic": "Đơn điệu Điều", "no_empty_chunk": "Ko node rỗng"}
_SC_LABEL = {"co_phan": "Có Phần (Bộ luật/Hiến pháp)", "co_muc": "Có Mục (Luật lớn)",
             "co_chuong": "Có Chương", "dieu_phang": "Điều phẳng (NĐ/TT/QĐ)",
             "khong_dieu": "Không có Điều (hành chính)"}


def _md(s: dict, env: dict) -> str:
    n = s["total_docs"]
    L = [f"# Kiểm bất biến Lớp 1 — AST parser trên dữ liệu thật — {env['date']}\n",
         "## Môi trường",
         f"- Nguồn: `{env['shards_dir']}` (chỉ đọc); mẫu: "
         + (f"`{os.path.basename(env['ids_path'])}`" if env.get("ids_path") else "toàn bộ shard"),
         f"- Văn bản kiểm: {n} | không sinh chunk: {s['docs_without_chunks']} | phủ VB trung bình: {s['avg_coverage']}",
         f"- Git HEAD: {env.get('git_head','?')}\n",
         "## Tỉ lệ ĐẠT từng bất biến (toàn mẫu, Wilson CI 95%)",
         "| Bất biến | ĐẠT | % | CI95% |", "|---|---|---|---|"]
    for inv, d in s["invariants"].items():
        L.append(f"| {_LABEL[inv]} | {d['pass']}/{n} | {d['rate_pct']} | [{d['ci95_lo_pct']}, {d['ci95_hi_pct']}] |")
    if "coverage_raw_invariant" in s:
        cr = s["coverage_raw_invariant"]
        L.append(f"\n> Ghi chú đối chứng: Sàn thận trọng chưa lọc boilerplate hành chính (`coverage_raw ≥ 0.95` chỉ trên `[CONTENT]`): "
                 f"{cr['pass']}/{n} ({cr['rate_pct']}% CI[{cr['ci95_lo_pct']}, {cr['ci95_hi_pct']}], TB={s.get('avg_coverage_raw', 0)}). "
                 f"Sau khi loại quốc hiệu/tiêu ngữ/ngày ký/chữ ký và tính cả tiêu đề trong `[HIERARCHY]`, tỉ lệ phủ thực chất là **{s['invariants']['coverage']['rate_pct']}%**.")
    L.append("\n## Theo TẦNG CẤU TRÚC (số ĐẠT / tổng tầng) — văn bản không đồng khuôn")
    L.append("| Tầng cấu trúc | docs | phủ TB | khớp Điều | phủ VB | đơn điệu | ko rỗng |")
    L.append("|---|---|---|---|---|---|---|")
    for sc, d in sorted(s["by_structural_class"].items(), key=lambda kv: -kv[1]["docs"]):
        L.append(f"| {_SC_LABEL.get(sc, sc)} | {d['docs']} | {d['avg_coverage']} | {d['article_match']} "
                 f"| {d['coverage']} | {d['monotonic']} | {d['no_empty_chunk']} |")
    L.append("\n## Theo nguồn (content_source) — số ĐẠT / tổng")
    L.append("| content_source | docs | khớp Điều | phủ VB | đơn điệu | ko rỗng |")
    L.append("|---|---|---|---|---|---|")
    for cs, d in sorted(s["by_content_source"].items(), key=lambda kv: -kv[1]["docs"]):
        L.append(f"| {cs} | {d['docs']} | {d['article_match']} | {d['coverage']} | {d['monotonic']} | {d['no_empty_chunk']} |")
    L.append("\n## Mẫu doc_id vi phạm (≤20 mỗi loại, để soi tay)")
    for inv, ids in s["fail_sample_doc_ids"].items():
        L.append(f"- **{_LABEL[inv]}** ({len(ids)}): {', '.join(ids[:20]) if ids else '—'}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    from configs.paths import RAW_SHARDS_DIR, REPORTS_DIR
    ap = argparse.ArgumentParser(description="Kiểm bất biến Lớp 1 của AST parser (chỉ đọc).")
    ap.add_argument("--shards-dir", default=str(RAW_SHARDS_DIR))
    ap.add_argument("--ids", default=str(REPORTS_DIR / "pilot_sample_5000_ids.jsonl"),
                    help="JSONL doc_id giới hạn mẫu (mặc định mẫu pilot 5000); '' = toàn bộ")
    ap.add_argument("--out", default=str(REPORTS_DIR))
    ap.add_argument("--limit", type=int, default=None, help="số shard tối đa")
    a = ap.parse_args()
    head = "?"
    try:
        import subprocess
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        pass
    out = run(a.shards_dir, a.out, a.ids or None, a.limit, head)
    s = out["summary"]
    print(f"Văn bản kiểm: {s['total_docs']} | phủ TB: {s['avg_coverage']}")
    for inv, d in s["invariants"].items():
        print(f"  {inv:16s}: {d['pass']}/{s['total_docs']} ({d['rate_pct']}% CI[{d['ci95_lo_pct']},{d['ci95_hi_pct']}])")
    print(f"Báo cáo (có bảng theo tầng cấu trúc): {a.out}/parser_sanity_*.{{json,md}}")
