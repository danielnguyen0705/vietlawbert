#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_vietlawbench.py — Làm sạch bộ test VietLawBench và gán Global Hierarchical ID.

Mục tiêu (theo đề xuất testing của nhóm):
  1. Loại các testcase dẫn chiếu SAI: câu hỏi sinh từ preamble ("Căn cứ..."),
     điều khoản thi hành / chữ ký, hoặc ground_truth_article = "Điều khoản liên quan".
  2. Gán Global Hierarchical ID tách 4 cấp: doc_id > chapter_id > article_id > clause_id.
  3. Chuẩn hoá số hiệu văn bản (doc_norm) để chấm điểm bằng metric đơn giản equal/contain-any.
  4. Xuất: bộ sạch (.jsonl), 100 mẫu để kiểm tay (.csv), và báo cáo thống kê (.json).

Chạy:
  python3 clean_vietlawbench.py --in vietlawbench_1000.jsonl --outdir ./out [--sample 100]

Không bịa số liệu: mọi con số in ra đều đếm trực tiếp từ file đầu vào.
"""
import argparse, json, re, csv, unicodedata, random, collections, os

ROMAN = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}

def roman_to_int(s):
    s = s.strip().upper()
    if not s or any(c not in ROMAN for c in s):
        return None
    total, prev = 0, 0
    for c in reversed(s):
        v = ROMAN[c]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total

def norm_doc(doc):
    """Chuẩn hoá số hiệu VB: bỏ khoảng trắng, viết hoa. VD '17/2026/QĐ-UBND' -> '17/2026/QĐ-UBND'."""
    if not doc:
        return ""
    d = re.sub(r"\s+", "", str(doc)).upper()
    return d

def doc_core(doc_norm):
    """Phần 'số/năm' đặc trưng để contain-any, VD '17/2026'."""
    m = re.match(r"(\d+)/(\d{4})", doc_norm)
    return f"{m.group(1)}/{m.group(2)}" if m else doc_norm

def parse_chunk(cid):
    """Trả về (doc_head, article_id, clause_id, is_preamble)."""
    head = cid.split("_")[0]
    pre = "_preamble" in cid
    art = re.search(r"_art_(\d+)", cid)
    cl  = re.search(r"_cl_(\d+)", cid)
    return head, (f"art_{art.group(1)}" if art else None), (f"cl_{cl.group(1)}" if cl else None), pre

# --- Bộ nhận diện testcase hỏng ---
RE_THIHANH = re.compile(r"chịu trách nhiệm thi hành|TM\.\s*ỦY BAN|\(Đã ký\)|KT\.\s*CHỦ TỊCH|Nơi nhận", re.I)
RE_CANCU   = re.compile(r"^\s*Căn cứ ", re.I)
RE_DIEU    = re.compile(r"^Điều\s+\d")

def bad_reasons(r):
    reasons = []
    cid = r.get("ground_truth_chunk", "")
    art = r.get("ground_truth_article", "")
    ev  = r.get("evidence_text", "") or ""
    hl  = r.get("hierarchy_label", "") or ""
    if "_preamble" in cid:          reasons.append("preamble_chunk")
    if hl.strip().upper() == "CAN_CU": reasons.append("hierarchy_CAN_CU")
    if not RE_DIEU.match(art):       reasons.append("article_not_Dieu")   # vd "Điều khoản liên quan"
    if RE_CANCU.match(ev):           reasons.append("evidence_can_cu")
    if RE_THIHANH.search(ev):        reasons.append("evidence_thi_hanh_or_sign")
    return reasons

def chapter_from_label(hl):
    m = re.match(r"\s*Chương\s+([IVXLCDM]+)", hl or "", re.I)
    return roman_to_int(m.group(1)) if m else None

def build_global_id(doc_norm, chapter_id, article_id, clause_id):
    parts = [doc_norm or "UNKDOC"]
    if chapter_id: parts.append(f"ch{chapter_id}")
    if article_id: parts.append(article_id)
    if clause_id:  parts.append(clause_id)
    return "#".join(parts)

# --- Phát hiện câu hỏi template máy / cắt giữa từ (lỗi chất lượng câu hỏi) ---
_PRE = ["Theo quy định hiện hành, trường hợp ", "Theo quy định hiện hành, ",
        "Căn cứ pháp lý đối với nội dung ", "Quy định pháp luật về Căn cứ ",
        "Quy định pháp luật về "]
_SUF = [" được giải quyết như thế nào?", " thì thẩm quyền thuộc cơ quan nào?",
        " được hướng dẫn ra sao?", " được quy định như thế nào?",
        " được quy định ra sao?", " được áp dụng như thế nào?"]

def _ws(s):
    return re.sub(r"\s+", " ", (s or "").strip())

def flag_query(query, evidence):
    """Trả 'ok?' | 'template' | 'template+cắt-giữa-từ'."""
    q = _ws(query); evb = re.sub(r"^\s*\d+\.\s*", "", _ws(evidence))
    pre = next((p for p in _PRE if q.startswith(p)), None)
    suf = next((s for s in _SUF if q.endswith(s)), None)
    if not pre and not suf:
        return "ok?"
    core = q[len(pre):] if pre else q
    if suf: core = core[:-len(suf)]
    core = core.strip()
    if core and evb.startswith(core):
        nxt = evb[len(core):len(core)+1]
        if nxt and nxt.isalpha() and core[-1:].isalpha():
            return "template+cắt-giữa-từ"
    return "template"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", default="./out")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)

    recs = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    reason_counter = collections.Counter()
    clean, dropped = [], []

    for r in recs:
        rs = bad_reasons(r)
        for x in rs: reason_counter[x] += 1
        head, art_id, cl_id, pre = parse_chunk(r.get("ground_truth_chunk", ""))
        doc_norm = norm_doc(r.get("ground_truth_doc_number", ""))
        chap = chapter_from_label(r.get("hierarchy_label", ""))
        # fallback article_id từ trường ground_truth_article nếu chunk không có
        if not art_id:
            ma = re.match(r"Điều\s+(\d+)", r.get("ground_truth_article", "") or "")
            if ma: art_id = f"art_{ma.group(1)}"
        enriched = dict(r)
        enriched.update({
            "doc_norm": doc_norm,
            "doc_core": doc_core(doc_norm),
            "chapter_id": chap,
            "article_id": art_id,
            "clause_id": cl_id,
            "global_id": build_global_id(doc_norm, chap, art_id, cl_id),
            "retrieval_level": "clause" if cl_id else ("article" if art_id else "doc"),
        })
        if rs:
            enriched["_drop_reasons"] = rs
            dropped.append(enriched)
        else:
            clean.append(enriched)

    # Ghi bộ sạch
    clean_path = os.path.join(args.outdir, "vietlawbench_clean.jsonl")
    with open(clean_path, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dropped_path = os.path.join(args.outdir, "vietlawbench_dropped.jsonl")
    with open(dropped_path, "w", encoding="utf-8") as f:
        for r in dropped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 100 mẫu kiểm tay — phân tầng theo query_type, chỉ lấy từ bộ sạch
    by_type = collections.defaultdict(list)
    for r in clean: by_type[r["query_type"]].append(r)
    sample = []
    n_types = max(1, len(by_type))
    per = max(1, args.sample // n_types)
    for t, lst in by_type.items():
        random.shuffle(lst)
        sample += lst[:per]
    random.shuffle(sample)
    sample = sample[:args.sample]
    sample_path = os.path.join(args.outdir, f"vietlawbench_sample{args.sample}.csv")
    with open(sample_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark_id","query_type","query_issue","retrieval_level","doc_norm",
                    "article_id","clause_id","global_id","query","evidence_text",
                    "manual_ok(Y/N)","manual_note"])
        for r in sample:
            w.writerow([r["benchmark_id"], r["query_type"],
                        flag_query(r["query"], r.get("evidence_text","")),
                        r["retrieval_level"], r["doc_norm"], r.get("article_id"),
                        r.get("clause_id"), r["global_id"], _ws(r["query"]),
                        _ws(r.get("evidence_text",""))[:300], "", ""])

    report = {
        "input_total": len(recs),
        "kept_clean": len(clean),
        "dropped": len(dropped),
        "dropped_pct": round(100*len(dropped)/max(1,len(recs)), 2),
        "drop_reason_counts": dict(reason_counter),
        "clean_by_query_type": dict(collections.Counter(r["query_type"] for r in clean)),
        "clean_by_retrieval_level": dict(collections.Counter(r["retrieval_level"] for r in clean)),
        "clean_query_issue": dict(collections.Counter(flag_query(r["query"], r.get("evidence_text","")) for r in clean)),
        "clean_doc_id_format": dict(collections.Counter(
            ("numeric" if re.fullmatch(r"\d+", r["ground_truth_chunk"].split("_")[0]) else "uuid")
            for r in clean)),
        "sample_written": len(sample),
    }
    with open(os.path.join(args.outdir, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[outputs] {clean_path}\n          {dropped_path}\n          {sample_path}")

if __name__ == "__main__":
    main()
