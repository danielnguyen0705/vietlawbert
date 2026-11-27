"""
benchmark/build_multihop_subset.py — Sinh subset Multi-hop từ đồ thị HIN (thiết kế duyệt 2026-11-27).

Hai bucket: TEMPORAL (THAY_THE/SUA_DOI_BO_SUNG) và REFERENCE (DAN_CHIEU/QUY_DINH_CHI_TIET/HUONG_DAN).
5 cổng chất lượng:
  0. Zero-leakage: loại cặp (doc_A, doc_B) đã có trong hin_triplets.parquet (anchor,positive).
  1. Lọc chunk [META] + điều thủ tục (hiệu lực/tổ chức/trách nhiệm thi hành).
  2. Ghép chunk theo tương đồng từ vựng (Jaccard) → cùng đối tượng điều chỉnh.
  3. Sinh câu CẤM lộ số hiệu văn bản (ép truy xuất lần theo đồ thị).
  4. Ablation: câu phải CẦN CẢ A và B (một mình không đủ) + faithfulness.
Xuất JSONL khớp schema VLQA + relevant_chunks/bucket/relation_type.

Generator ≠ Verifier (chống self-preference). Chạy qua hhtechapi (OpenAI-compatible), KHÔNG cần GPU.
"""
from __future__ import annotations

import os
import re
import csv
import json
import time
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, FrozenSet

# ───────────────────────── Pure helpers (có unit test) ─────────────────────────

_PROC_PAT = re.compile(r"(hiệu lực thi hành|tổ chức thực hiện|tổ chức thi hành|trách nhiệm thi hành|"
                       r"trách nhiệm thực hiện|điều khoản thi hành|điều khoản chuyển tiếp|quy định chuyển tiếp)",
                       re.IGNORECASE)
_WORD = re.compile(r"[a-zà-ỹ0-9]+", re.IGNORECASE)
_STOP = {"và", "của", "các", "được", "theo", "tại", "cho", "có", "là", "những", "này", "đó",
         "về", "với", "trong", "hoặc", "khi", "để", "từ", "đến", "sau", "như", "một"}


def doc_of_chunk(chunk_id: str) -> str:
    """UUID văn bản từ chunk_id '{uuid}_art_N' / '{uuid}_art_N_cl_M'."""
    return str(chunk_id or "").split("_art_")[0]


def load_leakage_pairs(parquet_path: str | Path, neo4j_client=None) -> Tuple[Set[FrozenSet[str]], Dict[str, Any]]:
    """Tập cặp doc đã dùng làm (anchor, positive) trong tập train — để chống rò rỉ.
    Có Neo4j → quy về DOC_NUMBER (bền qua các lần rebuild đổi doc_id). Trả (pairs, info);
    info['stale']=True khi parquet không khớp graph hiện tại (ví dụ triplet mined trên graph cũ)."""
    import pandas as pd
    p = Path(parquet_path)
    if not p.exists():
        return set(), {"stale": True, "reason": "no_parquet", "n_pairs": 0}
    df = pd.read_parquet(p, columns=["anchor_id", "positive_id"])
    raw: Set[Tuple[str, str]] = set()
    docs: Set[str] = set()
    for a, b in zip(df["anchor_id"].astype(str), df["positive_id"].astype(str)):
        da, db = doc_of_chunk(a), doc_of_chunk(b)
        if da and db and da != db:
            raw.add((da, db)); docs.add(da); docs.add(db)
    if neo4j_client is None:
        return {frozenset(x) for x in raw}, {"keyed_by": "doc_id", "n_pairs": len(raw)}
    ids = list(docs)
    id2dn: Dict[str, str] = {}
    for i in range(0, len(ids), 5000):
        for rec in neo4j_client.execute_query(
            "MATCH (d:LawDocument) WHERE d.doc_id IN $ids RETURN d.doc_id AS id, d.doc_number AS dn",
            {"ids": ids[i:i + 5000]}):
            if rec["dn"]:
                id2dn[rec["id"]] = norm_docnum(rec["dn"])
    resolved = len(id2dn) / max(len(docs), 1)
    pairs = {frozenset((id2dn[a], id2dn[b])) for a, b in raw if a in id2dn and b in id2dn}
    return pairs, {"keyed_by": "doc_number", "resolved_ratio": round(resolved, 3),
                   "n_pairs": len(pairs), "stale": resolved < 0.05}


def is_meta(text: str) -> bool:
    return str(text or "").lstrip().startswith("[META]")


def strip_headers(text: str) -> str:
    """Cắt hai dòng header dán trước mỗi chunk ('[META] …' và '[HIERARCHY] …'), giữ phần thân điều.
    Bỏ META cũng loại luôn số hiệu văn bản khỏi đầu vào generator (chống lộ từ gốc)."""
    t = str(text or "")
    t = re.sub(r"^\s*\[META\][^\n]*\n?", "", t)
    t = re.sub(r"^\s*\[HIERARCHY\][^\n]*\n?", "", t)
    t = re.sub(r"^\s*\[CONTENT\]\s*", "", t)
    return t.strip()


def norm_docnum(dn: str) -> str:
    """Chuẩn hóa số hiệu: cắt ký tự rác đầu (dấu nháy/chấm thấy khi crawl) rồi dùng chuẩn hóa toàn cục."""
    dn = re.sub(r"^[^0-9A-Za-zĐ]+", "", str(dn or "").strip())
    try:
        from configs.global_id import normalize_doc_number
        return normalize_doc_number(dn)
    except Exception:
        return dn


def is_procedural(text: str) -> bool:
    """Điều hành chính/thủ tục (thi hành, tổ chức thực hiện, chuyển tiếp) — không có quy phạm thực chất."""
    return bool(_PROC_PAT.search(str(text or "")))


def hierarchy_title(raw: str) -> str:
    """Tiêu đề điều lấy từ dòng [HIERARCHY] (vd 'Điều 20 (Hiệu lực thi hành)')."""
    m = re.search(r"\[HIERARCHY\]([^\n]*)", str(raw or ""))
    return m.group(1).strip() if m else ""


def _tokens(s: str) -> Set[str]:
    return {t for t in _WORD.findall(str(s or "").lower()) if len(t) > 1 and t not in _STOP}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / len(ta | tb)


def pick_best_chunk_pair(a_chunks: List[Dict[str, str]], b_chunks: List[Dict[str, str]]
                         ) -> Optional[Tuple[Dict[str, str], Dict[str, str], float]]:
    """Chọn cặp (chunk_A, chunk_B) cùng đối tượng điều chỉnh nhất theo Jaccard từ vựng."""
    best = None
    best_s = -1.0
    for ca in a_chunks:
        for cb in b_chunks:
            s = jaccard(ca["content"], cb["content"])
            if s > best_s:
                best_s, best = s, (ca, cb, s)
    return best


def leaks_doc_number(question: str, *doc_numbers: str) -> bool:
    """Câu hỏi có lộ số hiệu văn bản không (làm BM25 bắt trúng, vô hiệu hóa vai trò đồ thị)."""
    q = str(question or "").lower()
    for dn in doc_numbers:
        dn = str(dn or "").strip().lower()
        if not dn or dn == "n/a":
            continue
        if dn in q:
            return True
        m = re.search(r"(\d+\s*/\s*\d{4})", dn)   # lõi "số/năm" ví dụ 10/2023
        if m and m.group(1).replace(" ", "") in q.replace(" ", ""):
            return True
    return False


def clean_content(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


# ───────────────────────── LLM (hhtechapi, OpenAI-compatible) ─────────────────────────

BUCKETS = {
    "TEMPORAL": ["SUA_DOI_BO_SUNG", "THAY_THE"],              # ưu tiên quan hệ buộc ghép A+B (findings điểm 2)
    "REFERENCE": ["DAN_CHIEU", "QUY_DINH_CHI_TIET_HUONG_DAN", "HUONG_DAN_AP_DUNG"],
}

MIN_JACCARD = 0.08   # ngưỡng tương đồng tối thiểu A↔B — dưới ngưỡng là ghép nhầm chủ đề (findings điểm 3)

_GEN_SYS = (
    "Bạn là chuyên gia pháp lý Việt Nam soạn câu hỏi đánh giá hệ thống truy xuất. "
    "Bạn nhận HAI đoạn văn bản luật A và B (quan hệ '{rel}') kèm số hiệu của chúng. "
    "Tạo MỘT câu hỏi tình huống nghiệp vụ thực tế mà để trả lời ĐÚNG và ĐỦ thì BẮT BUỘC phải dùng CẢ A lẫn B. "
    "Câu hỏi phải hỏi về NỘI DUNG THỰC CHẤT (quyền, nghĩa vụ, điều kiện, mức, trình tự nghiệp vụ); "
    "KHÔNG hỏi về vị trí điều khoản ('nằm ở khoản/điều nào'), KHÔNG hỏi về trách nhiệm/tổ chức thi hành hành chính. "
    "TUYỆT ĐỐI KHÔNG nêu số hiệu hay tên bất kỳ văn bản nào trong CÂU HỎI. "
    "ĐÁP ÁN viết như luật sư trả lời: trích dẫn cụ thể Điều và SỐ HIỆU văn bản thật "
    "(ví dụ 'Theo Điều 5 Thông tư 03/2008/TT-NHNN...'); KHÔNG dùng nhãn 'Văn bản A/B', KHÔNG viết 'cần dùng cả hai đoạn'. "
    "Trả về DUY NHẤT JSON: {{\"question\":\"...\",\"answer\":\"...\"}}"
)
_GEN_USER = "VĂN BẢN A (số hiệu {a_dn}):\n{a}\n\nVĂN BẢN B (số hiệu {b_dn}):\n{b}"

_NEC_SYS = ("Chỉ với DUY NHẤT đoạn văn bản dưới đây, có thể trả lời ĐẦY ĐỦ và CHÍNH XÁC câu hỏi không? "
            "Trả về DUY NHẤT JSON {\"du\":true} hoặc {\"du\":false}.")
_NEC_USER = "CÂU HỎI: {q}\n\nĐOẠN VĂN BẢN:\n{t}"

_FAITH_SYS = ("Đáp án dưới đây có hoàn toàn dựa trên nội dung hai đoạn văn bản (không bịa thông tin ngoài) không? "
              "Trả về DUY NHẤT JSON {\"trung_thuc\":true} hoặc {\"trung_thuc\":false}.")
_FAITH_USER = "CÂU HỎI: {q}\nĐÁP ÁN: {ans}\n\nA:\n{a}\n\nB:\n{b}"


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = re.sub(r"^```(?:json)?|```$", "", str(raw or "").strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def _chat(client, model: str, system: str, user: str, retries: int = 2) -> str:
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.2 if i == 0 else 0.0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return r.choices[0].message.content or ""
        except Exception as exc:
            if i == retries - 1:
                return f'{{"error":"{type(exc).__name__}"}}'
            time.sleep(1.5)
    return ""


# ───────────────────────── Graph fetch ─────────────────────────

def fetch_doc_pairs(client, relation_types: List[str], limit: int) -> List[Dict[str, str]]:
    q = """
    MATCH (a:LawDocument)-[r:LEGAL_RELATION]->(b:LawDocument)
    WHERE toUpper(r.type) IN $types
      AND a.doc_id IS NOT NULL AND b.doc_id IS NOT NULL
      AND a.doc_number IS NOT NULL AND b.doc_number IS NOT NULL
      AND a.doc_number <> 'N/A' AND b.doc_number <> 'N/A'
      AND EXISTS { (a)-[:HAS_CHUNK]->(:Chunk) } AND EXISTS { (b)-[:HAS_CHUNK]->(:Chunk) }
    RETURN DISTINCT a.doc_id AS a_id, a.doc_number AS a_dn,
                    b.doc_id AS b_id, b.doc_number AS b_dn, toUpper(r.type) AS rel
    LIMIT $limit
    """
    return [dict(rec) for rec in client.execute_query(q, {"types": relation_types, "limit": limit})]


def fetch_content_chunks(client, doc_id: str, cap: int = 30) -> List[Dict[str, str]]:
    q = """
    MATCH (d:LawDocument {doc_id:$did})-[:HAS_CHUNK]->(c:Chunk)
    WHERE c.content IS NOT NULL AND size(c.content) > 550
    RETURN c.chunk_id AS cid, c.content AS content LIMIT $cap
    """
    out = []
    for rec in client.execute_query(q, {"did": doc_id, "cap": cap}):
        title = hierarchy_title(rec["content"])   # tiêu đề điều — tín hiệu thủ tục chính xác nhất
        body = strip_headers(rec["content"])      # cắt header [META]/[HIERARCHY], giữ thân điều
        if len(body) < 120 or is_procedural(title) or is_procedural(body[:220]):
            continue
        out.append({"chunk_id": rec["cid"], "content": clean_content(body)[:1200]})
    return out


# ───────────────────────── Orchestrate ─────────────────────────

def build(bucket: str, target: int, gen_model: str, ver_model: str, out_path: str,
          triplet_path: str, neo4j_client, gen_client, ver_client,
          fetch_mult: int = 4, seed: int = 42) -> Dict[str, Any]:
    random.seed(seed)
    rel_types = BUCKETS[bucket]
    leak, leak_info = load_leakage_pairs(triplet_path, neo4j_client)
    if leak_info.get("stale"):
        print(f"[CẢNH BÁO] Cổng 0 RỖNG — parquet train không khớp graph hiện tại ({leak_info}). "
              f"Phải re-mine triplet trên graph hiện tại rồi SINH LẠI để chống rò rỉ thật.", flush=True)
    else:
        print(f"[Cổng 0] {leak_info.get('n_pairs',0)} cặp train để loại ({leak_info}).", flush=True)
    pairs = fetch_doc_pairs(neo4j_client, rel_types, limit=target * fetch_mult * 3)
    random.shuffle(pairs)

    kept, stats = [], {"xet": 0, "leak_gate0": 0, "no_chunk": 0, "topic_mismatch": 0,
                       "leak_docnum": 0, "single_hop": 0, "unfaithful": 0, "gen_err": 0}
    seen_pairs: Set[FrozenSet[str]] = set()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w", encoding="utf-8")   # ghi tăng dần, flush mỗi câu — không mất nếu bị ngắt

    for p in pairs:
        if len(kept) >= target:
            break
        key = frozenset((p["a_id"], p["b_id"]))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        stats["xet"] += 1

        if frozenset((norm_docnum(p["a_dn"]), norm_docnum(p["b_dn"]))) in leak:   # Cổng 0 (theo doc_number)
            stats["leak_gate0"] += 1
            continue
        a_ch = fetch_content_chunks(neo4j_client, p["a_id"])
        b_ch = fetch_content_chunks(neo4j_client, p["b_id"])
        if not a_ch or not b_ch:                          # Cổng 1
            stats["no_chunk"] += 1
            continue
        best = pick_best_chunk_pair(a_ch, b_ch)           # Cổng 2
        if not best:
            stats["no_chunk"] += 1
            continue
        if best[2] < MIN_JACCARD:                         # ghép nhầm chủ đề → bỏ
            stats["topic_mismatch"] += 1
            continue
        ca, cb, _ = best

        gen = _parse_json(_chat(gen_client, gen_model, _GEN_SYS.format(rel=p["rel"]),
                                _GEN_USER.format(a=ca["content"], b=cb["content"], a_dn=p["a_dn"], b_dn=p["b_dn"])))
        q, ans = gen.get("question", ""), gen.get("answer", "")
        if not q or not ans:
            stats["gen_err"] += 1
            continue
        if leaks_doc_number(q, p["a_dn"], p["b_dn"]):     # Cổng 3
            stats["leak_docnum"] += 1
            continue
        # Cổng 4a: ablation — câu phải CẦN CẢ HAI
        du_a = _parse_json(_chat(ver_client, ver_model, _NEC_SYS, _NEC_USER.format(q=q, t=ca["content"]))).get("du")
        du_b = _parse_json(_chat(ver_client, ver_model, _NEC_SYS, _NEC_USER.format(q=q, t=cb["content"]))).get("du")
        if du_a is True or du_b is True:
            stats["single_hop"] += 1
            continue
        # Cổng 4b: faithfulness
        faith = _parse_json(_chat(ver_client, ver_model, _FAITH_SYS,
                                  _FAITH_USER.format(q=q, ans=ans, a=ca["content"], b=cb["content"]))).get("trung_thuc")
        if faith is False:
            stats["unfaithful"] += 1
            continue

        item = {
            "qid": f"MH_{bucket[:4]}_{len(kept):04d}",
            "question": q.strip(),
            "relevant_laws": [norm_docnum(p["a_dn"]), norm_docnum(p["b_dn"])],
            "answer": ans.strip(),
            "relevant_chunks": [ca["chunk_id"], cb["chunk_id"]],
            "bucket": bucket,
            "relation_type": p["rel"],
        }
        kept.append(item)
        fout.write(json.dumps(item, ensure_ascii=False) + "\n"); fout.flush()

    fout.close()
    stats["giu_lai"] = len(kept)
    return stats


def _make_client(base_env: str, key_env: str):
    from configs.config import config
    from openai import OpenAI
    base = (getattr(config, base_env, None) or os.getenv(base_env) or "https://hhtechapi.net/v1").rstrip("/") + "/"
    key = getattr(config, key_env, None) or os.getenv(key_env) or ""
    return OpenAI(base_url=base, api_key=key or "dummy")


def main():
    ap = argparse.ArgumentParser(description="Sinh subset Multi-hop từ đồ thị HIN")
    ap.add_argument("--bucket", choices=["TEMPORAL", "REFERENCE"], required=True)
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--gen-model", default="claude-sonnet-4.6")
    ap.add_argument("--ver-model", default="gemini-3.6-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--triplets", default="outputs/artifacts/triplets/hin_triplets.parquet")
    args = ap.parse_args()

    from database.neo4j_client import Neo4jClient
    neo = Neo4jClient()
    gen_client = _make_client("EVAL_API_BASE", "EVAL_API_KEY")
    ver_client = _make_client("EVAL_API_BASE", "EVAL_API_KEY")
    try:
        stats = build(args.bucket, args.target, args.gen_model, args.ver_model, args.out,
                      args.triplets, neo, gen_client, ver_client)
    finally:
        neo.close()
    print("THỐNG KÊ:", json.dumps(stats, ensure_ascii=False))
    print("OUT:", args.out)


if __name__ == "__main__":
    main()
