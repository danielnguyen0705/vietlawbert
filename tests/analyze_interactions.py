"""
tests/analyze_interactions.py — Phân tích nhật ký hỏi-đáp (interactions.jsonl)
=============================================================================
Đọc flagging/interactions.jsonl (do demo/server/interaction_service.py ghi) và
in bảng thống kê định lượng phục vụ phân tích RQ1–RQ4:

  - Độ trễ truy xuất / sinh / tổng (trung bình, p50, p95) — theo từng mode
  - Tỷ lệ đạt SLA truy xuất < 500ms (RQ1) và E2E < 8000ms
  - Attribution score trung bình (RQ4 — Faithfulness)
  - Điểm rerank của bằng chứng top-1 (chất lượng truy xuất)
  - Danh sách truy vấn attribution thấp (< ngưỡng) để soi thủ công

Thuần stdlib — không cần cài thêm. Chạy:
    python tests/analyze_interactions.py
    python tests/analyze_interactions.py --file flagging/interactions.jsonl --attr-threshold 0.5
    python tests/analyze_interactions.py --json        # xuất JSON tổng hợp (để pipeline khác dùng)

Lưu ý latency: KHÔNG dùng để quyết định pass/fail (xem triết lý trong retrieval_metrics.js).
Script này chỉ BÁO CÁO. Trên máy trạm CPU (bge-m3 chạy CPU) latency truy xuất sẽ cao — bình thường.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO_ROOT / "flagging" / "interactions.jsonl"


# ─── Helpers thống kê (stdlib) ──────────────────────────────────────────────

def _percentile(sorted_vals: List[float], p: float) -> float:
    """Phân vị tuyến tính (p trong [0,1]). sorted_vals phải đã sắp xếp tăng dần."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _pass_rate(vals: List[float], threshold: float, below: bool = True) -> float:
    """Tỷ lệ phần tử đạt ngưỡng (below=True → đếm < threshold)."""
    if not vals:
        return 0.0
    ok = sum(1 for v in vals if (v < threshold if below else v >= threshold))
    return 100.0 * ok / len(vals)


# ─── Nạp log ────────────────────────────────────────────────────────────────

def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[!] Không tìm thấy log: {path}", file=sys.stderr)
        print("    Hãy chạy demo và đặt vài câu hỏi trước (USE_MOCK_BACKEND=false python -m demo.app).",
              file=sys.stderr)
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[!] Dòng {ln} hỏng JSON, bỏ qua.", file=sys.stderr)
    return records


# ─── Tổng hợp ───────────────────────────────────────────────────────────────

def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gom nhóm theo mode và tính các chỉ số tổng hợp."""
    by_mode: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"retr": [], "gen": [], "total": [], "attr": [], "rank1": []}
    )

    for r in records:
        mode = r.get("mode", "—")
        m = r.get("metrics", {})
        by_mode[mode]["retr"].append(float(m.get("retrieval_latency_ms", 0.0)))
        by_mode[mode]["gen"].append(float(m.get("generation_latency_ms", 0.0)))
        by_mode[mode]["total"].append(float(m.get("total_latency_ms", 0.0)))
        by_mode[mode]["attr"].append(float(m.get("attribution_score", 0.0)))
        ev = r.get("evidence", [])
        if ev:
            by_mode[mode]["rank1"].append(float(ev[0].get("rerank_score", 0.0)))

    out: Dict[str, Any] = {"total_records": len(records), "by_mode": {}}
    for mode, d in by_mode.items():
        retr_s = sorted(d["retr"]); gen_s = sorted(d["gen"]); tot_s = sorted(d["total"])
        out["by_mode"][mode] = {
            "n"                : len(d["retr"]),
            "retr_mean"        : round(_mean(d["retr"]), 1),
            "retr_p50"         : round(_percentile(retr_s, 0.50), 1),
            "retr_p95"         : round(_percentile(retr_s, 0.95), 1),
            "gen_mean"         : round(_mean(d["gen"]), 1),
            "gen_p95"          : round(_percentile(gen_s, 0.95), 1),
            "total_mean"       : round(_mean(d["total"]), 1),
            "total_p95"        : round(_percentile(tot_s, 0.95), 1),
            "retr_sla_500ms_%" : round(_pass_rate(d["retr"], 500.0), 1),
            "e2e_sla_8s_%"     : round(_pass_rate(d["total"], 8000.0), 1),
            "attr_mean"        : round(_mean(d["attr"]), 4),
            "rank1_rerank_mean": round(_mean(d["rank1"]), 4),
        }
    return out


def low_attribution(records: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    lows = []
    for r in records:
        attr = float(r.get("metrics", {}).get("attribution_score", 0.0))
        if attr < threshold:
            lows.append({
                "query": r.get("query", "")[:70],
                "attr" : attr,
                "model": r.get("model_used", ""),
                "mode" : r.get("mode", ""),
            })
    return lows


# ─── In bảng ────────────────────────────────────────────────────────────────

def print_report(summary: Dict[str, Any], lows: List[Dict[str, Any]], threshold: float) -> None:
    print("=" * 78)
    print(f"  PHÂN TÍCH INTERACTION LOG — tổng {summary['total_records']} lượt hỏi-đáp")
    print("=" * 78)

    header = f"{'MODE':<24}{'N':>4}{'Retr̄(ms)':>10}{'Retr.p95':>10}{'Gen̄(ms)':>10}{'Attr̄':>8}{'SLA<500':>9}"
    print(header)
    print("-" * 78)
    for mode, s in summary["by_mode"].items():
        print(f"{mode:<24}{s['n']:>4}{s['retr_mean']:>10}{s['retr_p95']:>10}"
              f"{s['gen_mean']:>10}{s['attr_mean']:>8}{s['retr_sla_500ms_%']:>8}%")
    print("-" * 78)
    print("  (Retr̄ = retrieval trung bình · Gen̄ = generation trung bình · Attr̄ = attribution trung bình)")
    print("  Latency CHỈ để tham khảo, không gate pass/fail — CPU bge-m3 vốn chậm là bình thường.")

    print()
    print(f"  TRUY VẤN ATTRIBUTION THẤP (< {threshold}) — cần soi căn cứ / prompt:  [{len(lows)} lượt]")
    print("-" * 78)
    if not lows:
        print("  (không có — mọi câu trả lời đều bám căn cứ tốt)")
    else:
        for x in lows[:20]:
            print(f"  attr={x['attr']:<6} [{x['mode']:<22}] {x['query']}")
        if len(lows) > 20:
            print(f"  ... và {len(lows) - 20} lượt nữa.")
    print("=" * 78)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Phân tích interaction log của VietLawBERT demo.")
    ap.add_argument("--file", type=Path, default=DEFAULT_LOG, help="Đường dẫn interactions.jsonl")
    ap.add_argument("--attr-threshold", type=float, default=0.5, help="Ngưỡng attribution coi là thấp")
    ap.add_argument("--json", action="store_true", help="Xuất JSON tổng hợp thay vì bảng")
    args = ap.parse_args()

    records = load_records(args.file)
    if not records:
        return 1

    summary = summarize(records)
    lows = low_attribution(records, args.attr_threshold)

    if args.json:
        print(json.dumps({"summary": summary, "low_attribution": lows},
                         ensure_ascii=False, indent=2))
    else:
        print_report(summary, lows, args.attr_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
