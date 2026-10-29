#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_latency.py — Tách LATENCY khỏi pass/fail và báo cáo P50/P95/P99 theo provider.

Vì sao cần: trong harness cũ, assert `latency_sla_500ms` thực chất là `< 60000ms` nên LUÔN PASS,
khiến latency không liên quan tới pass/fail — gây nghịch lý "latency cao vẫn PASS, thấp vẫn FAIL".
Tool này đọc file eval promptfoo (định dạng {head, body}) và báo cáo latency ĐỘC LẬP với tính đúng.

Chạy:
  python3 analyze_latency.py --eval ../out/eval-rq3.json [--cpu-threshold 5000] [--gpu-threshold 500]
  python3 analyze_latency.py --eval eval-G18-....json      # dùng được với file cũ
"""
import argparse, json, statistics as st

def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)

def get_latency(out):
    t = out.get("text")
    if isinstance(t, str):
        try: return json.loads(t).get("latency_ms")
        except Exception: pass
    if isinstance(out.get("response"), dict):
        pass
    return out.get("latencyMs")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--cpu-threshold", type=float, default=5000.0)
    ap.add_argument("--gpu-threshold", type=float, default=500.0)
    args = ap.parse_args()

    o = json.load(open(args.eval, encoding="utf-8"))

    def locate(obj):
        # 1) định dạng share/table ở top-level (file G18 cũ)
        if isinstance(obj.get("head"), dict) and "body" in obj:
            return obj["head"], obj["body"]
        # 2) output của `promptfoo eval -o`: head/body nằm trong results.table
        res = obj.get("results", {})
        if isinstance(res, dict) and isinstance(res.get("table"), dict) \
           and "head" in res["table"] and "body" in res["table"]:
            return res["table"]["head"], res["table"]["body"]
        # 3) table ở top-level
        if isinstance(obj.get("table"), dict) and "head" in obj["table"]:
            return obj["table"]["head"], obj["table"]["body"]
        return None, None

    head, body = locate(o)
    if head is None:
        raise SystemExit("Không nhận ra định dạng eval. Top-level keys: " + ", ".join(o.keys()))
    provs = [p.get("provider") if isinstance(p, dict) else str(p) for p in head["prompts"]]

    print(f"=== LATENCY REPORT: {args.eval} ===")
    print(f"{'provider':<42} {'n':>3} {'P50':>8} {'P95':>8} {'P99':>8} {'max':>8}  {'<CPU':>5} {'<GPU':>5}")
    summary = {}
    for pi, prov in enumerate(provs):
        lats, pass_lat, fail_lat = [], [], []
        for row in body:
            if pi >= len(row["outputs"]): continue
            out = row["outputs"][pi]
            ms = get_latency(out)
            if not isinstance(ms, (int, float)): continue
            lats.append(ms)
            (pass_lat if out.get("pass") else fail_lat).append(ms)
        if not lats:
            print(f"{str(prov):<42} {'0':>3}  (không có latency_ms)"); continue
        under_cpu = round(100*sum(1 for x in lats if x < args.cpu_threshold)/len(lats))
        under_gpu = round(100*sum(1 for x in lats if x < args.gpu_threshold)/len(lats))
        print(f"{str(prov):<42} {len(lats):>3} {pct(lats,50):>8} {pct(lats,95):>8} "
              f"{pct(lats,99):>8} {round(max(lats)):>8}  {under_cpu:>4}% {under_gpu:>4}%")
        summary[str(prov)] = {
            "n": len(lats), "p50": pct(lats,50), "p95": pct(lats,95), "p99": pct(lats,99),
            "max": round(max(lats)), "pct_under_cpu_threshold": under_cpu,
            "pct_under_gpu_threshold": under_gpu,
            "latency_pass_median": round(st.median(pass_lat)) if pass_lat else None,
            "latency_fail_median": round(st.median(fail_lat)) if fail_lat else None,
        }

    # Kiểm chứng NGHỊCH LÝ: latency có tương quan với pass/fail không?
    all_pass, all_fail = [], []
    for pi in range(len(provs)):
        for row in body:
            if pi >= len(row["outputs"]): continue
            out = row["outputs"][pi]; ms = get_latency(out)
            if isinstance(ms, (int, float)):
                (all_pass if out.get("pass") else all_fail).append(ms)
    print("\n--- Latency vs Pass (toàn bộ) ---")
    print(f"  PASS: n={len(all_pass)} median={round(st.median(all_pass)) if all_pass else '-'} "
          f"max={round(max(all_pass)) if all_pass else '-'}")
    print(f"  FAIL: n={len(all_fail)} median={round(st.median(all_fail)) if all_fail else '-'} "
          f"min={round(min(all_fail)) if all_fail else '-'}")
    print("  => Nếu median PASS ≈ median FAIL, latency KHÔNG phân biệt đúng/sai: đúng như kỳ vọng")
    print("     khi đã TÁCH latency khỏi tiêu chí pass/fail (latency chỉ để báo cáo, không gate).")

    import os
    out_path = os.path.splitext(args.eval)[0] + ".latency.json"
    json.dump(summary, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[output] {out_path}")

if __name__ == "__main__":
    main()
