"""Tính Cohen's Kappa giữa 2 người gán nhãn (task 1.6 — audit Cochran). Pure Python, không cần sklearn.
Chạy:  python -m tests._common.datasets.compute_kappa --csv collaborator_inputs/cochran_sample_382.csv --col-a nhan_A --col-b nhan_B
Yêu cầu: file CSV có 2 cột nhãn (vd nhan_A, nhan_B), mỗi ô là SACH hoặc BAN."""
from __future__ import annotations
import argparse, csv as _csv
from collections import Counter


def cohen_kappa(labels_a, labels_b) -> float:
    """κ = (Po − Pe)/(1 − Pe). Po: đồng thuận quan sát; Pe: đồng thuận kỳ vọng ngẫu nhiên."""
    assert labels_a and len(labels_a) == len(labels_b), "hai cột phải cùng độ dài và khác rỗng"
    n = len(labels_a)
    cats = set(labels_a) | set(labels_b)
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)


def load_two_cols(path, col_a, col_b):
    a, b = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            va, vb = (row.get(col_a) or "").strip(), (row.get(col_b) or "").strip()
            if va and vb:            # chỉ tính dòng CẢ HAI đã gán
                a.append(va.upper()); b.append(vb.upper())
    return a, b


def main():
    ap = argparse.ArgumentParser(description="Cohen's kappa cho audit Cochran (1.6)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col-a", default="nhan_A")
    ap.add_argument("--col-b", default="nhan_B")
    args = ap.parse_args()
    la, lb = load_two_cols(args.csv, args.col_a, args.col_b)
    if not la:
        raise SystemExit("Không có dòng nào cả hai người đã gán — kiểm tra lại tên cột.")
    k = cohen_kappa(la, lb)
    verdict = "ĐẠT (κ ≥ 0.8)" if k >= 0.8 else "CHƯA ĐẠT — họp hiệu chỉnh rubric rồi gán lại 1 vòng"
    print(f"n (đã gán cả hai) = {len(la)} | Cohen's kappa = {k:.4f} | {verdict}")


if __name__ == "__main__":
    main()
