# VietLawBERT — Testing Pack (v1, 21/10/2026)

Bộ công cụ testing dựng theo **các đề xuất kiểm thử khác**: 
(1) metric đơn giản bằng cách chỉ sử dụng `equal`/`contain-any` tập trung dẫn chiếu số hiệu + điều/khoản
(2) **tách latency** thành số đo báo cáo (gỡ nghịch lý "latency cao PASS / thấp FAIL")
(3) **red-team** injection cho tầng searching. Kèm bước **làm sạch testcase/dẫn chiếu** và **phân loại nguyên nhân FAIL**.

> Mọi FAIL đều phải GIẢI THÍCH ĐƯỢC: mỗi assertion trả về `reason`, và `fail_category` gắn nhãn
> F1 (GT nghi ngờ) / F2 (retrieval miss) / F3 (lệch cấp khoản) / F4 (hạ tầng).

## Cây thư mục
Bộ công cụ nay được chia theo bố cục `tests/{_common, system, model}` (xem `tests/README.md`):
```
tests/_common/                  # dùng chung cho cả hai nhánh
  datasets/
    clean_vietlawbench.py       # làm sạch + gán Global Hierarchical ID + xuất 100 mẫu
    vietlawbench_clean.jsonl    # 729 câu sạch, có doc_norm/article_id/clause_id/global_id
    vietlawbench_dropped.jsonl  # 271 câu loại, kèm _drop_reasons
    vietlawbench_sample100.csv  # 100 mẫu để KIỂM TAY (cột manual_ok / manual_note)
    clean_report.json           # thống kê làm sạch
    load_tests.js               # nạp clean.jsonl -> test promptfoo (VLB_LIMIT giới hạn số câu)
  assertions/
    retrieval_metrics.js        # doc_hit@k (contain-any) · article_hit@k, clause_hit@k (equal) · latency_report (non-gating)
    fail_category.js            # gắn nhãn F1/F2/F3/F4/OK
  providers/                    # provider.py · retrieval_provider.py · generation_provider.py · gradio_chat_provider.py · mock_provider.py
  tools/
    analyze_latency.py          # P50/P95/P99 theo provider, ĐỘC LẬP với pass/fail
tests/system/
  configs/
    promptfooconfig.rq3_retrieval.yaml   # RQ3 — metric tất định, BỎ LLM judge, BỎ latency-gate
    promptfooconfig.redteam.yaml         # red-team ĐỘNG (plugins + strategies)
    promptfooconfig.redteam_cases.yaml   # red-team TẤT ĐỊNH (8 ca viết tay)
  redteam/
    injection_cases.jsonl       # 8 ca injection/OOD/bịa trích dẫn
    legal_injection_assert.js   # assert: abstain hoặc không bịa số hiệu
    load_injection.js
tests/model/
  configs/promptfooconfig.baselines.yaml # baseline đa backbone
```

## Kết quả làm sạch (chạy thật trên `vietlawbench_1000.jsonl`)
- Giữ **729/1000** câu sạch; loại **271 (27,1%)**.
- Lý do loại: `preamble_chunk` 257 · `article_not_Dieu` ("Điều khoản liên quan") 257 · `hierarchy_CAN_CU` 257 · `evidence_can_cu` 111 · `evidence_thi_hanh_or_sign` 14.
- Bộ sạch: cấp **clause** 560, cấp **article** 169; định danh doc vẫn lẫn uuid 582 / numeric 147 → đã chuẩn hoá qua `doc_norm` + `global_id`.

## Chạy
```bash
cd tests/system/configs
# vòng CPU, giới hạn 150 câu cho nhanh:
VLB_LIMIT=150 npx promptfoo@latest eval -c promptfooconfig.rq3_retrieval.yaml -o ../out/eval-rq3.json
npx promptfoo@latest view

# Latency (độc lập pass/fail):
python3 ../../_common/tools/analyze_latency.py --eval ../out/eval-rq3.json --cpu-threshold 5000 --gpu-threshold 500

# Red-team tất định:
npx promptfoo@latest eval -c promptfooconfig.redteam_cases.yaml -o ../out/redteam-cases.json
# Red-team động:
npx promptfoo@latest redteam generate -c promptfooconfig.redteam.yaml
npx promptfoo@latest redteam run      -c promptfooconfig.redteam.yaml -o ../out/redteam.json
```

## Khác biệt so với harness cũ (eval-G18/NRO/Xft)
| Cũ | Mới |
|---|---|
| `chunk_hit@5` exact trên GT sai + ID loạn | `doc_hit@k` contain-any + `article_hit@k`/`clause_hit@k` equal theo global-ID |
| `doc_hit@5` includes **2 chiều** (dương tính giả) | contain-any **một chiều** đã chuẩn hoá |
| `latency_sla_500ms` = `<60000ms` (luôn PASS, che tính đúng) | `latency_report` non-gating + `analyze_latency.py` (P50/P95/P99) |
| `legal_relevance_judge` LLM (không tất định) | bỏ khỏi retrieval; đánh giá câu trả lời để riêng cho RAGAS |
| FAIL không rõ nguyên nhân | `fail_category` F1/F2/F3/F4/OK + `reason` từng dòng |

