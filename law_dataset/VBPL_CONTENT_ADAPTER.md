# VBPL content adapter: cách dùng và giới hạn

## Mục tiêu

Adapter này xác minh cách lấy nội dung động từ `vbpl.vn` trước khi nối vào crawler chính. Nó hỗ trợ ID số, UUID và ID kiểu cũ, nhưng **không khẳng định mọi URL trong sitemap đều có nội dung dùng được**.

## Luồng xử lý

1. Đọc mã server action từ `config/vbpl_action_contract.json`.
2. Gọi action chi tiết để lấy metadata và HTML.
3. Gọi action file để tìm bản đính kèm/PDF.
4. Tìm chính xác theo số văn bản để lấy nhóm và hình thức do VBPL công bố.
5. Chỉ gọi action sơ đồ quan hệ khi thật sự cần.
6. Trả về một trạng thái rõ ràng để crawler quyết định bước tiếp theo.

| Trạng thái | Ý nghĩa | Xử lý tiếp |
|---|---|---|
| `HTML_VALID` | HTML đủ dài | Làm sạch, tách điều/khoản và lưu |
| `CONTENT_TOO_SHORT` | Có HTML nhưng quá ngắn | Ưu tiên file; nếu thất bại thì review |
| `PDF_ONLY` | Không có HTML nhưng có PDF | Đưa vào hàng đợi trích xuất PDF/OCR |
| `METADATA_ONLY` | Chỉ có metadata | Không đưa vào corpus nội dung; lưu để rà soát |
| `SECURITY_CHALLENGE` | VBPL yêu cầu kiểm tra bảo mật | Dừng/giảm tốc; không cố vượt qua |
| `CONTRACT_ERROR` | Action hoặc cấu trúc phản hồi đã đổi | Dừng batch và xác minh lại contract |
| `PARSE_ERROR` | Không đọc được phản hồi | Lưu mẫu lỗi, kiểm tra parser |

## Phân loại

Không suy ra “ngành” từ hostname hoặc tên bộ/ngành. Adapter giữ riêng hai trục:

- `official_group_*`: nhóm chính thức, ví dụ `VBQPPL`, `VBHN`, `VBHTH`, `BD`.
- `official_form_*`: hình thức văn bản, ví dụ Luật, Nghị định, Quyết định.

Nhóm chính thức là bằng chứng tốt nhất từ VBPL, nhưng không đồng nghĩa dữ liệu nguồn luôn đúng. Quyết định thuộc `VBQPPL` nhưng số văn bản không có mẫu năm như `/2025/QĐ-...` được gắn `DECISION_NUMBER_PATTERN_REVIEW_RECOMMENDED`. Đây chỉ là heuristic đưa vào hàng đợi review, không phải kết luận pháp lý. Sau này “ngành/lĩnh vực” nên lấy từ trường chủ đề/lĩnh vực của từng văn bản và chuẩn hóa bằng một bảng mã riêng; không tạo “bộ/ngành ảo” từ URL hay chuỗi tự do.

## Chạy phép kiểm tra đại diện

Từ thư mục gốc repository:

```bash
python law_dataset/src/crawler/verify_vbpl_contract.py \
  --diagram-id 3dedecb0-9239-11f1-889f-513c5fa29d01 \
  --diagram-id vbpqta_1964
```

Lệnh mặc định kiểm tra sáu mẫu đại diện và chỉ lưu hash/độ dài nội dung trong `artifacts/vbpl_contract_verification`; nó không lưu toàn bộ văn bản.

## Trước mỗi đợt crawl

1. Chạy verifier và yêu cầu kết quả `PASS`.
2. Nếu có `CONTRACT_ERROR`, đọc lại chunk JavaScript hiện hành và cập nhật bốn action hash; không tiếp tục bằng hash cũ.
3. Chạy pilot nhỏ, phân tầng theo phạm vi trung ương/địa phương, loại ID và nhóm văn bản.
4. Kiểm tra tỷ lệ trạng thái, trùng lặp, thiếu nội dung và cảnh báo phân loại.
5. Chỉ tăng quy mô khi pilot đạt ngưỡng chất lượng đã định.

## Chạy pilot 120 văn bản

Sau khi verifier đạt `PASS`, chạy:

```bash
python law_dataset/src/crawler/pilot_crawl_vbpl.py
```

Pilot lấy mẫu xác định theo ba trục `scope × id_type × observed_group`, chạy một luồng với delay mặc định 0,75 giây giữa các request, có checkpoint để tiếp tục khi gián đoạn và dừng sớm nếu gặp lỗi contract, parser hoặc security challenge. File `documents.jsonl.gz` và checkpoint là dữ liệu crawl cục bộ nên không được commit; manifest và báo cáo chất lượng có thể commit để tái lập phép thử.

Tại thời điểm xác minh 2026-08-10, action không cần cookie hoặc router state. Đây là tính chất của build đã kiểm tra, không phải cam kết API ổn định của VBPL.

## Rủi ro còn lại

- Action hash có thể đổi khi frontend được deploy lại: mức cao, nhưng verifier phát hiện sớm.
- HTML không đầy đủ, phải dùng PDF/OCR: mức trung bình-cao.
- Phân loại nguồn có thể gây tranh luận hoặc sai dữ liệu: mức trung bình; giữ cảnh báo và hàng đợi review.
- Chặn bảo mật/rate limit khi tăng tải: mức cao; cần giới hạn tốc độ, backoff và checkpoint.
- Toàn bộ sitemap chưa được kiểm tra nội dung: mức cao; inventory URL không được xem là corpus hợp lệ.
