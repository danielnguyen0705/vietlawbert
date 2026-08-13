# Quy tắc crawl và chấp nhận dữ liệu VBPL

Một request HTTP 200 chưa đủ để xem là crawl thành công. Một record chỉ được đưa vào corpus khi vừa đạt transport success, vừa có nội dung pháp luật tiếng Việt dùng được, có provenance, và không vi phạm các tín hiệu loại bỏ.

## Trạng thái kỹ thuật cần phân biệt

Crawler và content adapter phải phân biệt rõ: `HTML_VALID`, `CONTENT_TOO_SHORT`, `PDF_ONLY`, `METADATA_ONLY`, `SECURITY_CHALLENGE`, `HTTP_ERROR`, `CONTRACT_ERROR`, và `PARSE_ERROR`. `PDF_ONLY` hoặc PDF attachment chỉ chứng minh có nguồn fallback, chưa chứng minh trích xuất text/OCR đã thành công.

## Tín hiệu bắt buộc loại hoặc đưa vào review

Các tín hiệu cần phát hiện theo nhiều pattern, không dựa vào một chuỗi đơn: Cloudflare, Performing security verification, Ray ID, Access denied, captcha, login/sign-in page, trang rỗng, nội dung quá ngắn, thiếu legal body, trang tiếng Anh bất thường khi kỳ vọng văn bản tiếng Việt, JSON/JSONL lỗi, metadata bắt buộc bị thiếu, source URL trùng, document ID trùng, content hash trùng với văn bản hợp lệ khác.

## Quy tắc chất lượng

Không tối ưu số lượng văn bản bằng cách nhận record kém chất lượng. Mỗi record bị loại phải có lý do cụ thể. Không biến lỗi crawler thành văn bản corpus. Không suy đoán bộ/ngành/lĩnh vực pháp lý từ URL hoặc tên tự do. Giữ riêng nhóm chính thức và hình thức văn bản từ VBPL metadata. Heuristic phân loại chỉ là tín hiệu review, không phải kết luận pháp lý.

## Pilot an toàn

Trước batch crawl mới, chạy contract verifier. Nếu gặp security challenge, contract change, hoặc parser incompatibility thì dừng batch. Pilot phải có rate limit, retry/backoff, checkpoint, resume, và báo cáo chất lượng. Kết quả pilot 120 hiện tại chỉ cho phép đi tiếp tới pilot 500–1.000 sau khi có PDF extraction/OCR fallback đã test; chưa phải phê duyệt crawl toàn sitemap.
