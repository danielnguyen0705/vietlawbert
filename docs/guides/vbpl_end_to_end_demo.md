# Demo VBPL → AST → Qdrant/Elasticsearch → NVIDIA → Chainlit

Demo dùng dữ liệu thật, pipeline xử lý hiện có và API LLM thật. Phạm vi là
crawl, lọc nội dung, tách điều/khoản, embedding, tìm kiếm lai và hỏi đáp.
Chưa chạy huấn luyện MRL, xây đồ thị HIN, reranker hoặc đánh giá RQ1–RQ4.
Encoder demo là `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(384 chiều, CPU). Đây là baseline kiểm tra luồng, không phải model VietLawBERT đã fine-tune.

## Cấu hình

Trong Ubuntu:

```bash
cd /mnt/d/vietlawbert
```

Trong `.env` tại repo, tự điền key (file này được Git bỏ qua):

```dotenv
PRIMARY_LLM_API_BASE=https://integrate.api.nvidia.com/v1
PRIMARY_LLM_MODEL=openai/gpt-oss-20b
PRIMARY_LLM_API_KEY=YOUR_NVIDIA_KEY
LLM_API_KEY=ollama
```

Không đưa key vào lệnh terminal, báo cáo hoặc commit. Container đọc `.env` qua
repo được mount; Docker build demo không chép `.env` vào image.

Máy hiện tại có bản encoder đã sao chép từ cache vào `data/demo/embedding-model`;
`.env` có `DEMO_EMBED_MODEL=/app/data/demo/embedding-model`. Trên máy khác, bỏ dòng
này để tải model theo tên mặc định. Không dùng đường dẫn `/app/...` làm model
cho Python chạy trực tiếp trên Windows.

## 1. Build và kiểm tra LLM

Image demo kế thừa image pipeline `vietlawbert-app:latest`. Nếu chưa có image này:

```bash
docker compose -f docker-compose.yml build app
```

Sau đó:

```bash
docker compose -f docker-compose.demo.yml build worker ui
docker compose -f docker-compose.demo.yml run --rm --no-deps worker python -m cli.check_demo_llm
```

Chỉ đi tiếp phần sinh câu trả lời khi thấy `LLM_READY openai/gpt-oss-20b`.
Demo dùng tối đa 4096 output tokens và tắt fallback local; mỗi lần chat/check gọi
API thực bằng tài khoản NVIDIA đã cấu hình.

## 2. Cào pilot nhỏ

```bash
docker compose -f docker-compose.demo.yml run --rm --no-deps worker python -m cli.demo crawl --documents 20
```

Đọc `data/demo/raw_shards/crawl_pages_00001_00001.audit.json` và log cùng tên.
Shard đã đạt audit được dùng lại khi chạy lại. Muốn một lượt crawl mới, lưu dữ
liệu demo cũ sang thư mục khác và dùng một thư mục/collection demo mới; không
trộn snapshot mới với index cũ. Không xóa dữ liệu sản xuất để thử lại demo.

Record OCR_PENDING không được coi là nội dung hợp lệ. Demo này giữ chúng trong
quarantine và loại khỏi ingestion; OCR phục hồi là bước mở rộng cần kiểm tra riêng.

## 3. Xử lý và nạp hai kho

```bash
docker compose -f docker-compose.demo.yml up -d --wait qdrant elasticsearch
docker compose -f docker-compose.demo.yml run --rm worker python -m cli.demo prepare
```

`prepare` đọc shard đã audit, lọc HTML/nội dung/Unicode, dùng HybridASTParser,
tạo embedding, nạp Qdrant và Elasticsearch. Lượt nạp lại ghi theo chunk ID xác
định vào cả hai kho để phục hồi lỗi ghi dở. Báo cáo chỉ thành công nếu ID và nội
dung của mọi chunk trong hai kho khớp nhau, không rỗng và thuộc corpus demo.

Các file để kiểm tra:

- `data/demo/reports/content_gate.json`: số văn bản hợp lệ và các bản loại.
- `data/demo/reports/documents.json`: số hiệu, tiêu đề, metadata nguồn và API VBPL.
- `data/demo/reports/ingestion.json`: số văn bản/đoạn đã nạp, model, chiều vector.
- `data/demo/reports/chunks.json`: toàn bộ nội dung và phân cấp đã lưu.
- `data/demo/selected.jsonl`: các record thật được chọn cho ingestion.

Các volume `vietlawbert-demo_*` và collection/index `vietlawbert_demo_chunks`
tách riêng khỏi hạ tầng mặc định. Cơ sở dữ liệu demo không công bố port ra host.

## 4. Kiểm tra tìm kiếm

```bash
docker compose -f docker-compose.demo.yml run --rm worker python -m cli.demo evaluate
```

`cases.json` chứa tối đa 5 câu hỏi tự sinh theo số hiệu và điều/khoản thực sự đã
nạp. `retrieval.json` ghi kết quả top 5, Recall@5, MRR@5 và độ trễ từng câu.
Đây là phép kiểm tra tìm lại đoạn đã biết, không phải benchmark độc lập; điểm
cao không chứng minh chất lượng trả lời pháp luật hoặc khả năng tổng quát.

## 5. Mở RAG thật và gọi end-to-end

```bash
docker compose -f docker-compose.demo.yml --profile chat up -d --wait api ui
docker compose -f docker-compose.demo.yml run --rm worker python -m cli.demo chat-check --case-index 3
```

- Chat: <http://localhost:8011>
- Báo cáo, câu hỏi gợi ý và bằng chứng: <http://localhost:8010/demo>
- API: <http://localhost:8010/docs>

`chat-check` gửi một câu hỏi qua FastAPI, buộc phản hồi có mode RAG, đúng chunk
kỳ vọng, câu trả lời không phải fallback thiếu căn cứ và kết quả LLM thật; lưu câu trả
lời vào `reports/chat.json`. Health chỉ xác nhận provider
khởi tạo được, không thay thế kiểm tra dữ liệu/LLM này.

UI mock cũ vẫn ở 8001; hãy dùng **8011** cho demo dữ liệu thật.

## 6. Đánh giá thủ công

Lấy câu hỏi trong báo cáo, gửi vào UI, mở từng căn cứ ở cạnh câu trả lời và ghi:

| Tiêu chí | Cách kiểm tra |
| --- | --- |
| Truy xuất | Có tìm đúng số hiệu, điều và khoản kỳ vọng trong top 5? |
| Nội dung | Đoạn tách có giữ đúng văn bản nguồn, thiếu/thừa đoạn nào? |
| Căn cứ | Mỗi khẳng định trong câu trả lời có được đoạn trích hỗ trợ? |
| Phạm vi | Có nhầm tỉnh/thành, văn bản sửa đổi hoặc thời điểm hiệu lực? |
| Độ trễ | Ghi thời gian hiển thị ở cuối câu trả lời. |

Thử thêm câu ngoài corpus và câu diễn đạt lại. Đánh giá thủ công xem hệ thống có
thừa nhận thiếu căn cứ hay vẫn suy diễn. Demo chỉ tìm các đoạn có trạng thái nguồn
“Còn hiệu lực” và ngày bắt đầu hiệu lực không ở tương lai tại thời điểm ingestion.
Các trạng thái thiếu, chưa hiệu lực, hết hiệu lực một phần hoặc tạm ngưng được loại
khỏi tìm kiếm để chờ review. Đây vẫn chưa phải kiểm chứng hiệu lực pháp lý đầy đủ
theo ngày hỏi: cần nạp lại khi trạng thái/ngày thay đổi và kiểm tra quan hệ sửa đổi.
Attribution là phép đối sánh tên văn bản/điều, không phải xác suất đúng.

## Sau khi sửa code hoặc Git cập nhật

Toàn bộ repo được bind mount nên file đổi ngay; restart API/UI để nạp lại module:

```bash
docker compose -f docker-compose.demo.yml restart api ui
```

Nếu đổi logic tách/embedding, cần nạp lại dữ liệu; không chỉ restart ứng dụng.
Nếu đổi encoder hoặc chiều vector, dùng collection/index mới tương ứng. Nếu đổi
thư viện, build lại worker rồi tạo lại API. Dừng demo, giữ dữ liệu:

```bash
docker compose -f docker-compose.demo.yml --profile chat down
```

## Giới hạn hiện tại

Chưa chứng minh OCR toàn bộ, đồ thị HIN, huấn luyện MRL hoặc chất lượng pháp lý.
Retriever hiện không tự nạp graph cache/cross-encoder nên điểm graph bằng 0 và
điểm rerank là điểm hợp nhất tìm kiếm. Model MiniLM giới hạn độ dài đầu vào;
chunk dài bị cắt khi embedding nhưng nội dung gốc vẫn lưu để đối chiếu.

Tài liệu API NVIDIA: <https://docs.api.nvidia.com/nim/reference/openai-gpt-oss-20b-infer>.

## Những lỗi được phát hiện khi demo

- Lỗi ghi storage từng bị nuốt và có thể dẫn đến checkpoint sai: nay phát sinh lỗi rõ ràng.
- Điều trùng số trong văn bản sửa đổi từng dùng chung chunk ID: thêm thứ tự xuất hiện
  xác định để bảo toàn mọi đoạn; số đoạn parse phải bằng số ID thực sự lưu.
- Trạng thái chưa có hiệu lực từng bị coi là còn hiệu lực; nhánh sparse không lọc
  hiệu lực: đã thêm điều kiện trạng thái/ngày và áp dụng lọc ở cả dense/sparse.
- Các kết quả trước sửa hiệu lực được giữ trong `reports/*_before_validity_fix.json`.
  Chúng dùng bộ câu hỏi khác với lượt cuối, không dùng để tính mức cải thiện mô hình.
