# VietLawBERT - Legal Dataset Pipeline

Repo: `https://github.com/danielnguyen0705/vietlawbert`

## Mục tiêu

Repo này xây dựng pipeline thu thập, parse, chunk, load MongoDB và tạo dataset pháp luật theo domain.

Hiện tại hệ thống hỗ trợ:

- chạy MongoDB bằng Docker
- chạy FastAPI bằng Docker
- crawl link văn bản pháp luật
- parse và chuẩn hóa nội dung văn bản
- chunk văn bản để phục vụ search/RAG
- tạo index và load dữ liệu vào MongoDB
- xây dựng dataset chuyên biệt cho luật giao thông đường bộ
- cung cấp API để xem document, chunk và search theo từ khóa

## Quy trình nhanh

Nếu muốn chạy lại toàn bộ từ đầu và tạo dataset luật giao thông:

```powershell
docker compose up -d --build
docker compose exec app python reset_project_state.py --yes
docker compose exec app python run_full_pipeline.py
```

Khi chạy `run_full_pipeline.py`, script sẽ hỏi có muốn reset database và dữ liệu cũ trước khi chạy hay không. Nếu bạn đã chạy `reset_project_state.py --yes` ở bước trước thì trả lời `n`.

Sau đó mở:

- `http://localhost:8000/health`
- `http://localhost:8000/db-check`
- `http://localhost:8000/docs`
- `http://localhost:8000/documents`
- `http://localhost:8000/search?keyword=giao%20thông`

## 1. Chuẩn bị môi trường

### Kiểm tra Python

Python chủ yếu cần nếu bạn muốn chạy script trên máy local. Nếu chỉ dùng Docker thì bước này là tùy chọn.

```powershell
python --version
```

### Kiểm tra Docker

```powershell
docker --version
docker compose version
```

## 2. Clone repo và tạo branch

```powershell
git clone https://github.com/danielnguyen0705/vietlawbert.git
cd vietlawbert
git checkout -b cuong
```

## 3. Kiểm tra file cấu hình `.env`

Repo đã có sẵn file `.env`. Giá trị mặc định hiện tại:

```env
MONGO_HOST=mongodb
MONGO_PORT=27017
MONGO_INITDB_ROOT_USERNAME=admin
MONGO_INITDB_ROOT_PASSWORD=admin123
MONGO_DB_NAME=vietlawbert
MONGO_AUTH_SOURCE=admin
APP_PORT=8000
```

Thông thường không cần sửa gì để chạy trên máy local bằng Docker.

## 4. Chạy Docker

### Build và start container

```powershell
docker compose up -d --build
```

### Kiểm tra container

```powershell
docker ps
```

### Xem log nếu cần

```powershell
docker compose logs -f
```

## 5. Kiểm tra API cơ bản

### Health check

```text
http://localhost:8000/health
```

### Kiểm tra kết nối DB

```text
http://localhost:8000/db-check
```

### Swagger UI

```text
http://localhost:8000/docs
```

## 6. Tạo index cho MongoDB

```powershell
docker compose exec app python -m app.db.init_indexes
```

## 7. Load dữ liệu vào MongoDB

Lệnh này đọc dữ liệu từ `law_dataset/data` và ghi vào MongoDB.

```powershell
docker compose exec app python phase4_load_to_mongodb.py
```

Thông thường bạn không cần chạy riêng bước này nếu đã chạy:

```powershell
docker compose exec app python run_full_pipeline.py
```

Vì full pipeline đã tự chạy bước tạo index và load MongoDB.

## 8. Kiểm tra dữ liệu đã load

### Xem danh sách document

```text
http://localhost:8000/documents
```

### Xem chi tiết 1 document

```text
http://localhost:8000/documents/{document_id}
```

### Xem chunks của 1 document

```text
http://localhost:8000/documents/{document_id}/chunks
```

### Search theo từ khóa

```text
http://localhost:8000/search?keyword=giao%20thông
```

## 9. Reset toàn bộ database và dữ liệu crawl

### Có xác nhận

```powershell
docker compose exec app python reset_project_state.py
```

### Bỏ qua xác nhận

```powershell
docker compose exec app python reset_project_state.py --yes
```

Lệnh này sẽ:

- xóa database MongoDB hiện tại
- xóa các file dữ liệu trong `law_dataset/data`
- tạo lại các thư mục cần thiết

## 10. Chạy full pipeline từ đầu

```powershell
docker compose exec app python run_full_pipeline.py
```

Lệnh này chạy lần lượt:

- `phase1_collect_links.py`
- `phase2_parse_documents.py`
- `phase3_chunk_documents.py`
- `python -m app.db.init_indexes`
- `phase4_load_to_mongodb.py`
- `phase5_build_domain_dataset.py`

Khi bắt đầu, script sẽ hỏi:

```text
Reset database và dữ liệu cũ trước khi chạy? (y/n):
```

Nếu muốn reset luôn trong script, nhập `y`. Nếu đã reset trước đó hoặc muốn chạy tiếp trên dữ liệu hiện có, nhập `n`.

### Reset trước rồi chạy lại từ đầu

```powershell
docker compose exec app python reset_project_state.py --yes
docker compose exec app python run_full_pipeline.py
```

Khi `run_full_pipeline.py` hỏi reset, nhập `n` vì dữ liệu đã được reset ở lệnh trước.

## 11. Xây dựng dataset luật giao thông đường bộ

Sau khi hoàn thành các phase crawl, parse, chunk và load MongoDB, hệ thống có thể xây dựng thêm một bộ dataset chuyên biệt cho lĩnh vực luật giao thông đường bộ.

Mục tiêu của phase này là lọc các văn bản liên quan đến:

- giao thông đường bộ
- trật tự, an toàn giao thông đường bộ
- xử phạt vi phạm giao thông
- giấy phép lái xe
- xe cơ giới
- đăng kiểm
- tốc độ, nồng độ cồn, biển báo, vạch kẻ đường

Dataset này là bước đầu để sau này mở rộng sang nhiều lĩnh vực pháp luật khác như lao động, đất đai, thuế, giáo dục, y tế.

### 11.1. Tạo dataset luật giao thông sau khi chạy full pipeline

Hiện tại `run_full_pipeline.py` đã chạy sẵn Phase 5 với cấu hình:

```text
configs/domains/traffic_law.json
```

Vì vậy cách chạy đầy đủ là:

```powershell
docker compose exec app python run_full_pipeline.py
```

Nếu chỉ muốn chạy riêng phase build dataset theo domain:

```powershell
docker compose exec app python phase5_build_domain_dataset.py --domain-config configs/domains/traffic_law.json
```

Nếu muốn lấy thêm các văn bản có khả năng liên quan nhưng cần kiểm tra thủ công, dùng thêm option `--include-review`:

```powershell
docker compose exec app python phase5_build_domain_dataset.py --domain-config configs/domains/traffic_law.json --include-review
```

### 11.2. Kết quả đầu ra

Sau khi chạy thành công, dữ liệu sẽ được xuất ra thư mục:

```text
law_dataset/data/domain_datasets/traffic_law/
```

Các file chính gồm:

| File | Mô tả |
|---|---|
| `documents.jsonl` | Danh sách văn bản pháp luật thuộc domain luật giao thông đường bộ |
| `chunks.jsonl` | Các chunk thuộc domain luật giao thông, dùng cho RAG/search |
| `need_review_documents.jsonl` | Các văn bản có thể liên quan nhưng cần kiểm tra lại |
| `summary.json` | Thống kê số lượng văn bản, chunk và trạng thái lọc domain |

### 11.3. Ý nghĩa các trạng thái domain

Mỗi văn bản sau khi phân loại domain sẽ có các trường:

| Field | Ý nghĩa |
|---|---|
| `domain_id` | Mã domain, ví dụ `traffic_law` |
| `domain_name` | Tên domain, ví dụ `Luật giao thông đường bộ` |
| `domain_score` | Điểm liên quan của văn bản với domain |
| `domain_status` | Trạng thái phân loại domain |
| `matched_keywords` | Các từ khóa đã khớp |
| `matched_patterns` | Các mẫu title/pattern đã khớp |

`domain_status` có 3 giá trị chính:

| Giá trị | Ý nghĩa |
|---|---|
| `relevant` | Văn bản liên quan rõ ràng đến luật giao thông |
| `need_review` | Văn bản có khả năng liên quan, nên kiểm tra thủ công |
| `ignore` | Văn bản không thuộc domain luật giao thông |

### 11.4. Chạy pipeline đầy đủ có cả Traffic Law Dataset

Chỉ cần chạy:

```powershell
docker compose exec app python run_full_pipeline.py
```

Pipeline sẽ chạy lần lượt:

- Phase 1 - Collect Links
- Phase 2 - Parse Documents
- Phase 3 - Chunk Documents
- Init MongoDB Indexes
- Phase 4 - Load to MongoDB
- Phase 5 - Build Traffic Law Dataset

### 11.5. Kiểm tra kết quả

Xem file thống kê:

```powershell
docker compose exec app cat law_dataset/data/domain_datasets/traffic_law/summary.json
```

Xem một vài văn bản đã được lọc:

```powershell
docker compose exec app sh -c "head -n 3 law_dataset/data/domain_datasets/traffic_law/documents.jsonl"
```

Xem một vài chunk dùng cho RAG:

```powershell
docker compose exec app sh -c "head -n 3 law_dataset/data/domain_datasets/traffic_law/chunks.jsonl"
```

Nếu container có PowerShell, có thể dùng:

```powershell
docker compose exec app powershell -Command "Get-Content law_dataset/data/domain_datasets/traffic_law/documents.jsonl -TotalCount 3"
```

### 11.6. Mở rộng sang domain pháp luật khác

Thiết kế hiện tại theo hướng:

- một pipeline chung
- nhiều file cấu hình domain

Ví dụ hiện tại có:

```text
configs/domains/traffic_law.json
```

Sau này có thể thêm:

```text
configs/domains/labor_law.json
configs/domains/land_law.json
configs/domains/tax_law.json
configs/domains/education_law.json
```

Sau đó chạy:

```powershell
docker compose exec app python phase5_build_domain_dataset.py --domain-config configs/domains/labor_law.json
```

Hoặc:

```powershell
docker compose exec app python phase5_build_domain_dataset.py --domain-config configs/domains/land_law.json
```

Như vậy, code crawler, parser, chunker và dataset builder vẫn dùng lại được. Chỉ cần thay file cấu hình domain để mở rộng sang lĩnh vực pháp luật mới.

## 12. Workflow chạy từ đầu

Nếu muốn chạy lại toàn bộ từ đầu và tạo dataset luật giao thông:

```powershell
docker compose up -d --build
docker compose exec app python reset_project_state.py --yes
docker compose exec app python run_full_pipeline.py
```

Khi `run_full_pipeline.py` hỏi reset, nhập `n`.

Nếu muốn lấy cả nhóm cần kiểm tra thủ công:

```powershell
docker compose exec app python phase5_build_domain_dataset.py --domain-config configs/domains/traffic_law.json --include-review
```

Sau đó kiểm tra:

```powershell
docker compose exec app cat law_dataset/data/domain_datasets/traffic_law/summary.json
docker compose exec app sh -c "head -n 3 law_dataset/data/domain_datasets/traffic_law/documents.jsonl"
docker compose exec app sh -c "head -n 3 law_dataset/data/domain_datasets/traffic_law/chunks.jsonl"
```

Nếu chỉ muốn tạo lại index và load lại dữ liệu vào MongoDB:

```powershell
docker compose exec app python -m app.db.init_indexes
docker compose exec app python phase4_load_to_mongodb.py
```

Mở API:

```text
http://localhost:8000/docs
http://localhost:8000/documents
http://localhost:8000/search?keyword=giao%20thông
```

## 13. Các lệnh thường dùng

### Build lại app

```powershell
docker compose up -d --build
```

### Stop container

```powershell
docker compose down
```

### Xem log

```powershell
docker compose logs -f
```

### Tạo index

```powershell
docker compose exec app python -m app.db.init_indexes
```

### Load dữ liệu

```powershell
docker compose exec app python phase4_load_to_mongodb.py
```

### Reset dữ liệu

```powershell
docker compose exec app python reset_project_state.py --yes
```

### Chạy full pipeline

```powershell
docker compose exec app python run_full_pipeline.py
```

### Build lại Traffic Law Dataset

```powershell
docker compose exec app python phase5_build_domain_dataset.py --domain-config configs/domains/traffic_law.json
```

### Build lại Traffic Law Dataset gồm cả nhóm cần review

```powershell
docker compose exec app python phase5_build_domain_dataset.py --domain-config configs/domains/traffic_law.json --include-review
```

## 14. Trạng thái hiện tại

Hiện tại hệ thống đã làm được:

- chạy FastAPI bằng Docker
- chạy MongoDB bằng Docker
- kết nối app với MongoDB
- crawl link văn bản pháp luật
- parse document
- chunk document
- tạo index MongoDB
- load document và chunk vào database
- tạo dataset luật giao thông đường bộ
- xem dữ liệu qua API
- search text trên collection `chunks`

Lưu ý:

- nếu chưa có `chunks.jsonl` thì collection `chunks` sẽ chưa có dữ liệu
- khi đó endpoint `/documents/{document_id}/chunks` và `/search` có thể không có kết quả
- nếu chưa có `law_dataset/data/domain_datasets/traffic_law/summary.json` thì cần chạy `run_full_pipeline.py` hoặc chạy riêng `phase5_build_domain_dataset.py`
- nếu muốn tạo dữ liệu đầy đủ từ đầu, hãy chạy `docker compose exec app python run_full_pipeline.py`

## 15. Cấu trúc dữ liệu MongoDB

Database hiện tại là `vietlawbert` với các collection chính: `source_links`, `documents`, `chunks`, `pipeline_logs`.

### Collection: `source_links`

| Field | Type | Mô tả |
|---|---|---|
| `_id` | `ObjectId` | Khóa chính do MongoDB tự tạo |
| `source_url` | `string` | URL nguồn của văn bản |
| `source_site` | `string` | Tên website hoặc domain nguồn |
| `crawl_status` | `string` | Trạng thái crawl của link, hiện tại thường là `discovered` |
| `updated_at` | `string` | Thời gian cập nhật bản ghi, đang lưu dưới dạng ISO datetime string |

### Collection: `documents`

| Field | Type | Mô tả |
|---|---|---|
| `_id` | `ObjectId` | Khóa chính do MongoDB tự tạo |
| `external_id` | `string \| null` | ID gốc từ nguồn dữ liệu ban đầu, có thể không có |
| `source_url` | `string` | URL nguồn của văn bản |
| `title` | `string \| null` | Tiêu đề văn bản pháp luật |
| `document_type` | `string \| null` | Loại văn bản, ví dụ luật, nghị định, thông tư |
| `document_number` | `string \| null` | Số hiệu văn bản |
| `issuer` | `string \| null` | Cơ quan ban hành chính |
| `issuer_codes` | `array[string]` | Danh sách mã cơ quan ban hành |
| `issuers` | `array[string]` | Danh sách đầy đủ các cơ quan ban hành |
| `raw_text` | `string \| null` | Nội dung thô của văn bản |
| `cleaned_text` | `string \| null` | Nội dung văn bản sau khi làm sạch |
| `parse_status` | `string` | Trạng thái parse dữ liệu, hiện tại thường là `success` |
| `created_at` | `string` | Thời gian tạo document, đang lưu dưới dạng ISO datetime string |
| `updated_at` | `string` | Thời gian cập nhật document, đang lưu dưới dạng ISO datetime string |
| `pipeline_version` | `string` | Phiên bản pipeline tạo ra dữ liệu |
| `document_key` | `string` | Khóa định danh logic được băm từ một số trường chính |

### Collection: `chunks`

| Field | Type | Mô tả |
|---|---|---|
| `_id` | `ObjectId` | Khóa chính do MongoDB tự tạo |
| `document_id` | `ObjectId` | Tham chiếu tới `_id` của collection `documents` |
| `chunk_index` | `int` | Số thứ tự chunk trong một document |
| `section_header` | `string \| null` | Tiêu đề mục hoặc phần chứa chunk |
| `chapter` | `string \| null` | Thông tin chương của chunk |
| `article` | `string \| null` | Thông tin điều của chunk |
| `clause` | `string \| null` | Thông tin khoản của chunk |
| `chunk_text` | `string \| null` | Nội dung text của chunk |
| `char_count` | `int \| null` | Số lượng ký tự của chunk |
| `embedding_status` | `string` | Trạng thái embedding, hiện tại thường là `pending` |
| `created_at` | `string` | Thời gian tạo chunk, đang lưu dưới dạng ISO datetime string |

### Collection: `pipeline_logs`

| Field | Type | Mô tả |
|---|---|---|
| `_id` | `ObjectId` | Khóa chính do MongoDB tự tạo |
| `phase` | `string` | Tên giai đoạn pipeline ghi log |
| `status` | `string` | Trạng thái của lần chạy |
| `document_count` | `int` | Số lượng document được xử lý hoặc load |
| `chunk_count` | `int` | Số lượng chunk được xử lý hoặc load |
| `message` | `string` | Nội dung mô tả ngắn về kết quả chạy |
| `timestamp` | `string` | Thời điểm ghi log, đang lưu dưới dạng ISO datetime string |
