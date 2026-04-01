# VietLawBERT - Stage 1 Setup

Repo: `https://github.com/danielnguyen0705/vietlawbert`

## Mục tiêu

Giai đoạn 1 tập trung vào:

- chạy MongoDB bằng Docker
- chạy FastAPI bằng Docker
- load dữ liệu vào MongoDB
- cung cấp API để xem document và chunk

## Quy trình nhanh

Nếu muốn chạy Stage 1 theo cách ngắn gọn nhất:

```powershell
git clone https://github.com/danielnguyen0705/vietlawbert.git
cd vietlawbert
git checkout -b cuong
docker compose up -d --build
docker compose exec app python -m app.db.init_indexes
docker compose exec app python phase4_load_to_mongodb.py
```

Sau đó mở:

- `http://localhost:8000/health`
- `http://localhost:8000/db-check`
- `http://localhost:8000/docs`
- `http://localhost:8000/documents`

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

Thông thường không cần sửa gì để chạy Stage 1 trên máy local.

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

Mở trình duyệt:

```text
http://localhost:8000/health
```

### Kiểm tra kết nối DB

Mở trình duyệt:

```text
http://localhost:8000/db-check
```

### Swagger UI

Mở trình duyệt:

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
http://localhost:8000/search?keyword=giao%20thong
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

Lệnh này chạy lần lượt:

- `phase1_collect_links.py`
- `phase2_parse_documents.py`
- `phase3_chunk_documents.py`
- `python -m app.db.init_indexes`
- `phase4_load_to_mongodb.py`

```powershell
docker compose exec app python run_full_pipeline.py
```

### Reset trước rồi chạy lại từ đầu

```powershell
docker compose exec app python reset_project_state.py --yes
docker compose exec app python run_full_pipeline.py
```

## 11. Các lệnh thường dùng

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

## 12. Trạng thái hiện tại

Hiện tại hệ thống đã làm được:

- chạy FastAPI bằng Docker
- chạy MongoDB bằng Docker
- kết nối app với MongoDB
- tạo index MongoDB
- load document vào database
- xem dữ liệu qua API
- search text trên collection `chunks`

Lưu ý:

- nếu chưa có `chunks.jsonl` thì collection `chunks` sẽ chưa có dữ liệu
- khi đó endpoint `/documents/{document_id}/chunks` và `/search` có thể không có kết quả
- nếu muốn tạo dữ liệu đầy đủ từ đầu, hãy chạy `docker compose exec app python run_full_pipeline.py`

## 13. Cấu trúc dữ liệu MongoDB

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
