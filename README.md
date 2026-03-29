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
