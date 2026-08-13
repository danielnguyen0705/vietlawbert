# Kiến trúc VietLawBERT và Harness Engineering v1

## Pipeline dữ liệu

```mermaid
flowchart TD
    A[VBPL sitemap discovery] --> B[Public metadata extraction]
    B --> C[Content adapter: HTML/PDF]
    C --> D[Raw-first records]
    D --> E[Harness validation]
    E --> F{Accepted?}
    F -->|Yes| G[Clean and legal structure parsing]
    F -->|No| H[Review queue with explicit reason]
    G --> I[Chunk validation]
    I --> J[Embedding / Vector index]
    J --> K[RAG retrieval]
    K --> L[Answer generation with citations]
```

## Feedback loop cho AI agent

```mermaid
flowchart TD
    A[Receive task] --> B[Read README, AGENTS, docs]
    B --> C[Inspect existing implementation and tests]
    C --> D[Make smallest reasonable change]
    D --> E[Run syntax checks and tests]
    E --> F[Run harness validation]
    F --> G{Pass?}
    G -->|No| H[Diagnose, fix, rerun]
    H --> E
    G -->|Yes| I[Produce report]
```

Harness v1 không thay crawler đang chạy. Nó bọc pipeline bằng check có thể chạy trong CI: content quality, metadata, duplicate, JSON/JSONL, report JSON nhỏ. Mục tiêu là phát hiện dữ liệu độc hại cho corpus trước khi dữ liệu đi vào cleaning, chunking, embedding, Qdrant/Milvus, hoặc training.

## Nguyên tắc thiết kế

Ưu tiên: correctness, legal-data quality, reproducibility, observability, performance, corpus size. Không tăng corpus size bằng cách nhận record thiếu nội dung, security page, duplicate placeholder, hoặc metadata sai. Các rule có khả năng reject nhầm phải cấu hình được, test được, và ghi rõ giới hạn.
