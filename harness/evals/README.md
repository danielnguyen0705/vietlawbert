# VietLawBERT evaluation scaffold

Thư mục này giữ scaffold đánh giá RAG/BERT trong tương lai. File `golden_queries.json` hiện chỉ chứa ví dụ synthetic, không phải benchmark thật và không được dùng để công bố điểm mô hình.

Các metric dự kiến: Recall@K, MRR, Hit Rate, document retrieval correctness, article retrieval correctness, answer citation correctness, và tỷ lệ câu trả lời có nguồn kiểm chứng.

Khi bổ sung golden set thật, mỗi câu hỏi cần có provenance, document ID ổn định, điều/khoản kỳ vọng nếu có, và ghi chú rõ dữ liệu train/test để tránh leakage.
