# VietLawBERT — Bảng đặc tả chi tiết tiến độ thực hiện (v4.1, 13/11/2026)
 
> Tổng hợp từ: **Đề cương v3.1** (bảng đối soát VNLawBERT↔VietLawBERT), **Ghi chú họp** (thầy Sơn + anh Khánh), **các quyết định kiến trúc hai bên đã thống nhất**, và toàn bộ nhật ký thực thi tới 13/11/2026 — đặc biệt **vòng kiểm toán chất lượng ngữ liệu + vá tận gốc 11–13/11**.
 
**Quy ước trạng thái:** 🔴 Chưa bắt đầu · 🟡 Đang làm · 🟢 Hoàn thành · ⚪ Tạm hoãn.
`🟢 code` = đã viết + compile OK + test xanh trên CPU. `🟢 verify` = đã kiểm chứng bằng **số thật** (cần hệ thống sống/cloud).
 
**Nguồn trích:** [ĐC]=Đề cương v3.1 · [Họp]=thầy Sơn · [Khánh]=anh Khánh · [TN]=thống nhất hai bên · [AUDIT]=kiểm toán 11–13/11 · [VLQA]=benchmark · [PF]=promptfoo · [DIAG]=chẩn đoán cơ chế.
 
---
 
## I. BẢNG ĐỐI SOÁT KHOA HỌC CỐT LÕI (Đề cương v3.1) + TRẠNG THÁI
 
| Tiêu chí | Nền tảng VNLawBERT (2020) | Đề xuất VietLawBERT-MRL (2026) | Trạng thái |
| :-- | :-- | :-- | :-- |
| Quy mô ngữ liệu | 23.254 VB tĩnh, phân đoạn cơ học | VBQPPL quốc gia từ vbpl.vn | 🟡 **65.280 doc_number phân biệt** (ratio 1.0); phát biểu quy mô theo `count(DISTINCT doc_number)` chứ không theo ~160k trang |
| Bảo toàn cấu trúc | Cắt đoạn phẳng, mất ngữ cảnh cấp trên | AST + tiêm `[META]`/`[HIERARCHY]` | 🟢 code — **gia cố mạnh 11–13/11** (reset điều, chống node cụt, bảng pipe) |
| Khai phá mẫu đối lập | Ghép ngẫu nhiên / BM25 | GG-SLM trên HIN (Neo4j) | 🟡 code — **đã tinh chỉnh định nghĩa cực tính** (§II.1); re-mine ở cloud |
| Hàm mục tiêu | MLM + phân loại cặp câu | **Matryoshka InfoNCE Loss** | 🔴 Chưa bắt đầu (cần GPU) |
| Độ phân giải biểu diễn | Cố định 768d | Đa chiều {64…1024} | 🔴 Chưa bắt đầu (cần GPU) |
| Động cơ truy xuất | Bi-Encoder đơn trên văn bản thô | **Hybrid 3 tầng RRF** (Vector+BM25+Graph) | 🟡 graph-EXPANSION 🟢 code; cross-encoder 🟢 code; verdict chờ corpus cloud |
| Căn cứ hiệu lực | Không nhận thức thời gian | Theo vết 22 quan hệ, loại điều hết hiệu lực | 🟡 — **thêm version-VBHN 11–13/11** (node `version`/`canonical_group`) làm nền phân giải hiệu lực thời gian |
 
---
 
## II. CÁC QUYẾT ĐỊNH KIẾN TRÚC HAI BÊN ĐÃ THỐNG NHẤT (so với đề cương gốc)
 
| # | Quyết định | Lý do / điểm đánh đổi | Trạng thái |
| :-- | :-- | :-- | :-- |
| TN-1 | **Triplet theo QUAN HỆ pháp lý, nhưng `thay-thế/bãi-bỏ` KHÔNG làm negative cho embedding** | Hai VB ở quan hệ thay thế là *cùng chủ đề, gần trùng nội dung, chỉ khác hiệu lực* → lấy làm negative sẽ **đầu độc không gian ngữ nghĩa**. Hiệu lực thuộc **tầng đồ thị lọc** (đúng đề cương "22 quan hệ, loại điều hết hiệu lực"), không thuộc loss. Positive = `căn cứ/hướng dẫn/quy định chi tiết` + cặp (query↔điều-gold VLQA); hard-negative = gần từ vựng **khác chủ đề** (BM25 top-k trừ gold) | 🟡 chốt định nghĩa; áp dụng khi re-mine cloud |
| TN-2 | **VLQA khớp ở cấp ĐIỀU** (`relevant_laws = [aid]`), `law_id` là cầu nối văn bản | Gold của VLQA ở cấp điều; đo ở cấp văn bản vừa lỏng vừa lệch mục tiêu truy xuất điều/khoản. Trùng số hiệu phân biệt bằng (ngày ký + cơ quan) | 🟡 chốt; **P4 hiện khớp cấp văn bản → cần nâng lên cấp điều** |
| TN-3 | **Chống trùng khóa VBHN bằng VERSION** | Số hiệu VBHN đánh lại theo năm (vd `30/VBHN-NHNN` có **6 bản** khác nội dung trên vbpl). Mỗi bản 1 node riêng (không mất bản); cạnh (chỉ có số hiệu trần) trỏ **bản mới nhất** | 🟢 code 11–13/11 (§III.C) |
| TN-4 | **Corpus KHÔNG mặc định "sạch" — phải ĐO** | Chỉ số cấp số hiệu (ratio=1.0) cho bức tranh sạch giả; rác thật nằm ở content/hierarchy/chunk/OCR. Dựng auditor đo tỉ lệ lỗi có khoảng tin cậy | 🟢 code 11–13/11 (§III) |
| TN-5 | VLQA = thước đo, KHÔNG train; dùng trong phạm vi Biên bản | Theo [Họp] 24/10 | 🟢 giữ nguyên |
| TN-6 | Nền "xanh" CPU trước, mới thuê GPU; "làm đến đâu kiểm thử đến đó" | Kiểm soát chi phí A100, chống hồi quy | 🟢 toàn suite **182 passed** |
 
---
 
## III. ⚡ VÒNG 11–13/11/2026 — KIỂM TOÁN CHẤT LƯỢNG NGỮ LIỆU + VÁ TẬN GỐC
 
> Khởi từ phản biện về các chunk bẩn (số hiệu lạ, OCR vỡ, bảng dẹt, node cụt, chữ ký lọt). Đã **tự kiểm chứng trên 5 file dữ liệu thật** trước khi kết luận, vá tận gốc 9 nhóm, **chạy test thật trên Ubuntu**.
 
### A. Auditor toàn-corpus + baseline số thật (mẫu Cochran-382, CI 95% Wilson)
 
Bộ `quality/corpus_audit/` (7 detector thuần + CI) — chạy được toàn corpus (`--source jsonl` từ dump Mongo) lẫn mẫu. **Baseline "before" trên mẫu 382:**
 
| Khuyết tật | Tỉ lệ ước lượng | CI 95% | Tính chất |
| :-- | :-- | :-- | :-- |
| doc_type sai (loại chuẩn bị dồn "Khác") | **16,5%** | [13,1–20,5] | Classifier — sửa rẻ, không cào lại |
| Hierarchy tùy hứng (cấp phi cấu trúc / Điều lệch) | **11,0%** | [8,2–14,5] | Parser — sửa rẻ |
| OCR/encoding vỡ | **6,0%** | [4,1–8,9] | Nguồn — đắt (OCR lại) |
| Rủi ro trùng khóa (sàng lọc VBHN) | **5,5%** | [3,6–8,3] | Khóa định danh |
| Node cụt — **lá mất nội dung thật** | **3,4%** | [2,0–5,7] | Parser — mất dữ liệu |
| Đuôi chữ ký/thi hành còn giữ | 1,8%* | [0,9–3,7] | *dưới ước lượng (mẫu cắt 1200) |
| Bảng bị dẹt | 1,1%* | [0,4–2,7] | *dưới ước lượng |
 
### B. Chín nhóm vá (file · lỗi gốc · bản vá · test)
 
| # | File | Lỗi gốc | Bản vá | Test |
| :-- | :-- | :-- | :-- | :-- |
| 1 | `configs/global_id.py` → `doc_type_and_rank` | Chỉ xét đoạn sau "/" → `60-TC`, `05-BXD`, `HĐBT`, `QĐ-UB-VX`, VBHN rớt "Khác" | Quét **toàn ký hiệu** + cứu bằng **tiêu đề** (khớp đầu, tránh nhầm "Luật" giữa câu) + bucket "Văn bản hợp nhất" | 13 ca |
| 2 | `preprocess/ast_parser.py` | `flush_article_buffer` không reset số Điều → tiêu đề chương dính vào Điều cũ ("art_4 nội dung Điều 5") | Reset `current_art_num=None` tại **Phần/Chương/Mục** (4 chỗ); dòng mồ côi sau chương → `_section_` đúng ngữ cảnh (không dán "Căn cứ pháp lý") | 6 ca |
| 2b | `preprocess/ast_parser.py` | `re_attached` nuốt "QUY ĐỊNH CHUNG/CỤ THỂ" (tiêu đề chương) thành cấp hierarchy tùy hứng | Negative lookahead `(?!\s+(?:CHUNG\|CỤ THỂ\|CHI TIẾT\|CHUYÊN))` | ↑ |
| 2c | `preprocess/ast_parser.py` | Node cụt chỉ còn nhãn ("Điều 1.") làm nhiễu + rớt gate Tier-1 | Bỏ node nếu thân thực < 10 ký tự | ↑ |
| 3 | `preprocess/html_to_md.py` + `crawler/pipelines.py` | `soup.get_text("\n")` phá cấu trúc bảng → số lìa nhãn | `tables_to_markdown()` giữ bảng dạng pipe **trước** get_text/html2text; nối vào cả 2 đường | 3 ca |
| 4 | `crawler/ocr_runner.py` | Tesseract dpi=200, không chặn rác ("O6i vOi t6 chric" lọt) | Cổng `ocr_broken_ratio` → cách ly `OCR_LOW_QUALITY` (tái-OCR); nâng **dpi=300** | (trong 3 ca #3) |
| 5 | `preprocess/text_cleaner.py` | Khối chữ ký `T/M`,`K/T` (gạch chéo) + chức danh + tên người ký lọt vào corpus | `SIGNATURE_BLOCK_START` + `SIGNATURE_TITLE_LINE` + cờ `in_signature` loại tới ranh giới Điều kế | 2 ca |
| 6 | `crawler/shard_runner.py` | **Bug có sẵn:** gọi `get_subsystem_logger` ở dòng 20 **trước** import dòng 22 → `NameError` (chặn cả 1 test cũ) | Đưa import lên trước | (gỡ chặn) |
| 7 | `quality/corpus_audit/` (mới) | — | Auditor 7-detector + CI 95% + tách node-cụt lá/cấu trúc | 20 ca |
 
**doc_type — cải thiện thật trên mẫu:** 84 chunk "Khác" → **cứu 52 (61,9%)** về bucket đúng; phần còn lại là "Khác" đúng nghĩa (QĐ cấp bộ không có bucket).
 
### C. VERSION-VBHN (TN-3) — chống trùng khóa, giữ đủ bản
 
| Thành phần | Nội dung |
| :-- | :-- |
| `configs/global_id.py` | `is_collision_prone` (chỉ VBHN) · `canonical_doc_id(doc_number, sign_date, issuer)` **chỉ trộn version cho VBHN** (số thường giữ id CŨ → không phá ES/cạnh) · `compute_record_doc_id(record)` **dùng chung** cho ingest & build_graph · `resolve_target_id(..., latest_version_map)` trỏ bản mới nhất · `detect_collisions` |
| `database/build_hin_graph.py` | Pre-scan **toàn cục mọi shard** chọn bản ngày-ký-lớn-nhất mỗi nhóm VBHN; node mang `d.version` + `d.canonical_group`; cạnh resolve về bản mới nhất; `doc_type_and_rank(doc_number, title)` |
| `pipeline/ingest_pipeline.py` | Dùng **chung** `compute_record_doc_id` → ES/Qdrant khớp Neo4j |
| Kiểm chứng an toàn | tier2 `test_id_consistency` **PASS** → thay đổi không phá đồng nhất ES↔Neo4j; 5 test version (2 bản khác ngày → khác id; số thường không đổi id; cạnh về bản mới nhất) |
 
### D. Kết quả kiểm thử (chạy thật trên Ubuntu dell-G7-7588)
 
- `tests/corpus_fixes/` : **29 passed** (ast 6 · global_id 18 · table+ocr 3 · text_cleaner 2).
- `quality/corpus_audit/` : **20 passed**.
- Toàn suite: **182 passed, 2 skipped, 3 xfailed, 0 failed** (2 skip = cổng chờ dữ liệu cloud: Neo4j chưa có cạnh gold / index chưa embed; 3 xfail đúng thiết kế).
- Git: 4 commit (`deca72d`→`8d5e504`), cây sạch.
> ⚠️ Bản vá nằm ở **khâu nạp** (parser/cleaner/doc_type/OCR/khóa) → chỉ hiện số đẹp **sau re-ingest cloud**. Số "after" thật lấy bằng cách dump Mongo chạy auditor, đối chiếu baseline §III.A.
 
---
 
## IV. BẢNG TIẾN ĐỘ THEO WORK PACKAGE
 
### WP1 — CORPUS
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 1.1 Dòng chảy thu thập (crawl → shard) | Thu đủ VBQPPL quốc gia từ vbpl.vn, shard hoá, idempotent | 🟡 Đang làm | 66.421 node / 65.280 doc_number; cào lại trên cloud |
| 1.3 Zero-orphan | Không còn node chunk mồ côi | 🟢 Hoàn thành | |
| 1.4 Tỉ lệ ký tự Việt ≥ 0.05 | Lọc tệp rác / không phải tiếng Việt | 🟢 Hoàn thành | TextGate |
| 1.5 AST + cap `macro_label` ≤ 200 | Bóc cấu trúc 4 cấp, nhãn gom không vỡ index RANGE | 🟢 Hoàn thành | |
| 1.5b Vá AST sâu [AUDIT] | Hierarchy đúng, không node cụt / tiêu đề lẫn điều, bảng giữ cấu trúc | 🟢 code | §III.B; verify sau re-ingest |
| 1.6 Lấy mẫu Cochran độ sạch | Ước lượng % corpus bẩn với CI 95% | 🟡 Đang làm | mẫu 382 đã sinh (`cochran_sample_382.csv`); chờ gán nhãn tay |
| 1.7 Nạp đủ corpus (embed) | Toàn kho có vector trên Qdrant | 🔴 Chưa bắt đầu | cần GPU |
| 1.8 Parser văn bản 1-dòng | Khôi phục cấu trúc khi HTML bị phẳng | 🟢 code | `_resplit_if_flat` |
| 1.9 OCR gate + dpi300 [AUDIT] | Không cho text OCR vỡ lọt corpus | 🟢 code | cách ly `OCR_LOW_QUALITY` |
 
### WP2 — ĐỒ THỊ HIN
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 2.1 Gold cạnh | Bộ cạnh chuẩn để đo recall | 🟢 Hoàn thành | |
| 2.2 Recall cạnh | Khôi phục đủ cạnh từ lược đồ | 🟢 Hoàn thành | 1718/1718 |
| 2.3 UNMAPPED ≤ 5% | ≤ 5% quan hệ không map được loại | 🟡 Đang làm | cần soát bảng ontology |
| 2.4 Chiều/thời gian quan hệ | Gắn hướng + mốc thời gian cho cạnh | 🟢 Hoàn thành | |
| 2.5 Dựng cạnh + doc_type/rank | Node mang loại VB + bậc hiệu lực | 🟢 Hoàn thành | |
| 2.7 Khử trùng node (canonical) | 1 doc_number = 1 node | 🟢 verify | 65.280/65.280 ratio 1.0 |
| 2.8 Version-VBHN [TN-3] | Mỗi bản VBHN 1 node; cạnh → bản mới nhất | 🟢 code | §III.C |
 
### WP3 — TRAIN DATA
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 3.1 Positive | Tập cặp gần / bổ trợ | 🟢 Hoàn thành | 49.759 (bản cũ) |
| 3.2 Negative | Tập cặp đối lập | 🟢 Hoàn thành | 10.000 (bản cũ) |
| 3.3 Kiểm 50 triplet cực tính | Xác nhận pos/neg đúng cực tính | 🔴 Chưa bắt đầu | **sau re-mine cloud** (triplet hiện là bản cũ) |
| 3.4 Miner GG-SLM | Đào triplet từ HIN theo node | 🟢 Hoàn thành | positive gom theo NODE → sanitation tự làm sạch triplet |
| 3.5 Disjoint train/test | Không rò rỉ test vào train | 🔴 Chưa bắt đầu | |
| 3.6 Định nghĩa cực tính [TN-1] | thay-thế/bãi-bỏ → tầng graph, không làm negative | 🟡 Đang làm | áp dụng khi re-mine |
 
### WP4 — BENCHMARK
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 4.1 VLQA alignment | Gold VLQA khớp corpus | 🟢 verify | 97,3% (2.130/2.190); 60 miss đã xuất `vlqa_miss.csv` |
| 4.6 Thẩm định 200 câu | Bộ câu hỏi chuẩn để đo | 🟡 Đang làm | chờ thẩm định tay |
| 4.8 Multi-hop v2 | Câu hỏi suy luận nhiều bước | 🟢 Hoàn thành | cần tự nhiên hoá câu chữ |
| 4.9 P4 eval ra SỐ | Hit/MRR/NDCG/MAP@K + P95 + t-test | 🟢 code | `--rq vlqa` |
| 4.10 Nâng P4 khớp cấp ĐIỀU [TN-2] | Đo ở cấp điều theo `relevant_laws=aid` | 🔴 Chưa bắt đầu | hiện khớp cấp văn bản |
 
### WP5 — GLOBAL ID & METRIC
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 5.1 Global ID canonical | doc_id tất định từ doc_number | 🟢 code | |
| 5.2 Metric theo cấp | Đo ở cấp văn bản & điều | 🟢 Hoàn thành | |
| 5.3 6 IR metric | Hit / MRR / NDCG / MAP | 🟢 code | |
| 5.4 Latency P95 | Đáp ứng SLA P95 < 500ms | 🟢 Hoàn thành | |
| 5.5 fail_category | Phân loại ca lỗi | 🟢 Hoàn thành | |
| 5.6 Vệ sinh doc_number | classify + cổng canonical | 🟢 code | 25/25 ví dụ PASS |
| 5.7 doc_type quét toàn ký hiệu + tiêu đề [AUDIT] | Gán đúng loại VB, không dồn "Khác" | 🟢 code | cứu 61,9% "Khác" |
| 5.8 `compute_record_doc_id` dùng chung [TN-3] | ingest ↔ graph không lệch khóa | 🟢 code | |
 
### WP6 — FINE-TUNE (GPU)
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 6.2 Thử @256d | Chạy thử một lát cắt chiều | 🟡 Đang làm | |
| 6.3 Fine-tune MRL (Matryoshka InfoNCE) | Mô hình đa chiều hội tụ mượt, đơn điệu loss | 🔴 Chưa bắt đầu | cần GPU |
| 6.5 Experimental setup | Cấu hình huấn luyện chuẩn, tái lập được | 🔴 Chưa bắt đầu | cần GPU |
 
### WP7 — RETRIEVAL
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 7.1 Chẩn đoán điểm nghẽn | Hiểu đúng nguyên nhân số thấp | 🟢 Hoàn thành | |
| 7.3 Cross-encoder `bge-reranker-v2-m3` | Tái xếp hạng tăng chính xác | 🟢 code | bật qua `.env RERANKER_MODEL` |
| 7.4 Graph-EXPANSION | Duyệt neighbor bổ trợ tăng recall | 🟢 code | |
| 7.5 Seed top-M | Chọn seed mở rộng đồ thị | 🔴 Chưa bắt đầu | |
 
### WP8 — GENERATION + GUARDRAIL (RQ4)
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 8.1 Abstain | Từ chối khi score thấp / rỗng | 🟢 code | |
| 8.2 Injection | Chặn lách luật / jailbreak | 🟢 code | |
| 8.3 Grounded reasoning | Suy luận có căn cứ, chống ảo giác | 🟡 Đang làm | |
| 8.5 Citation precision | Trích dẫn chính xác điều/khoản | 🟢 code | |
 
### WP9 / WP11 / WP12 — HẠ TẦNG & CHATBOX
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 9.1/9.4/9.5/9.6 Hạ tầng | LLM endpoint · dataset · checkpoint · graph-walk | 🟢 Hoàn thành | |
| WP11 App / backend / logging | Chatbox vận hành + ghi log tương tác | 🟢 Hoàn thành | |
| WP11b Streaming `ask_stream` | Trả lời theo luồng token | 🟢 code | wiring Gradio handler còn lại |
| WP11c Ablation dropdown | Chọn chế độ truy xuất trên UI | 🟢 Hoàn thành | có sẵn |
| WP12 (12.2–12.12) | Hoàn thiện tính năng chatbox | 🔴 Chưa bắt đầu | |
 
### WP13 — CỨU GRAPH
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 13.1 Recall cạnh | Đủ cạnh so với gold | 🟢 Hoàn thành | 1718/1718 |
| 13.2 EXPANSION | Tăng recall bằng duyệt đồ thị | 🟢 code | |
| 13.3 pos/neg | Cặp đối lập đúng | 🟢 Hoàn thành | |
| 13.4 Multi-hop v2 | Câu hỏi nhiều bước | 🟢 Hoàn thành | |
| 13.5 Ablation end-to-end (verdict) | Full > Hybrid > Dense/Sparse | 🟡 Đang làm | chờ corpus cloud đủ lớn |
| 13.6 DIAG cơ chế | Khẳng định cơ chế graph đúng | 🟢 Hoàn thành | |
| 13.7 Gia cố `execute_batch` | Tái kết nối + đếm lô rớt, không mất âm thầm | 🟢 Hoàn thành | |
 
### WP14 — MCP §V + KIỂM THỬ TIER 4/5
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| 14.1 MCP servers | 3 tool (retrieval/graph/validator) Pydantic + FastMCP guarded | 🟢 code | |
| 14.2 promptfoo Tier-4/5 | 2 yaml + `vlqa_to_promptfoo` (gold thật) | 🟢 code | |
| 14.3 Wipe nhanh | `drop_database` CALL-in-transactions + fallback | 🟢 code | |
 
---
 
## V. SỐ THỰC NGHIỆM CHỐT (BASELINE — dùng để đối chiếu "after")
 
| Chỉ số | Giá trị baseline | Ghi chú |
| :-- | :-- | :-- |
| Node LawDocument | 66.421 | |
| doc_number phân biệt | 65.280 (ratio 1.0) | phát biểu quy mô theo số này |
| UNKNOWN / orphan | 1.141 (1,7%) / 3.128 (4,7%) | |
| VLQA alignment | 97,3% (2.130/2.190) | 60 miss (2,7%) đã liệt kê |
| doc_type sai (mẫu 382) | 16,5% → sau vá cứu 61,9% "Khác" | |
| Hierarchy tùy hứng | 11,0% | |
| OCR vỡ | 6,0% | |
| Node cụt lá (mất nội dung) | 3,4% | |
| `30/VBHN-NHNN` | 6 bản khác nội dung trên vbpl | ca điển hình version |
| Test | 182 passed / 2 skip / 3 xfail / 0 fail | toàn suite Ubuntu |
 
---
 
## VI. CÁC HẠNG MỤC CÒN LẠI
 
> Khâu code pre-cloud **đã đóng** (9 nhóm vá + version-VBHN + auditor, tất cả có test, chạy thật xanh). Còn lại:
 
| Hạng mục | Kỳ vọng đạt được | Trạng thái | Ghi chú |
| :-- | :-- | :-- | :-- |
| Chạy cloud một mạch | `bash auto_cloud_runner.sh <IP> <PORT> <USER>` → crawl→ingest→graph→embed→eval; RQ1–4 + t-test/t-SNE | 🔴 Chưa bắt đầu | 24–36h/A100 |
| Đo "after" bằng auditor | Dump Mongo → `audit_corpus.py` → số thật đối chiếu baseline §III.A | 🔴 Chưa bắt đầu | sau re-ingest |
| Nâng P4 lên cấp điều [TN-2] | Đo benchmark theo `relevant_laws=aid` | 🔴 Chưa bắt đầu | việc code |
| Re-mine triplet theo cực tính [TN-1] | Triplet sạch theo quan hệ, hiệu lực sang tầng graph | 🔴 Chưa bắt đầu | sau re-ingest |
| Gán nhãn / thẩm định tay | DATA-1, 3.3, 1.6, 4.6, 2.3, 3.5, 4.8 | 🔴 Chưa bắt đầu | quyết cách làm sau |
| Fine-tune MRL + viết Experiments | Số RQ1–4 so VNLawBERT, bảng kết quả | 🔴 Chưa bắt đầu | sau khi có số cloud |
 
---
 
## VII. NGUYÊN TẮC XUYÊN SUỐT
 
1. VLQA = thước đo, không train; dùng trong phạm vi Biên bản.
2. Train = pos/neg **đúng cực tính theo quan hệ** (TN-1); hiệu lực là việc của **tầng đồ thị**, không của loss.
3. "Chính xác" / sạch corpus = **đo bằng auditor có CI**, không giả định sạch (TN-4).
4. Graph = EXPANSION tăng recall, KHÔNG re-score; verdict chốt ở corpus đủ (cloud).
5. Retrieval tất định; Generation = RAGAS + human.
6. Nền "xanh" CPU → mới thuê GPU; "làm đến đâu kiểm thử đến đó".
7. Khóa liên-kho = `doc_number` qua `classify_doc_number` (VALID→canonical · NO_NUMBER→khóa riêng · CONTAMINATED→cách ly); **VBHN thêm version**; `compute_record_doc_id` **dùng chung** ingest↔graph.
8. Cạnh VBHN (số hiệu trần) → trỏ **bản mới nhất** toàn cục.
9. Quy mô phát biểu theo `count(DISTINCT doc_number)` (~65k), không theo số trang (~160k).
10. Mọi lô nạp **đếm & báo cáo bản ghi bị bỏ**; không mất âm thầm.
11. "Title leak" là lỗi **tầng crawler** (extraction) — soi lại khi cào cloud.
12. Logger qua `get_subsystem_logger`; ngoại lệ cố ý: config bootstrap, Scrapy settings, tiện ích test, `__main__` demo.
13. Test integration phụ thuộc dữ liệu phải **data-aware** (skip khi tiền đề chưa thỏa, không fail giả).
14. "Số thấp" / bất thường phải **truy tới mắt xích gốc** (file/tầng) trước khi kết luận — không vá mò.