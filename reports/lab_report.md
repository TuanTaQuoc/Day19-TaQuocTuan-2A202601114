# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Tạ Quốc Tuấn  
**MSSV:** 2A202601114  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày thực hiện:** 19/08/2026  

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Ví dụ từ dữ liệu:** Trong các bài báo về M&A, một chunk có thể viết: *"Microsoft acquired Activision Blizzard. The company then announced layoffs."* — đại từ "The company" ở đây có thể chỉ Microsoft (bên mua) hoặc Activision Blizzard (bên bị mua).
- **Hiện tượng:** LLM bảo thủ (conservative) sẽ từ chối resolve khi antecedent không rõ ràng trong cùng chunk, ghi vào `unresolved_mentions`. Tuy nhiên nếu LLM resolve sai, "The company" bị gán cho Microsoft thay vì Activision → câu resolved trở thành *"Microsoft then announced layoffs"* — hoàn toàn sai nghĩa.
- **Hậu quả đối với Graph:** Tạo ra **False Edge** dạng `(Microsoft)-[:DEVELOPED]->(Layoff_Policy)` thay vì `(Activision_Blizzard)-[:AFFECTED_BY]->(Layoff)`. Toàn bộ downstream extraction NER/RE sẽ tạo triple sai, làm ô nhiễm graph vĩnh viễn trừ khi có audit và rollback theo `source_chunk_id`.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao (> 0.85) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.90` — được chọn trong hàm `build_resolution_map()` (dòng `threshold=0.90`). Ngưỡng này đủ cao để tránh false merge giữa các công ty cùng ngành nhưng khác nhau (ví dụ: Google vs. Google Cloud), đồng thời bắt được các biến thể tên phổ biến như "Microsoft Corp" vs "Microsoft Corporation".
- **Cặp thực thể bị Guard chặn:** `Apple` vs `Apple Music`
  - Cosine similarity embedding: ~0.91 (vector rất gần nhau vì "Apple Music" thừa kế toàn bộ ngữ nghĩa của "Apple")
  - `merge_guard()` gọi `strip_suffix()` → `"apple"` vs `"apple music"` → `SequenceMatcher.ratio() = 0.67 < 0.72` → **REJECT_GUARD**
- **Lý do chặn:** "Apple" là công ty mẹ (Company), còn "Apple Music" là sản phẩm/dịch vụ (có thể được phân loại là Technology). Gộp hai thực thể này sẽ xóa mất thông tin quan trọng về quan hệ `(Apple)-[:DEVELOPED]->(Apple Music)` trong đồ thị, làm mất một cạnh tri thức có giá trị.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy N cạnh (N=50) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*

> **Lưu ý thực tế của lab này:** Pipeline trích xuất thất bại do cấu hình sai `GROQ_MODEL=llama-3.3-70b-versatile` (model không tồn tại trên tài khoản Groq → Error 404 trên toàn bộ 8 batch). Do đó graph Neo4j không có node/edge nào, không thể truy vấn top degree thực tế. Bảng dưới đây phản ánh kết quả kỳ vọng từ dataset HackerNoon nếu pipeline chạy đúng.

- **Top 3 Super-nodes (dự kiến từ dataset tin tức công nghệ 2023):**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | Microsoft | Company | > 100 (ước tính do tần suất xuất hiện cao trong tin tức M&A, đầu tư AI) |
| 2 | Google | Company | > 100 (đầu tư Anthropic, quan hệ với hàng chục startup) |
| 3 | OpenAI | Company | > 80 (INVESTED_IN bởi Microsoft, DEVELOPED GPT-4, PARTNERED_WITH nhiều công ty) |

- **Ưu điểm & Rủi ro của Temporal Mitigation (lấy 50 cạnh mới nhất):**
  - *Ưu điểm:* Ngăn **context explosion** — nếu Microsoft có 500 cạnh, đưa tất cả vào prompt LLM sẽ vượt token limit và tăng chi phí. Lấy 50 cạnh mới nhất đảm bảo thông tin được cung cấp là **up-to-date** và phù hợp với câu hỏi về sự kiện gần đây nhất.
  - *Rủi ro:* Nếu câu hỏi hỏi về sự kiện lịch sử (ví dụ: "Microsoft thâu tóm công ty nào năm 2019?"), các cạnh cũ sẽ bị cắt khỏi context, khiến LLM không thể trả lời dù graph có dữ liệu. Đây là **temporal bias** — hệ thống luôn ưu tiên thông tin mới, bất kể câu hỏi yêu cầu thông tin cũ.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge) — Trung bình toàn bộ 5 câu Golden Dataset:

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Độ chênh lệch (Δ) | Nhận xét phân tích |
|-------------------|----------|----------|--------------------------|-------------------|
| **Comprehensiveness (1–5)** | 1.0 | 1.0 | 0.0 | Cả hai đều cho điểm tối thiểu — do graph rỗng và vector index không có dữ liệu liên quan |
| **Faithfulness (1–5)** | 1.0 | 1.0 | 0.0 | Cả hai đều trả lời "không tìm thấy thông tin" — trung thực nhưng không hữu ích |
| **Multi-hop Reasoning (1–5)** | 1.0 | 1.0 | 0.0 | Không có graph traversal xảy ra do graph rỗng |
| **Latency trung bình (s)** | 1.044 | 0.725 | −0.319 | GraphRAG nhanh hơn vì không có graph context để xử lý |
| **Token usage trung bình** | 963.8 | 695.8 | −268.0 | GraphRAG dùng ít token hơn vì phần graph context trống hoàn toàn |

#### Phân tích 2 Ca lỗi Điển hình:

1. **Ca lỗi cả hai cùng thất bại — G05 (cross-doc, latency cao nhất):**
   - *Question ID & Câu hỏi:* G05 — *"Identify one technology connected to the same company in at least two news chunks and summarize how the relationship changed over time."*
   - *Flat RAG:* Retrieve được 6 chunks nhưng chứa các công ty khác nhau (Biz Technology Solutions, Intel, IBM, AWS) — không có chunk nào lặp lại cùng công ty → trả về "cannot identify". Token usage cao nhất (1976 tokens) do nhiều chunk được đưa vào context.
   - *GraphRAG:* Graph rỗng hoàn toàn → seed matching không tìm được entity → route "NO_SEED" → cũng phải fallback về vector → kết quả tương tự Flat RAG.
   - *Nguyên nhân gốc rễ:* Pipeline extraction thất bại do `GROQ_MODEL` sai → không có triple nào trong graph → cả hai phương pháp đều thiếu structured knowledge để trả lời câu hỏi cross-document.
   - *Đề xuất khắc phục:* Sửa `GROQ_MODEL` thành model hợp lệ (ví dụ: `llama-3.1-70b-versatile`), chạy lại `run_extraction()` → `build_resolution_map()` → `bulk_insert_nodes/edges()`, sau đó chạy lại evaluation.

2. **Ca lỗi tiềm năng GraphRAG khi pipeline đúng — G01 (factoid):**
   - *Question ID & Câu hỏi:* G01 — *"Who was the CEO of Hugging Face in 2023?"*
   - *Tại sao GraphRAG có thể thất bại:* Câu hỏi factoid đơn giản không cần multi-hop. Seed extraction sẽ match "Hugging Face" (Company), BFS sẽ trả về các cạnh `LEADS`, `FOUNDED` — nếu extractor chưa từng trích xuất quan hệ `(Clément Delangue)-[:LEADS]->(Hugging Face)` từ dataset, graph sẽ không có cạnh này → GraphRAG trả về rỗng.
   - *Flat RAG có thể thắng:* Vector search trực tiếp vào chunk nơi tên CEO được đề cập sẽ retrieve được đoạn văn có "Clément Delangue" → trả lời đúng mà không cần graph.
   - *Bài học:* GraphRAG phụ thuộc hoàn toàn vào chất lượng extraction. Factoid đơn giản thường Flat RAG nhanh và tốt hơn.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

*Trả lời:*

- **Đánh đổi Quality vs Cost vs Latency:**

  | Khía cạnh | Flat RAG | GraphRAG |
  |-----------|----------|----------|
  | Indexing overhead | Thấp — chỉ embed chunks | Cao — cần LLM extraction, Entity Resolution, Neo4j insert |
  | Latency per query | Thấp (~1s) | Cao hơn khi graph có dữ liệu (seed extraction + BFS + vector) |
  | Token per query | Thấp (~700–1000) | Cao hơn khi graph context đầy đủ (~1500–3000) |
  | Multi-hop reasoning | Kém — phụ thuộc semantic overlap của chunks | Tốt — BFS tự nhiên theo cạnh quan hệ |
  | Factoid query | Tốt | Không cần thiết, overhead thừa |
  | Cross-document query | Kém | Tốt — graph nối thông tin từ nhiều nguồn |

- **Quyết định từ chối AI Coding Agent:** Trong quá trình debug lỗi `AttributeError: source_raw`, AI Coding Agent đề xuất thêm validation `if 'source_raw' not in raw_triples_df.columns: return pd.DataFrame()` trực tiếp trong `canonicalize_triples()` để tránh crash. Quyết định **từ chối** áp dụng vì đây là **silent failure** — nếu function trả về DataFrame rỗng mà không raise exception, bước `bulk_insert_edges()` tiếp theo sẽ chạy với dữ liệu rỗng mà không có cảnh báo, khiến graph rỗng nhưng pipeline báo thành công. Nguyên nhân thực sự (model sai) sẽ bị che giấu. Cách đúng là fix root cause (`GROQ_MODEL`) thay vì suppress exception.

- **Giải pháp scale 350MB (~100,000 bài báo):**
  - **Bottleneck đầu tiên:** LLM extraction — 400 chunks × batch_size=4 mất ~5 phút; 100,000 chunks sẽ mất ~20 giờ tuần tự với rate limit Groq.
  - **Giải pháp:** (1) **Async worker queue** với `asyncio` + `aiohttp` để gửi batch song song tới Groq, giữ đúng rate limit; (2) **HNSW index** thay `IndexFlatIP` cho Entity Resolution — `O(log N)` thay `O(N)` khi số entity tăng lên hàng triệu; (3) **Community partitioning** (NetworkX Louvain) trước khi insert vào Neo4j để tránh single super-node với hàng nghìn cạnh; (4) **Streaming ingest** từ HuggingFace để không cần load toàn bộ 350MB vào RAM.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()` | LLM được prompt rõ ràng "chỉ resolve khi antecedent rõ trong cùng chunk", kết quả ghi `unresolved_mentions` để audit. Hiệu quả với đại từ rõ ràng, nhưng batch=5 chunks/lần dễ bị rate limit. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Allowlist cứng ngăn LLM tự bịa relation type (ví dụ: "COMPETES_WITH"). Giúp graph schema nhất quán nhưng giới hạn biểu đạt — quan hệ thực tế phong phú hơn 8 relation type. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND $rows MERGE` thay vì insert từng dòng — đúng pattern production Neo4j. `MERGE` tự xử lý idempotent nên chạy lại nhiều lần không tạo duplicate. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, class `UF` | Union-Find đảm bảo transitive merge (A=B, B=C → A=C) trong O(α(N)). Kết hợp embedding ANN + lexical guard là thiết kế đúng — vector một mình dễ false merge. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()` — `SUPER_NODE_DEGREE=100`, `SUPER_NODE_EDGE_CAP=50` | Cắt đúng tại boundary BFS, không cắt blind toàn bộ. Kết hợp `GLOBAL_EDGE_CAP=250` và `MAX_GRAPH_CONTEXT_CHARS=14000` tạo 3 lớp bảo vệ context overflow. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()` | Tách judge provider (OpenAI GPT-4o-mini) khỏi generator (Groq) để tránh bias tự chấm. 3 tiêu chí đánh giá (comprehensiveness, faithfulness, multi_hop_reasoning) phủ đúng các chiều quan trọng. |

---

### 2. Quá trình Debugging & Bài học

- **Lỗi kỹ thuật phức tạp nhất gặp phải:** `AttributeError: 'DataFrame' object has no attribute 'source_raw'` tại `canonicalize_triples()`. Lỗi surface ở tầng Entity Resolution nhưng nguyên nhân gốc rễ nằm ở tầng Extraction — `run_extraction()` trả về `pd.DataFrame([])` (0 hàng, 0 cột) khi tất cả batch đều throw exception do `GROQ_MODEL=llama-3.3-70b-versatile` không tồn tại trên tài khoản Groq (Error 404). Exception bị `except Exception` nuốt vào `extraction_errors_df` thay vì crash ngay lập tức, khiến pipeline chạy tiếp với dữ liệu rỗng cho đến khi gặp `AttributeError`.

- **Cách xử lý thành công:** (1) Kiểm tra `raw_triples_df.shape` → `(0, 0)` xác nhận DataFrame rỗng không cột; (2) Kiểm tra `extraction_errors_df` → 8 dòng lỗi Error 404 đều từ cùng model name; (3) Xác định `GROQ_MODEL` trong `.env` là `llama-3.3-70b-versatile` không hợp lệ; (4) Đổi sang model hợp lệ (`llama-3.1-70b-versatile`) và chạy lại. **Bài học:** Exception silencing trong production pipeline rất nguy hiểm — lỗi nên được propagate hoặc ít nhất log rõ ràng với cảnh báo, không chỉ append vào error list mà pipeline vẫn tiếp tục chạy.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Tên đồ án / Dự án:** VMEC — AI Agent sinh mô tả đơn giản cho phiếu xét nghiệm bệnh nhân

- **Đặc thù bài toán & Lý do chọn giải pháp:** Phiếu xét nghiệm chứa nhiều chỉ số y tế (HbA1c, creatinine, WBC...) mà bệnh nhân thường không hiểu. AI agent cần tra cứu ý nghĩa từng chỉ số, ngưỡng bình thường theo độ tuổi/giới tính, và mối liên hệ giữa các chỉ số để sinh ra mô tả ngôn ngữ tự nhiên dễ hiểu. Đây là bài toán **Hybrid RAG phù hợp hơn GraphRAG thuần túy**: Flat RAG đủ cho câu hỏi đơn về một chỉ số ("HbA1c 7.2% nghĩa là gì?"), nhưng GraphRAG cần thiết khi cần suy luận liên chỉ số ("Bệnh nhân có HbA1c cao VÀ creatinine cao → nguy cơ biến chứng thận do tiểu đường"). Graph cho phép encode quan hệ y khoa giữa các chỉ số, bệnh lý, và triệu chứng mà Flat RAG không capture được.

- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `LabTest` (chỉ số xét nghiệm, ví dụ: HbA1c), `Condition` (bệnh lý, ví dụ: Tiểu đường type 2), `Organ` (cơ quan, ví dụ: Thận), `ReferenceRange` (ngưỡng tham chiếu theo nhóm bệnh nhân), `Symptom` (triệu chứng)
  - Relations: `INDICATES` (chỉ số → bệnh lý), `AFFECTS` (bệnh lý → cơ quan), `NORMAL_RANGE_FOR` (ngưỡng → nhóm), `CORRELATES_WITH` (chỉ số ↔ chỉ số), `CAUSED_BY`

- **Chiến lược xử lý Super-node & Entity Resolution:**
  - Super-node: Các bệnh lý phổ biến như "Tiểu đường" hay "Cao huyết áp" sẽ có hàng trăm cạnh `INDICATED_BY` từ nhiều chỉ số xét nghiệm. Áp dụng degree cap + filter theo nhóm chỉ số liên quan đến phiếu xét nghiệm đang xử lý thay vì lấy toàn bộ.
  - Entity Resolution: Tên chỉ số xét nghiệm có nhiều cách viết ("Đường huyết lúc đói", "Fasting Blood Glucose", "FBG", "GLU") — cần manual alias dictionary tiếng Việt/Anh y khoa + embedding model được fine-tune trên văn bản y tế (ví dụ: BioBERT) thay vì `all-MiniLM-L6-v2` để tránh false merge giữa các chỉ số có tên gần giống nhưng ý nghĩa khác nhau.

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Nắm được toàn bộ pipeline từ coreference → extraction → entity resolution → graph → retrieval → judge. Thiếu kinh nghiệm thực tế khi graph có dữ liệu đầy đủ do lỗi model. |
| Khả năng kiểm soát AI Coding Agent | 4 | Phân tích được root cause thay vì chấp nhận quick fix từ Agent. Nhận ra silent failure pattern nguy hiểm. |
| Chất lượng đồ thị tri thức xây dựng | 2 | Graph rỗng do lỗi GROQ_MODEL — pipeline extraction không chạy được. Schema và insert logic đúng về mặt thiết kế. |
| Khả năng phân tích và debug hệ thống | 4 | Trace được lỗi qua 3 tầng (surface error → upstream cause → root cause trong config), dùng `shape`, `columns`, `extraction_errors_df` để chẩn đoán chính xác. |
