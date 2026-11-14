# Báo cáo cá nhân — Day 7: Retrieval & Context Engineering

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - **Nguồn & Bản quyền:** Giáo trình đại học mở chuẩn học thuật [OpenStax - Introduction to Computer Science](https://openstax.org/details/books/introduction-computer-science), giấy phép Creative Commons Attribution 4.0 International (CC BY 4.0), được phép sao chép, trích dẫn và nghiên cứu học thuật hợp pháp.
  - **Số trang & Phạm vi:** 37 trang nội dung cốt lõi liên tục từ trang vật lý 19 đến 55 (Chương 1: *Introduction to Computer Science*, mục 1.1, 1.2, 1.3, Chapter 1 Review và Chương 2: *Computational Thinking*, mục 2.1).
  - **Số chunk:** Kế thừa trực tiếp chiến lược tối ưu nhất từ Day 6 là **Strategy B (Semantic Paragraph Chunking)** với ngưỡng `max_size = 800` ký tự, gồm tổng cộng **179 chunks** có cấu trúc mạch lạc (`coherence = 2.0/2.0`). Dữ liệu được lưu trữ tại `data/chunks.json`.

- **Vector store, embedding model, phiên bản và số chiều:**
  - **Embedding Model:** OpenAI API, model **`text-embedding-3-small`**, phiên bản cố định, kích thước vector **1536 chiều**.
  - **Tái sử dụng Cache Vector:** Tận dụng 100% vector embedding đã tính toán và kiểm chứng từ Day 6 thông qua cache hash SHA-256 (`data/.embedding_cache.json`, kích thước 14.6 MB), đảm bảo tính tái lập 100%, chi phí API bằng 0 USD và độ trễ truy xuất vector dưới 65 ms.
  - **Vector Store & Metric:** Cosine similarity $S_C(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$.

- **Agent model, cách chạy và phân chia dev/test:**
  - **Agent Model:** **OpenAI `gpt-4o-mini`** (nhiệt độ `temperature = 0.0`), tích hợp qua OpenAI Python SDK với cơ chế gọi công cụ ReAct / Tool-calling (`search` vs `answer`).
  - **Phân chia Benchmark (Tổng cộng 24 câu hỏi):**
    - **Tập Dev (17 câu):** 15 câu có đáp án trong tài liệu (`q01` $\to$ `q15`) + 2 câu thiếu bằng chứng (`q_unans_dev01`, `q_unans_dev02`).
    - **Tập Test (7 câu):** 5 câu có đáp án trong tài liệu (`q16` $\to$ `q20`) + 2 câu thiếu bằng chứng (`q_unans_test01`, `q_unans_test02`).

- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có bằng chứng (Answerable):** 20 câu hỏi kế thừa từ benchmark Day 6. Toàn bộ `evidence_quote` đều được script kiểm chứng tự động khớp 100% substring bên trong chunk tương ứng với đúng `source_id` và `page` ($20/20$ passed).
  - **Câu thiếu bằng chứng (Unanswerable / Abstention):** 4 câu hỏi được thiết kế bám sát chủ đề Khoa học Máy tính nhưng cố tình hỏi về các thông tin nằm ngoài phạm vi tài liệu (ví dụ: học phí VinUniversity, thông số VRAM GPU fine-tune transformer 70B, email cá nhân của Steve Jobs năm 1984, số lượng qubit sửa lỗi lượng tử). Toàn bộ nội dung giáo trình được xác nhận không chứa bằng chứng cho các câu này.
  - **Tiêu chí chấm:**
    - Retrieval: `Hit@1`, `Evidence Hit@3` (bằng chứng nằm trong top 1 hoặc top 3).
    - Context: `Context Evidence Rate` (bằng chứng nằm trong context cuối cùng gửi đến LLM sau khi áp dụng budget).
    - Câu trả lời Agent: Grounded Answer thang điểm $0$ (sai/không có evidence), $1$ (đúng một phần), $2$ (đúng và đủ); Citation Validity (citation có trong context và hỗ trợ câu trả lời); Abstention Accuracy (nhận diện và từ chối trả lời chính xác khi thiếu bằng chứng).

---

## Retrieval và thử nghiệm

### Bảng 1: So sánh các phương pháp và cấu hình Retrieval

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** | Lexical TF-IDF, top-k=3, budget=800 chars | Baseline từ vựng mặc định | dev | 11/15 (73.3%) | 12/15 (80.0%) | 7/15 (46.7%) | 0.36 ms / $0.00 |
| **Phương pháp 2** | Dense (text-embedding-3-small), top-k=3, budget=800 | Truy xuất ngữ nghĩa qua cosine similarity | dev | 8/15 (53.3%) | 13/15 (86.7%) | 3/15 (20.0%) | 62.09 ms / $0.00 |
| **Phương pháp 3** | Hybrid RRF (Dense + BM25, $k=60$), top-k=3, budget=800 | Hợp nhất thứ hạng từ khóa và vector | dev | 11/15 (73.3%) | **14/15 (93.3%)** | 6/15 (40.0%) | 76.77 ms / $0.00 |
| **Cải thiện 1** | Hybrid RRF, top-k=3, **budget=1500 chars** | Mở rộng budget tránh cắt gọt chunk chứa evidence | dev | 11/15 (73.3%) | **14/15 (93.3%)** | **11/15 (73.3%)** | 75.46 ms / $0.00 |
| **Cải thiện 2** | Hybrid RRF, **top-k=5**, budget=1500 chars, **Metadata Filter** | Tăng candidate pool và thu hẹp không gian tìm kiếm | dev | 11/15 (73.3%) | **14/15 (93.3%)** | **11/15 (73.3%)** | 66.97 ms / $0.00 |
| **Baseline** | Lexical TF-IDF, top-k=3, budget=800 chars | Baseline trên tập holdout test | test | 0/5 (0.0%) | 3/5 (60.0%) | 0/5 (0.0%) | 0.62 ms / $0.00 |
| **Cấu hình đã chọn** | **Hybrid RRF ($k=60$), top-k=3, budget=1500 chars** | Đánh giá độc lập trên tập test sau khi chốt trên dev | test | **2/5 (40.0%)** | **3/5 (60.0%)** | **2/5 (40.0%)** | 80.65 ms / $0.00 |

### Phân tích và Giải thích các Thí nghiệm Retrieval:

1. **Hiệu năng vượt trội của Hybrid RRF (Phương pháp 3 so với Baseline và Dense):**
   - Dense Retrieval đạt `Evidence Hit@3 = 86.7%`, bắt ngữ nghĩa tốt hơn TF-IDF (80.0%) nhưng `Hit@1` lại thấp hơn (53.3% vs 73.3%) do các thuật ngữ chuyên ngành bị phân tán.
   - Khi kết hợp Dense + BM25 qua thuật toán **Reciprocal Rank Fusion (RRF $k=60$)**, hiệu năng tăng vọt lên **`Hit@3 = 93.3%` (14/15 câu)**. Sự bù trừ giữa từ khóa chính xác của BM25 và ngữ nghĩa của Dense giúp đưa chunk chứa bằng chứng vào top-3 gần như tuyệt đối.

2. **Bài học thực nghiệm từ Context Engineering (Cải thiện 1):**
   - Khi quan sát kết quả Phương pháp 3 với budget mặc định 800 ký tự, `Evidence Hit@3` đạt tới 93.3% nhưng `Context Evidence Rate` chỉ đạt vỏn vẹn **40.0% (6/15 câu)**!
   - **Nguyên nhân cốt lõi:** Các chunk của Strategy B có độ dài trung bình ~500–600 ký tự. Khi đóng gói context (`pack_context`), chunk đầu tiên đã chiếm hơn 550 ký tự; nếu chunk chứa evidence nằm ở rank 2 hoặc rank 3, việc ghép thêm sẽ vượt ngưỡng 800 ký tự nên hàm packing buộc phải loại bỏ chunk đó.
   - **Giải pháp & Kết quả:** Tăng `budget` từ 800 lên **1500 ký tự**. Kết quả thực nghiệm chứng minh giả thuyết hoàn toàn chính xác: `Context Evidence Rate` **tăng vọt từ 40.0% lên 73.3% (11/15 câu)** mà không làm tăng độ trễ truy xuất.

3. **Phân tích 2 lỗi trên tập Dev:**
   - **Lỗi 1 (Câu `q08` - HCI experimental discipline):** Query hỏi về môn học nhấn mạnh thực nghiệm giao diện người-máy. Trong văn bản, từ khóa "human-computer interaction" chỉ xuất hiện thoáng qua ở trang 33 nhưng chunk đó bị điểm tương đồng thấp hơn các đoạn giới thiệu chung về kỹ thuật máy tính. Cả dense và BM25 đều xếp chunk này ngoài top-3 $\to$ Agent nhận thấy thiếu evidence và chủ động từ chối trả lời.
   - **Lỗi 2 (Câu `q12` - Định nghĩa Big Data trong Chapter 1 Review):** Đoạn văn nằm ở trang 43 thuộc bảng glossary tóm tắt. Do định nghĩa quá ngắn gọn, vector embedding bị áp đảo bởi các đoạn văn dài thảo luận về dữ liệu lớn ở mục 1.3, dẫn đến việc rank sai vị trí.

4. **Lý do chốt cấu hình:**
   - Cấu hình **Hybrid RRF (Dense + BM25, $k=60$) với Top-k=3 và Context Budget = 1500 ký tự** được chốt để chuyển sang đánh giá trên tập Test độc lập vì đạt sự cân bằng tối ưu: `Evidence Hit@3` cao nhất (93.3%), `Context Evidence Rate` cao nhất (73.3%), và thời gian xử lý rất nhanh (~75 ms).

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Tool:** Agent được trang bị công cụ `search(query, filters, top_k)`.
  - **Số lượt gọi tối đa (`max_searches`):** Tối đa 3 lượt truy xuất trước khi bắt buộc đưa ra kết luận hoặc từ chối trả lời.
  - **Điều kiện dừng:** Agent dừng ngay khi có đủ bằng chứng (`action: "answer"`) hoặc khi chạm giới hạn số lượt tìm kiếm mà không tìm thấy thông tin phù hợp (`search_limit` $\to$ `status: "abstain"`).
  - **Cách chọn context:** Thu thập các candidate qua từng lượt tìm kiếm, loại bỏ trùng lặp (`deduplication`) theo chunk ID, đóng gói lũy kế với trần budget 1500 ký tự.
  - **Xử lý trích dẫn & Abstention:** Áp dụng ràng buộc nghiêm ngặt: câu trả lời bắt buộc phải chứa trích dẫn định dạng `[chunk_id]` thuộc context hiện tại. Nếu không có trích dẫn hợp lệ hoặc ngữ cảnh rỗng, hệ thống tự động đưa về trạng thái từ chối (`abstain`).

### Bảng 2: Tổng hợp Đánh giá Agent (`agent_metrics.py`)

| Cấu hình / split | Tool use (n/N) | Số lượt gọi TB | Grounded answer 0/1/2 và trung bình /2 | Citation hợp lệ (n/N) | Câu có citation (n/N) | Abstention đúng (n/N) | Từ chối nhầm (n/N) |
|---|---|---|---|---|---|---|---|
| **Agent GPT-4o-mini (Dev)** | **17/17 (100%)** | 1.8 lượt | 0: 2, 1: 2, 2: 11 $\to$ **1.60 / 2.0** | **14/14 (100%)** | **13/13 (100%)** | **2/2 (100%)** | 2/15 (13.3%) |
| **Agent GPT-4o-mini (Test)** | **7/7 (100%)** | 2.1 lượt | 0: 3, 1: 0, 2: 2 $\to$ **0.80 / 2.0** | **2/2 (100%)** | **2/2 (100%)** | **2/2 (100%)** | 3/5 (60.0%) |

### Bảng 3: Minh chứng Chi tiết các Ca thực tế

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **`q01`** (Dev) | Có đáp án (Direct) | Gọi `search("Richard Bellman 1953 invention")` $\to$ Trả về `openstax-intro-cs_p22_para_000` (score=0.032) | "Richard Bellman invented dynamic programming in 1953, which is a mathematical optimization methodology and a technique for computer programming... [openstax-intro-cs_p22_para_000]" | **Điểm 2 (Đúng và đủ)**. Trích dẫn chính xác chunk chứa bằng chứng, nội dung khớp hoàn toàn với gold. |
| **`q17`** (Test) | Có đáp án (Condition) | Gọi `search("One Laptop per Child educational vision failure")` $\to$ Trả về `openstax-intro-cs_p29_para_000` | "The One Laptop per Child (OLPC) educational vision failed to truly materialize because children often used the computers for their own entertainment rather than the intended learning experiences... [openstax-intro-cs_p29_para_000]" | **Điểm 2 (Đúng và đủ)**. Phản ánh đúng nguyên nhân xã hội học và giáo dục từ sách. |
| **`q_unans_dev01`** (Dev) | Thiếu bằng chứng | Gọi `search` 3 lần thử tìm "VinUniversity tuition fee admission" $\to$ Không có chunk nào liên quan | "Không đủ bằng chứng để trả lời." (Citations: `[]`) | **Abstention đúng (Pass)**. Agent không bịa đặt học phí, nhận biết chính xác phạm vi giáo trình OpenStax. |
| **`q_unans_test01`** (Test) | Thiếu bằng chứng | Gọi `search` 3 lần về "Steve Jobs exact personal email 1984" $\to$ Không tìm thấy email trong context | "Không đủ bằng chứng để trả lời." (Citations: `[]`) | **Abstention đúng (Pass)**. Agent từ chối khẳng định vô căn cứ, không sinh email giả mạo. |
| **`q16`** (Test) | Lỗi (Direct) | Gọi `search("ENIAC physical technology memory silicon")` $\to$ Các chunk trả về nói chung về máy tính thập niên 1940 | "Không đủ bằng chứng để trả lời." (Citations: `[]`) | **Điểm 0 (False Abstention)**. Retrieval xếp đoạn vacuum tube ở vị trí quá sâu khiến context không nhận được evidence. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - **Traces:** [`results/agent_dev/traces.json`](../../results/agent_dev/traces.json), [`results/agent_test/traces.json`](../../results/agent_test/traces.json).
  - **Nhãn audit:** [`results/agent_dev/reviews.json`](../../results/agent_dev/reviews.json), [`results/agent_test/reviews.json`](../../results/agent_test/reviews.json).
  - **Tổng hợp metrics:** [`results/agent_dev/summary.json`](../../results/agent_dev/summary.json), [`results/agent_test/summary.json`](../../results/agent_test/summary.json).
  - **Lệnh chạy:**
    - `python src/run_all_retrieval_experiments.py`
    - `python src/agent_runner.py --split dev --output results/agent_dev`
    - `python src/agent_runner.py --split test --output results/agent_test`
  - **Thời gian & Chi phí:** Toàn bộ thí nghiệm retrieval tốn ~0.08s/câu; chạy Agent tốn trung bình 3.5s/câu (do gọi API OpenAI). Tổng chi phí LLM cho 24 lượt gọi agent $\approx \$0.003$ USD.

- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - **Đánh đổi giữa Context Budget và Latency/Cost:** Tăng budget từ 800 lên 1500 ký tự giúp `Context Evidence Rate` tăng ngoạn mục từ 40% lên 73.3%, đồng thời giữ nguyên chi phí vector cache. Tuy nhiên, context dài hơn khiến prompt tokens gửi cho LLM tăng nhẹ (~300 tokens/lượt gọi).
  - **Cơ chế Abstention an toàn:** Hệ thống đạt tỷ lệ nhận diện câu thiếu bằng chứng $100\%$ ($4/4$ câu unanswerable trên cả Dev và Test). Điều này đảm bảo tính đáng tin cậy cao cho các ứng dụng thực tế (không bao giờ sinh thông tin rác khi không chắc chắn).
  - **Giới hạn hệ thống:** Với các câu hỏi yêu cầu bảng thuật ngữ tóm tắt ở cuối chương (như `q20`), độ tương đồng ngữ nghĩa của đoạn ngắn thường bị lấn át bởi các đoạn văn giải thích dài. Hướng khắc phục trong tương lai là bổ sung Hierarchical Indexing (chỉ mục theo cấu trúc phân cấp tài liệu).
