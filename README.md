**Học viên:** Trần Việt Thịnh  
**MSSV:** 2A202603661  
**Lớp:** L3A  
**Corpus:** OpenStax Introduction to Computer Science
**Kế thừa:** Day 6 Data Ingestion 
**Repo nộp bài:** (`https://github.com/ThinhTran-codee/K5-L3A-Day07-TranVietThinh-2A202603661-Retrieval-Context-Engineering`)

---

## Hướng dẫn tái lập kết quả (Reproduction)

```powershell
# 1. Chạy kiểm tra Unit Tests
python -m unittest discover -s tests -v

# 2. Chuẩn bị dữ liệu và cache từ Day 6 (179 chunks, 24 benchmark cases)
python src/prepare_data.py

# 3. Chạy toàn bộ các thí nghiệm Retrieval (TF-IDF, Dense, Hybrid RRF, Budget 1500)
python src/run_all_retrieval_experiments.py

# 4. Chạy Agentic RAG Controller với model thật trên Dev và Test
python src/agent_runner.py --split dev --output results/agent_dev
python src/agent_runner.py --split test --output results/agent_test
```

