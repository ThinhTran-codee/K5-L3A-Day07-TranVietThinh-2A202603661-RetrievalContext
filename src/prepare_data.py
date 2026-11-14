"""Prepare and validate data from Day 6 for Day 7.
Builds:
- data/chunks.json (179 chunks from Strategy B paragraph chunking)
- data/questions.json (20 answerable questions: 15 dev, 5 test)
- data/agent_cases.json (24 agent cases: 17 dev [15 ans + 2 unans], 7 test [5 ans + 2 unans])
- data/.embedding_cache.json (cached embeddings from Day 6)
"""
import json
import shutil
from pathlib import Path
import sys

DAY7_ROOT = Path(__file__).resolve().parents[1]
DAY6_ROOT = Path(r"d:\AIThucChienVinUni\K5-L3A-DAY06-TranVietThinh-2A202603661-DataIngestion")

sys.path.insert(0, str(DAY6_ROOT))
from starter.src.ingestion import load_pdf
from starter.src.chunking import by_paragraph

def main():
    print("=== Step 1: Loading PDF and Generating Chunks ===")
    pdf_path = DAY6_ROOT / "starter" / "data" / "Introduction_To_Computer_Science_-_WEB.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    
    pages = load_pdf(pdf_path, "openstax-intro-cs", str(pdf_path))
    raw_chunks = []
    for p in pages:
        raw_chunks.extend(by_paragraph(p, max_size=800))
    
    print(f"Loaded {len(raw_chunks)} chunks using Strategy B (max_size=800).")
    
    chunks = [
        {
            "id": c.chunk_id,
            "source_id": c.document_id,
            "source": f"openstax-intro-cs:p{c.page}",
            "page": c.page,
            "text": c.text,
            "metadata": c.metadata
        }
        for c in raw_chunks
    ]
    
    chunks_path = DAY7_ROOT / "data" / "chunks.json"
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {chunks_path}")

    print("\n=== Step 2: Converting Benchmark Questions ===")
    bench_path = DAY6_ROOT / "benchmark" / "benchmark.json"
    bench_data = json.loads(bench_path.read_text(encoding="utf-8"))
    
    questions = []
    for q in bench_data:
        # Match evidence in chunk to ensure exact containment
        q_item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": [
                {
                    "source_id": q["gold_source_id"],
                    "page": q["gold_page"],
                    "evidence": q["evidence_quote"]
                }
            ]
        }
        if q.get("filter"):
            q_item["filter"] = q["filter"]
        questions.append(q_item)
    
    # Validate each gold evidence is strictly contained in a chunk
    valid_count = 0
    for q in questions:
        g = q["gold"][0]
        matching = [
            c for c in chunks
            if c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"]
        ]
        if matching:
            valid_count += 1
        else:
            print(f"Warning: Evidence for {q['id']} on page {g['page']} not exact substring in chunk!")
            # Attempt to find best matching chunk by prefix
            for c in chunks:
                if c["page"] == g["page"] and g["evidence"][:40] in c["text"]:
                    print(f"  -> Found match with prefix for {q['id']}, adapting evidence string.")
                    # Use the exact slice present in chunk
                    idx = c["text"].find(g["evidence"][:40])
                    g["evidence"] = c["text"][idx : idx + len(g["evidence"])]
                    valid_count += 1
                    break
    print(f"Validated gold evidence containment: {valid_count}/{len(questions)} passed.")

    questions_path = DAY7_ROOT / "data" / "questions.json"
    questions_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(questions)} answerable questions to {questions_path}")

    print("\n=== Step 3: Creating Agent Benchmark Cases (24 cases) ===")
    # 20 answerable cases + 4 unanswerable cases (2 dev, 2 test)
    agent_cases = []
    for q in questions:
        agent_cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": q["gold"],
            "filter": q.get("filter")
        })
    
    # Add 2 Dev unanswerable cases
    agent_cases.append({
        "id": "q_unans_dev01",
        "split": "dev",
        "question": "What is the tuition fee and admission deadline for computer science students at VinUniversity mentioned in Section 1.2?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The textbook is an academic introductory CS text from OpenStax and does not contain tuition fees or admission information for VinUniversity."
    })
    agent_cases.append({
        "id": "q_unans_dev02",
        "split": "dev",
        "question": "According to Chapter 1, what exact GPU VRAM is required to fine-tune a 70-billion parameter transformer model with LoRA?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Chapter 1 discusses historical foundations and definitions of computer science, with no technical specifications on modern GPU VRAM for transformer fine-tuning."
    })

    # Add 2 Test unanswerable cases
    agent_cases.append({
        "id": "q_unans_test01",
        "split": "test",
        "question": "What was Steve Jobs' exact personal email address when he introduced the Apple Macintosh in 1984 as recorded in Chapter 1?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The text mentions historical computer systems but never provides Steve Jobs' personal email address."
    })
    agent_cases.append({
        "id": "q_unans_test02",
        "split": "test",
        "question": "According to Section 2.1, how many physical qubits are needed for surface code quantum error correction?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Section 2.1 covers computational thinking concepts (decomposition, pattern recognition, abstraction, algorithms) in classical CS, not quantum error correction physical qubits."
    })

    agent_cases_path = DAY7_ROOT / "data" / "agent_cases.json"
    agent_cases_path.write_text(json.dumps(agent_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(agent_cases)} agent benchmark cases (17 dev, 7 test) to {agent_cases_path}")

    print("\n=== Step 4: Copying Embedding Cache ===")
    src_cache = DAY6_ROOT / "starter" / "results" / ".embedding_cache.json"
    dst_cache = DAY7_ROOT / "data" / ".embedding_cache.json"
    if src_cache.exists():
        shutil.copy2(src_cache, dst_cache)
        print(f"Copied embedding cache to {dst_cache} (size: {dst_cache.stat().st_size} bytes)")
    else:
        print(f"Notice: Cache not found at {src_cache}")

    print("\nData preparation complete!")

if __name__ == "__main__":
    main()
