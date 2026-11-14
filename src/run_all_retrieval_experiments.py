"""Run all retrieval experiments required by Day 7 Rubric and produce comparative table.
Strategies:
1. Baseline: TF-IDF (dev, budget 800, top_k 3)
2. Method 2: Dense Retrieval (dev, budget 800, top_k 3)
3. Method 3: BM25 / Hybrid RRF (dev, budget 800, top_k 3)
4. Improvement 1: Hybrid RRF with Context Budget 1500 chars (dev)
5. Improvement 2: Hybrid RRF with Budget 1500 + Filter + Top-K 5 (dev)
6. Holdout Test: Baseline vs Chosen Strategy
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    # Dev comparisons
    {
        "name": "Baseline (TF-IDF)",
        "cmd": [sys.executable, "starter/baseline.py", "--chunks", "data/chunks.json",
                "--questions", "data/questions.json", "--split", "dev", "--output", "results/dev_tfidf"],
        "summary_file": "results/dev_tfidf/summary.json",
        "split": "dev",
        "details": "TF-IDF lexical, top-k=3, budget=800 chars"
    },
    {
        "name": "Method 2 (Dense)",
        "cmd": [sys.executable, "src/retrieval_pipeline.py", "--split", "dev",
                "--strategy", "dense", "--budget", "800", "--top_k", "3", "--output", "results/dev_dense"],
        "summary_file": "results/dev_dense/summary.json",
        "split": "dev",
        "details": "Dense cosine similarity (text-embedding-3-small), top-k=3, budget=800"
    },
    {
        "name": "Method 3 (Hybrid RRF)",
        "cmd": [sys.executable, "src/retrieval_pipeline.py", "--split", "dev",
                "--strategy", "hybrid_rrf", "--budget", "800", "--top_k", "3", "--output", "results/dev_hybrid_rrf"],
        "summary_file": "results/dev_hybrid_rrf/summary.json",
        "split": "dev",
        "details": "Hybrid RRF (Dense + BM25, k=60), top-k=3, budget=800"
    },
    {
        "name": "Improvement 1 (Hybrid + Budget 1500)",
        "cmd": [sys.executable, "src/retrieval_pipeline.py", "--split", "dev",
                "--strategy", "hybrid_rrf", "--budget", "1500", "--top_k", "3", "--output", "results/dev_hybrid_b1500"],
        "summary_file": "results/dev_hybrid_b1500/summary.json",
        "split": "dev",
        "details": "Hybrid RRF, tăng budget lên 1500 chars tránh gãy evidence"
    },
    {
        "name": "Improvement 2 (Hybrid + Budget 1500 + Top-5 + Filter)",
        "cmd": [sys.executable, "src/retrieval_pipeline.py", "--split", "dev",
                "--strategy", "hybrid_rrf", "--budget", "1500", "--top_k", "5", "--use_filter", "--output", "results/dev_hybrid_best"],
        "summary_file": "results/dev_hybrid_best/summary.json",
        "split": "dev",
        "details": "Hybrid RRF, top-k=5, budget 1500 + metadata filtering"
    },
    # Test evaluations
    {
        "name": "Baseline Test (TF-IDF)",
        "cmd": [sys.executable, "starter/baseline.py", "--chunks", "data/chunks.json",
                "--questions", "data/questions.json", "--split", "test", "--output", "results/test_tfidf"],
        "summary_file": "results/test_tfidf/summary.json",
        "split": "test",
        "details": "Baseline TF-IDF trên tập test, top-k=3, budget=800"
    },
    {
        "name": "Chosen Strategy Test (Hybrid RRF)",
        "cmd": [sys.executable, "src/retrieval_pipeline.py", "--split", "test",
                "--strategy", "hybrid_rrf", "--budget", "1500", "--top_k", "3", "--output", "results/test_hybrid_chosen"],
        "summary_file": "results/test_hybrid_chosen/summary.json",
        "split": "test",
        "details": "Cấu hình chốt: Hybrid RRF, budget=1500 chars trên tập test"
    }
]

def main():
    rows = []
    print("=" * 80)
    print("RUNNING ALL RETRIEVAL EXPERIMENTS (DAY 7)")
    print("=" * 80)

    for exp in EXPERIMENTS:
        print(f"\n---> Running: {exp['name']} ({exp['split']})")
        subprocess.run(exp["cmd"], check=True)
        summary_path = ROOT / exp["summary_file"]
        data = json.loads(summary_path.read_text(encoding="utf-8"))

        n = data.get("n", 0)
        h1 = data.get("hit1")
        h3 = data.get("hit3")
        c_rate = data.get("context_evidence_rate", data.get("context_hit"))
        time_ms = data.get("median_ms", 0)

        # Standardize strings
        if isinstance(h1, int):
            h1_str = f"{h1}/{n} ({h1/n*100:.1f}%)"
        else:
            h1_str = str(h1)

        if isinstance(h3, int):
            h3_str = f"{h3}/{n} ({h3/n*100:.1f}%)"
        else:
            h3_str = str(h3)

        if isinstance(c_rate, int):
            crate_str = f"{c_rate}/{n} ({c_rate/n*100:.1f}%)"
        else:
            crate_str = str(c_rate)

        rows.append({
            "name": exp["name"],
            "details": exp["details"],
            "split": exp["split"],
            "hit1": h1_str,
            "hit3": h3_str,
            "context_evidence_rate": crate_str,
            "time_ms": f"{time_ms:.2f} ms"
        })

    # Save summary table
    out_table = ROOT / "results" / "retrieval_comparison_table.json"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 105)
    print(f"{'Cấu hình':<35} | {'Split':<5} | {'Hit@1':<16} | {'Evidence Hit@3':<18} | {'Context Ev. Rate':<18} | {'Latency'}")
    print("-" * 105)
    for r in rows:
        print(f"{r['name']:<35} | {r['split']:<5} | {r['hit1']:<16} | {r['hit3']:<18} | {r['context_evidence_rate']:<18} | {r['time_ms']}")
    print("=" * 105)


if __name__ == "__main__":
    main()
