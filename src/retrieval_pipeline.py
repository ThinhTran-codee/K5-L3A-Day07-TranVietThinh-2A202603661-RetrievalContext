"""Retrieval Pipeline supporting 3+ distinct retrieval strategies and improvements:
1. Lexical: TF-IDF & BM25
2. Dense: Cosine similarity on OpenAI text-embedding-3-small vectors
3. Hybrid: Reciprocal Rank Fusion (RRF, k=60) combining Dense + BM25
Also supports configurable context budget (800 vs 1500 chars), top_k, and metadata filtering.
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path


def tokenize(text: str) -> list[str]:
    norm = unicodedata.normalize('NFKC', text).lower()
    return re.findall(r'\b[a-zA-Z0-9_\-]+\b', norm)


def hit(chunk: dict, gold: list[dict]) -> bool:
    return any(
        chunk['source_id'] == g['source_id']
        and chunk['page'] == g['page']
        and g['evidence'] in chunk['text']
        for g in gold
    )


def pack_context(results: list[dict], budget: int = 800) -> tuple[str, list[str]]:
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        candidate = '\n\n'.join(blocks + [block])
        if len(candidate) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids


class BM25Index:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.N = len(chunks)
        self.corpus_tokens = [tokenize(c['text']) for c in chunks]
        self.doc_lens = [len(toks) for toks in self.corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 1.0

        self.df = Counter()
        for doc in self.corpus_tokens:
            for word in set(doc):
                self.df[word] += 1

    def search(self, query: str, top_k: int = 3, filter_dict: dict | None = None) -> list[dict]:
        q_tokens = tokenize(query)
        scores = []
        for i, c in enumerate(self.chunks):
            if filter_dict:
                match = all(c['metadata'].get(k) == v for k, v in filter_dict.items())
                if not match:
                    continue

            doc_tokens = self.corpus_tokens[i]
            doc_len = self.doc_lens[i]
            tf = Counter(doc_tokens)
            score = 0.0
            for q in q_tokens:
                if q not in self.df:
                    continue
                n_q = self.df[q]
                idf = math.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1.0)
                f = tf[q]
                score += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl)))
            if score > 0:
                scores.append((score, c))
        scores.sort(key=lambda x: (-x[0], x[1]['id']))
        return [{**c, 'score': s, 'strategy': 'bm25'} for s, c in scores[:top_k]]


class DenseIndex:
    def __init__(self, chunks: list[dict], cache_path: Path):
        self.chunks = chunks
        self.cache = json.loads(cache_path.read_text(encoding='utf-8'))
        
        # Precompute chunk embeddings
        self.chunk_embs = []
        for c in chunks:
            h = hashlib.sha256(f"text-embedding-3-small::{c['text']}".encode('utf-8')).hexdigest()
            if h not in self.cache:
                raise KeyError(f"Embedding cache missing for chunk {c['id']}")
            self.chunk_embs.append(self.cache[h])

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0

    def get_query_embedding(self, query: str) -> list[float]:
        h = hashlib.sha256(f"text-embedding-3-small::{query}".encode('utf-8')).hexdigest()
        if h in self.cache:
            return self.cache[h]
        # Fallback to OpenAI API if not cached
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.embeddings.create(input=query, model="text-embedding-3-small")
        emb = resp.data[0].embedding
        self.cache[h] = emb
        return emb

    def search(self, query: str, top_k: int = 3, filter_dict: dict | None = None) -> list[dict]:
        q_emb = self.get_query_embedding(query)
        scores = []
        for i, c in enumerate(self.chunks):
            if filter_dict:
                match = all(c['metadata'].get(k) == v for k, v in filter_dict.items())
                if not match:
                    continue
            sim = self.cosine_similarity(q_emb, self.chunk_embs[i])
            scores.append((sim, c))
        scores.sort(key=lambda x: (-x[0], x[1]['id']))
        return [{**c, 'score': s, 'strategy': 'dense'} for s, c in scores[:top_k]]


class HybridRRFIndex:
    def __init__(self, dense_idx: DenseIndex, bm25_idx: BM25Index, k: int = 60):
        self.dense_idx = dense_idx
        self.bm25_idx = bm25_idx
        self.k = k

    def search(self, query: str, top_k: int = 3, filter_dict: dict | None = None) -> list[dict]:
        # Retrieve top 20 candidates from both
        dense_res = self.dense_idx.search(query, top_k=20, filter_dict=filter_dict)
        bm25_res = self.bm25_idx.search(query, top_k=20, filter_dict=filter_dict)

        chunk_map = {}
        rrf_scores = Counter()

        for rank, c in enumerate(dense_res):
            cid = c['id']
            chunk_map[cid] = c
            rrf_scores[cid] += 1.0 / (self.k + rank + 1)

        for rank, c in enumerate(bm25_res):
            cid = c['id']
            chunk_map[cid] = c
            rrf_scores[cid] += 1.0 / (self.k + rank + 1)

        sorted_candidates = rrf_scores.most_common(top_k)
        results = []
        for cid, score in sorted_candidates:
            orig = chunk_map[cid]
            results.append({**orig, 'score': score, 'strategy': 'hybrid_rrf'})
        return results


def run_evaluation(index, questions: list[dict], strategy_name: str, budget: int = 800,
                   top_k: int = 3, use_filter: bool = False) -> tuple[dict, list[dict]]:
    traces = []
    times = []
    for q in questions:
        start = time.perf_counter()
        f = q.get('filter') if use_filter else None
        results = index.search(q['question'], top_k=top_k, filter_dict=f)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

        context, ids = pack_context(results, budget=budget)
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]

        traces.append({
            'query_id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'strategy': strategy_name,
            'top_k': top_k,
            'budget_chars': budget,
            'use_filter': use_filter,
            'gold': q['gold'],
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': bool(marked and marked[0]['hit']),
            'hit3': any(c['hit'] for c in marked[:3]),
            'context_hit': any(c['hit'] and c['id'] in ids for c in marked)
        })

    n = len(traces)
    h1 = sum(t['hit1'] for t in traces)
    h3 = sum(t['hit3'] for t in traces)
    chit = sum(t['context_hit'] for t in traces)

    summary = {
        'strategy': strategy_name,
        'split': questions[0]['split'] if questions else 'unknown',
        'n': n,
        'top_k': top_k,
        'budget_chars': budget,
        'use_filter': use_filter,
        'hit1': f"{h1}/{n} ({h1/n*100:.1f}%)" if n else "0",
        'hit3': f"{h3}/{n} ({h3/n*100:.1f}%)" if n else "0",
        'context_evidence_rate': f"{chit}/{n} ({chit/n*100:.1f}%)" if n else "0",
        'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces), 2) if n else 0,
        'median_ms': round(statistics.median(times), 2) if times else 0
    }
    return summary, traces


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategy on Day 7 benchmark")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--strategy", choices=("bm25", "dense", "hybrid_rrf"), required=True)
    parser.add_argument("--budget", type=int, default=800, help="Context budget in characters")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--use_filter", action="store_true", help="Apply question filters if present")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    selected_q = [q for q in questions if q.get('split') == args.split]

    cache_path = Path("data/.embedding_cache.json")

    if args.strategy == "bm25":
        idx = BM25Index(chunks)
    elif args.strategy == "dense":
        idx = DenseIndex(chunks, cache_path)
    elif args.strategy == "hybrid_rrf":
        dense = DenseIndex(chunks, cache_path)
        bm25 = BM25Index(chunks)
        idx = HybridRRFIndex(dense, bm25)

    summary, traces = run_evaluation(
        idx, selected_q, args.strategy,
        budget=args.budget, top_k=args.top_k, use_filter=args.use_filter
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
