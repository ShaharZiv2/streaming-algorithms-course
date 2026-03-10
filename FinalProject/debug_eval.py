"""Debug script to trace ID alignment in evaluation pipeline."""
from evaluation.benchmark import load_queries_and_qrels_from_qrels_jsonl
from evaluate_retrievers import build_evaluation_corpus
from logic.constants import QRELS_JSONL
from retrievers.classic_retriever import ClassicRetriever

queries, qrels = load_queries_and_qrels_from_qrels_jsonl(QRELS_JSONL)

print("Sample qrels:")
for qid, docs in list(qrels.items())[:5]:
    print(f"  qid={qid!r}  relevant={sorted(list(docs))[:3]}")

print("\nSample queries:")
for q in queries[:5]:
    print(f"  key={q['key']!r}  in_qrels={q['key'] in qrels}  text={q['data'][:50]!r}")

print("\nBuilding eval corpus...")
corpus_docs, eval_qrels, eval_queries = build_evaluation_corpus(
    qrels=qrels, queries=queries, corpus_size=500, max_queries=30
)

corpus_ids = {d['key'] for d in corpus_docs}
print(f"\nCorpus IDs sample: {sorted(list(corpus_ids))[:10]}")
print(f"\nEval qrels sample:")
for qid, docs in list(eval_qrels.items())[:3]:
    print(f"  qid={qid!r}  relevant={sorted(docs)}")

# For each eval query, check if relevant docs are in corpus
r = ClassicRetriever(top_k=10, num_initial_documents=0)
r.build_corpus_from_docs(corpus_docs)

for q in eval_queries[:5]:
    qid = q['key']
    relevant = eval_qrels[qid]
    in_corpus = relevant & corpus_ids
    result = r.retrieve(q['data'], top_k=10)
    retrieved = [doc_id for doc_id, _ in result]
    print(f"\nQuery: {q['data'][:60]!r}")
    print(f"  relevant={sorted(relevant)}  in_corpus={sorted(in_corpus)}")
    print(f"  retrieved: {retrieved[:5]}")
    print(f"  OVERLAP: {set(retrieved) & relevant}")
