from retrievers.minhashLSH_retriever import MinHashLSHRetriever
from retrievers.sketch_retriever import SketchRetriever

docs = [
    {"key": str(i), "data": f"test document about topic number {i} with some extra words"}
    for i in range(20)
]
docs += [
    {"key": "100", "data": "blood sugar levels and foods that lower glucose"},
    {"key": "101", "data": "vegetarian diet health benefits nutrition study research"},
]

print("=== MinHashLSHRetriever ===")
mh = MinHashLSHRetriever(num_initial_documents=0, top_k=3)
mh.build_corpus_from_docs(docs)
print("ids        :", type(mh.ids).__name__, f"len={len(mh.ids)}")
print("signatures :", type(mh.signatures).__name__, f"len={len(mh.signatures)}")
print("doc_texts  :", type(mh.doc_texts).__name__, f"len={len(mh.doc_texts)}")
print("lsh_index  :", type(mh.lsh_index).__name__)
r = mh.retrieve("blood sugar foods lower glucose", top_k=3)
print("retrieve   :", r)
mh.update({"key": "200", "data": "update test document"})
print("update OK, ids len:", len(mh.ids))

print()
print("=== SketchRetriever ===")
sk = SketchRetriever(num_initial_documents=0)
sk.build_corpus_from_docs(docs)
print("ids        :", type(sk.ids).__name__, f"len={len(sk.ids)}")
print("signatures :", type(sk.signatures).__name__)
print("doc_texts  :", type(sk.doc_texts).__name__, f"len={len(sk.doc_texts)}")
r2 = sk.retrieve("blood sugar foods lower glucose", top_k=3)
print("retrieve   :", r2)

print()
print("=== API alignment check ===")
for attr in ["ids", "signatures", "doc_texts"]:
    mh_t = type(getattr(mh, attr)).__name__
    sk_t = type(getattr(sk, attr)).__name__
    match = "✓" if mh_t == sk_t else "✗"
    print(f"  {attr}: MinHash={mh_t}  Sketch={sk_t}  {match}")

