import json

BASE = '/Users/tamara/Documents/school/streaming-algorithms-course/FinalProject/datasets'

print("=== First 5 collection.jsonl entries ===")
with open(f'{BASE}/jsonl/collection.jsonl') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        print(f"  key={obj['key']!r}  data[:60]={obj['data'][:60]!r}")
        if i >= 4:
            break

print("\n=== First 5 qrels.jsonl entries ===")
with open(f'{BASE}/jsonl/qrels.jsonl') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        parts = obj['data'].strip().split('\t')
        print(f"  passage_id(key)={obj['key']!r}  query_id={parts[0]!r}  query={parts[1][:40]!r}")
        if i >= 4:
            break

# Check overlap
print("\n=== ID overlap check ===")
collection_keys = set()
with open(f'{BASE}/jsonl/collection.jsonl') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        collection_keys.add(obj['key'])
        if i >= 9999:
            break

qrel_passage_ids = set()
with open(f'{BASE}/jsonl/qrels.jsonl') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        qrel_passage_ids.add(obj['key'])
        if i >= 9999:
            break

overlap = collection_keys & qrel_passage_ids
print(f"  Sample collection keys: {sorted(list(collection_keys))[:10]}")
print(f"  Sample qrel passage_ids: {sorted(list(qrel_passage_ids))[:10]}")
print(f"  Overlap (first 10k vs first 10k): {len(overlap)}")

