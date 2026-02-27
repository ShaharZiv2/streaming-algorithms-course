import json
from typing import Dict, Generator


def load_jsonl(file_path: str) -> Generator[Dict]:
    with open(file_path, 'r', ) as jsonl_file:
        for line in jsonl_file:
            yield json.loads(line)
