import json
import pickle
from pathlib import Path
from typing import Dict, Generator, Any

import numpy as np
import joblib


def load_jsonl(file_path: str) -> Generator[Dict]:
    with open(file_path, 'r', ) as jsonl_file:
        for line in jsonl_file:
            yield json.loads(line)


def load_jsonl_data(file_path: str, num_rows: int | None = None) -> Generator[Dict]:
    row = 0
    if num_rows is None:
        for jsonl in load_jsonl(file_path):
            yield jsonl['data']

    else:
        for jsonl in load_jsonl(file_path):
            if num_rows is not None and row > num_rows:
                break
            yield jsonl['data']



def load_processed_file(file_path: str) -> Any:
    extenstion = Path(file_path).suffix
    try:
        match extenstion:
            case '.npy' | '.npz':
                return np.load(file_path, allow_pickle=True)
            case '.joblib':
                return joblib.load(file_path)
            case '.pkl':
                with open(file_path, 'rb') as file:
                    return pickle.load(file)
    except FileNotFoundError:
        return None
