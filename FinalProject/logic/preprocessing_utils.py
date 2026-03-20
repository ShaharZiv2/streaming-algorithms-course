import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import requests

from logic.constants import (
    COLLECTION_URL, COLLECTION_TSV, QUERIES_TSV, QUERIES_URL, QRELS_TSV, QRELS_URL,
    COLLECTION_JSONL, QUERIES_JSONL, QRELS_JSONL,
    TSV_FILES_DIR, JSONL_FILES_DIR, PROCESSED_FILES_DIR,
    PROCESSED_MIN_HASH_FILES_DIR, PROCESSED_PROB_MIN_HASH_FILES_DIR,
    CORPUS_DIR, MIN_HASH_CORPUS_DIR, PROB_MIN_HASH_CORPUS_DIR, TF_IDF_DIR,
)

FIVE_MB = 5 * 1024 * 1024

# Pretend to be a normal browser (some CDNs / hosts reject bare Python clients).
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# All dataset directory paths (folders only) from constants
_OUTPUT_DIRS = (
    TSV_FILES_DIR,
    JSONL_FILES_DIR,
    PROCESSED_FILES_DIR,
    PROCESSED_MIN_HASH_FILES_DIR,
    PROCESSED_PROB_MIN_HASH_FILES_DIR,
    CORPUS_DIR,
    MIN_HASH_CORPUS_DIR,
    PROB_MIN_HASH_CORPUS_DIR,
    TF_IDF_DIR,
)


def create_output_dirs():
    """Create all dataset directories from constants if they do not exist. Creates only folders, not files."""
    for dir_path in _OUTPUT_DIRS:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def download_data(url: str, destination: str, skip_exists):
    """Download the MS MARCO passage collection and save it as datasets/MS_MACRO_raw.tsv."""
    if skip_exists and Path(destination).exists():
        print(f'Skip downloading {url}, the destination already exists.')
        return
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        resp = requests.get(url, stream=True, headers=_BROWSER_HEADERS)
        resp.raise_for_status()
        downloaded = 0
        start_time = time.perf_counter()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=FIVE_MB):
                f.write(chunk)
                downloaded += len(chunk)
                print(f"Downloaded {downloaded} bytes out of 1035009698 in {time.perf_counter() - start_time} seconds.")
        with tarfile.open(tmp_path, "r:gz") as tf:
            member = tf.getmembers()[0]
            with tf.extractfile(member) as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def convert_tsv_to_jsonl(input_tsv_path, output_jsonl_path, skip_exists, max_line=500_000):
    """Convert a two-column TSV (key, data) to JSONL in datasets/json_data."""
    if skip_exists and Path(output_jsonl_path).exists():
        print(f'Skip converting {input_tsv_path}, the destination already exists.')
        return
    with open(input_tsv_path, "r", encoding="utf-8", errors="replace") as tsv_file, open(
        output_jsonl_path, "w", encoding="utf-8"
    ) as jsonl_file:
        for i, line in enumerate(tsv_file):
            print(f'\rProcessing line {i}', end='', flush=True)
            parts = line.split("\t", 1)
            key = parts[0]
            data = parts[1]
            jsonl_file.write(json.dumps({"key": key, "data": data}) + "\n")
            if i >= max_line:
                return


def prepare_data(download=False, skip_exists=True):
    """
    Prepares the data. If download=True, downloads the data. This process take roughly 1.5 hours.
    We recommend downloading using the browser. The links are:
    The collection: https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
    The queries: https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
    The qrels: https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv
    """
    create_output_dirs()
    if download:
        download_data(COLLECTION_URL, COLLECTION_TSV, skip_exists)
        download_data(QUERIES_URL, QUERIES_TSV, skip_exists)
        download_data(QRELS_URL, QRELS_TSV, skip_exists)

    convert_tsv_to_jsonl(COLLECTION_TSV, COLLECTION_JSONL, skip_exists)
    convert_tsv_to_jsonl(QUERIES_TSV, QUERIES_JSONL, skip_exists)
    convert_tsv_to_jsonl(QRELS_TSV, QRELS_JSONL, skip_exists)
