import shutil
import tarfile
import tempfile
from pathlib import Path

import requests

COLLECTION_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
DATA_FILE_PATH = "datasets/MS_MACRO_raw.tsv"


def download_data():
    """Download the MS MARCO passage collection and save it as datasets/MS_MACRO_raw.tsv."""
    base = Path(__file__).resolve().parent.parent
    out_path = base / "datasets" / "MS_MACRO_raw.tsv"

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        resp = requests.get(COLLECTION_URL, stream=True)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        with tarfile.open(tmp_path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile() and member.name.endswith(".tsv"):
                    with tf.extractfile(member) as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    break
    finally:
        Path(tmp_path).unlink(missing_ok=True)
