import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_itransformer_hash_unchanged():
    assert sha256(ROOT / "third_party" / "TaTS" / "models" / "iTransformer.py") == sha256(ROOT / "tats_cen" / "models" / "iTransformer.py")
