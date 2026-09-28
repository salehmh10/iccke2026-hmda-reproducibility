import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def test_final_generation_lock_matches_all_test_bearing_artifacts() -> None:
    lock_path = ROOT / "artifacts" / "models" / "FINAL_GENERATION.lock.json"
    assert lock_path.is_file()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["generation_id"] == "final_v1_versioned_equal_60k"
    assert len(lock["sha256"]) >= 13
    for relative, expected in lock["sha256"].items():
        path = ROOT / relative
        assert path.is_file()
        assert _sha256(path) == expected

