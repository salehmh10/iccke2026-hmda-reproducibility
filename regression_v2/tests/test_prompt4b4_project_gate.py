from pathlib import Path

import pytest

from src.prompt4b4_experiments import (
    DATA, EXPECTED_AUDIT_DIGEST, EXPECTED_SELECTION_DIGEST,
    EXPECTED_TRAIN_DIGEST, EXPECTED_VALIDATION_DIGEST,
    guarded_read, load_context, ordered_digest, root_path,
)


def test_project_raw_iid_and_path_guards():
    root=root_path()
    assert guarded_read(root,DATA) == (root/DATA).resolve()
    for prohibited in (Path("data/raw.csv"),Path("outputs/data/iid_holdout.parquet"),Path("../outside.parquet")):
        with pytest.raises(PermissionError): guarded_read(root,prohibited)


def test_exact_frozen_membership_and_feature_order():
    root=root_path(); features,frame,train,validation,roles,split=load_context(root)
    assert len(features)==35 and len(frame)==500_000 and len(train)==400_000 and len(validation)==100_000
    assert ordered_digest(train.row_hash)==EXPECTED_TRAIN_DIGEST
    assert ordered_digest(validation.row_hash)==EXPECTED_VALIDATION_DIGEST
    assert split["selection_row_hash_digest"]==EXPECTED_SELECTION_DIGEST
    assert split["audit_row_hash_digest"]==EXPECTED_AUDIT_DIGEST
    assert (roles=="selection").sum()==70_000 and (roles=="audit").sum()==30_000
