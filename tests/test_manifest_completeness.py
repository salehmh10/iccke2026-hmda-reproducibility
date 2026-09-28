import json
from package_checks import manifest
from generate_public_manifest import generate

def test_repository_manifest_complete():
    assert manifest()==[]

def test_missing_entry_and_modified_bytes_detected(tmp_path):
    meta=tmp_path/'data/manifests';meta.mkdir(parents=True)
    (meta/'packaging_provenance.json').write_text('{}')
    artifact=tmp_path/'README.md';artifact.write_text('synthetic')
    generate(tmp_path)
    assert manifest(tmp_path)==[]
    artifact.write_text('changed synthetic')
    assert any('hash/size mismatch' in e for e in manifest(tmp_path))
    (tmp_path/'new.md').write_text('synthetic')
    assert any('coverage mismatch' in e for e in manifest(tmp_path))
