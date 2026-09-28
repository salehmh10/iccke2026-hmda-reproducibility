"""Generate the two self-exempt artifact manifests after deliberate edits."""
import csv
import hashlib
import json
from package_checks import ROOT, MANIFESTS, public_files


def generate(root=ROOT):
    provenance=json.loads((root/'data/manifests/packaging_provenance.json').read_text())
    entries=[]
    for p in public_files(root):
        rel=p.relative_to(root).as_posix()
        if rel in MANIFESTS:
            continue
        meta=provenance.get(rel, {})
        role=('aggregate_results' if rel.startswith('results/') else 'static_source' if rel.startswith(('src/classification/','src/regression/','archive/')) else 'sanitized_notebook' if p.suffix=='.ipynb' else 'configuration' if rel.startswith('configs/') else 'validator_or_test' if rel.startswith(('scripts/','tests/','src/common/','.github/')) else 'artifact_metadata' if rel.startswith('data/') else 'documentation')
        privacy=('aggregate_non_row_level' if role=='aggregate_results' else 'source_code_no_data' if role in {'static_source','sanitized_notebook'} else 'synthetic_validation' if role=='validator_or_test' else 'artifact_metadata' if role=='artifact_metadata' else 'documentation')
        row={'relative_path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':str(p.stat().st_size),'artifact_role':role,'experiment_generation':meta.get('experiment_generation','package'),'evaluation_scope':meta.get('evaluation_scope','not_applicable'),'scientific_status':meta.get('scientific_status','supporting_analysis'),'privacy_classification':privacy,'source_evidence':meta.get('source_evidence','repository packaging specification'),'copied_or_transformed':meta.get('copied_or_transformed','created'),'transformation_description':meta.get('transformation_description','Authored package documentation or synthetic validation')}
        # Mixed-generation/result tables carry exact provenance at row level.
        if role=='aggregate_results' and p.suffix=='.csv':
            with p.open(encoding='utf-8',newline='') as stream:
                rows=list(csv.DictReader(stream))
            sources=sorted({r.get('source_artifact','') for r in rows}-{''})
            row['source_evidence']='; '.join(sources)
        entries.append(row)
    dest=root/'data/manifests'
    with (dest/'public_artifact_manifest.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
    (dest/'public_artifact_manifest.json').write_text(json.dumps({'format_version':1,'self_exclusions':sorted(MANIFESTS),'reason':'Generated manifest metadata cannot contain its own content hash; every other public file is hashed. Git commit authenticates the manifests.','artifacts':entries},indent=2)+'\n',encoding='utf-8')
    print('MANIFEST GENERATED:',len(entries),'hashed artifacts; 2 self-exempt generated manifest files')


if __name__=='__main__':
    generate()
