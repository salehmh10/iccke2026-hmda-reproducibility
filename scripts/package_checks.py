"""Read-only validators for the packaged evidence; never import scientific source."""
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {
    'authoritative_manuscript_result', 'supporting_analysis', 'development_only',
    'historical', 'post_test_descriptive', 'superseded', 'invalid',
    'failed_technical', 'not_executed', 'unresolved',
}
SCOPES = {
    'screening', 'development', 'train_only_validation', 'out_of_fold', 'validation',
    'calibration', 'threshold_selection', 'final_test', 'iid_evaluation', 'post_test',
    'historical_test', 'aggregate_explainability', 'not_applicable',
}
MANIFESTS = {
    'data/manifests/public_artifact_manifest.csv',
    'data/manifests/public_artifact_manifest.json',
}
GENERATED_DIRS = {'.git', '__pycache__', '.pytest_cache'}
FORBIDDEN_EXTENSIONS = {
    '.pkl', '.pickle', '.joblib', '.cbm', '.pt', '.pth', '.ckpt', '.npy', '.npz',
    '.parquet', '.arrow', '.feather', '.h5', '.hdf5', '.onnx', '.bin', '.rds',
    '.zip', '.rar', '.pdf', '.exe', '.dll', '.so',
}
FILE_PATTERNS = [
    r'(?:^|/)(?:raw|processed|predictions|checkpoints|models|membership)/',
    r'(?:train|test|validation|calibration|threshold|iid)_(?:ids|indices|membership)\b',
    r'(?:^|/)(?:row_hashes|row_ids|sample_row_hashes)\.',
    r'hmda.*(?:500k|all-records|ready)\.csv$',
    r'(?:predictions?|errors?)_(?:rows|rowwise|row_level|test|iid)\.(?:csv|json)$',
    r'(?:sensitive_rows|local_cases|applicant_records)\.(?:csv|json)$',
]
# Static scientific source directories named models contain implementation code,
# not model objects. They are allowed only for Python source below reference/src.
CONTENT_PATTERNS = [
    ('personal_drive_path', re.compile(r'\b[A-Za-z]:[\\/]')),
    ('personal_home_path', re.compile('/ho' + 'me/|/Us' + 'ers/')),
    ('github_credential', re.compile(r'gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}')),
    ('api_credential', re.compile(r'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}')),
    ('private_key', re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
    ('credential_assignment', re.compile(r'''(?i)(?:api_key|access_token|client_secret)\s*[:=]\s*["'][A-Za-z0-9+/=_-]{20,}["']''')),
]
ROW_COLUMNS = {'row_id', 'row_hash', 'record_hash', 'applicant_id', 'y_true', 'y_pred', 'respondent_id'}


def public_files(root=ROOT):
    return sorted(p for p in root.rglob('*') if p.is_file() and not set(p.relative_to(root).parts) & GENERATED_DIRS)


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def scan_file(path, root):
    relative = path.relative_to(root).as_posix()
    errors = []
    if path.is_symlink():
        errors.append('symlink')
    if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        errors.append('forbidden extension')
    reference_code = relative.startswith('src/classification/reference/src/models/') and path.suffix == '.py'
    for pattern in FILE_PATTERNS:
        if re.search(pattern, relative, re.I) and not reference_code:
            errors.append('forbidden data/model filename')
    if path.stat().st_size > 2_000_000:
        errors.append('file exceeds 2 MB limit; no binary allowlist')
    try:
        text = path.read_text(encoding='utf-8-sig')
    except UnicodeError:
        return errors + ['non-text artifact']
    for name, pattern in CONTENT_PATTERNS:
        if pattern.search(text):
            errors.append(name)
    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            parsed = csv.DictReader(stream)
            if set(parsed.fieldnames or []) & ROW_COLUMNS:
                errors.append('applicant-level column')
            if sum(1 for _ in parsed) > 10_000:
                errors.append('oversized table')
    if path.suffix == '.ipynb':
        errors += notebook_errors(json.loads(text))
    if path.suffix == '.json' and path.name != 'public_artifact_manifest.json':
        obj = json.loads(text)
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ROW_COLUMNS and isinstance(item, (list, dict)):
                        errors.append('row-level JSON array')
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(obj)
    return errors


def notebook_errors(obj):
    errors = []
    for index, cell in enumerate(obj.get('cells', [])):
        if cell.get('outputs'):
            errors.append(f'notebook outputs at cell {index}')
        if cell.get('execution_count') is not None:
            errors.append(f'notebook execution count at cell {index}')
        if cell.get('attachments'):
            errors.append(f'notebook attachment at cell {index}')
    return errors


def safety(root=ROOT):
    return [f'{p.relative_to(root)}: {e}' for p in public_files(root) for e in scan_file(p, root)]


def scopes(root=ROOT):
    errors = []
    required = {'source_artifact', 'source_row_key', 'source_generation', 'source_evaluation_scope', 'scientific_status', 'evaluation_scope'}
    for p in sorted((root/'results').rglob('*.csv')):
        data = rows(p)
        seen = set()
        for number, row in enumerate(data, 2):
            label = f'{p.relative_to(root)}:{number}'
            if not required.issubset(row) or any(not row.get(k) for k in required):
                errors.append(label + ': missing result provenance')
            if row.get('scientific_status') not in STATUSES or row.get('evaluation_scope') not in SCOPES:
                errors.append(label + ': invalid result role')
            if row.get('source_evaluation_scope') not in SCOPES:
                errors.append(label + ': invalid source role')
            key = tuple(row.get(k, '') for k in ['source_generation', 'source_artifact', 'source_row_key'])
            if key in seen:
                errors.append(label + ': duplicate source row key')
            seen.add(key)
            if 'historical' in row.get('source_generation', '') and row.get('scientific_status') == 'authoritative_manuscript_result':
                errors.append(label + ': historical generation promoted')
            if p.parts[-2] == 'regression' and 'stage4l' in str(row).lower():
                errors.append(label + ': historical regression mixed with V2')
            if row.get('positive_class') and row['positive_class'] != 'denial':
                errors.append(label + ': wrong classification orientation')
    for p in sorted((root/'results').rglob('*.json')):
        value = json.loads(p.read_text(encoding='utf-8'))
        if value.get('scientific_status') not in STATUSES or value.get('evaluation_scope') not in SCOPES:
            errors.append(str(p.relative_to(root)) + ': invalid JSON role')
        if not value.get('source_artifact') or not value.get('source_row_key'):
            errors.append(str(p.relative_to(root)) + ': missing JSON provenance')
    return errors


def traceability(root=ROOT):
    errors = []
    index = {r['source_artifact']: r for r in rows(root/'data/manifests/source_artifact_index.csv')}
    for p in (root/'results').rglob('*.csv'):
        for row in rows(p):
            source = index.get(row['source_artifact'])
            if source is None:
                errors.append(f'{p.name}: source absent from index')
            elif row.get('source_sha256') and row['source_sha256'] != source['sha256']:
                errors.append(f'{p.name}: source hash mismatch')
    for p in (root/'results').rglob('*.json'):
        value = json.loads(p.read_text(encoding='utf-8'))
        if value['source_artifact'] not in index:
            errors.append(f'{p.name}: JSON source absent from index')
    checks = {
        'classification/validation_results_84.csv': 84,
        'classification/final_test_results_6.csv': 6,
        'classification/feature_v1_dictionary.csv': 33,
        'classification/feature_v2_interactions_33.csv': 33,
        'classification/interaction_candidates_98.csv': 98,
        'regression/feature_dictionary_35.csv': 35,
        'regression/development_advanced_candidates_120.csv': 120,
        'regression/development_advanced_scopes_360.csv': 360,
    }
    for rel, count in checks.items():
        if len(rows(root/'results'/rel)) != count:
            errors.append(rel + ': wrong count')
    validation = rows(root/'results/classification/validation_results_84.csv')
    keys = {(r['representation'], r['model'], r['dataset_variant']) for r in validation}
    if len(keys) != 84 or len({r['model'] for r in validation}) != 14:
        errors.append('classification grid incomplete')
    long = rows(root/'results/regression/development_advanced_scopes_360.csv')
    lookup = {(r['candidate_id'], r['source_artifact'], r['scope'].replace('_descriptive','')): r for r in long}
    for r in rows(root/'results/regression/development_advanced_candidates_120.csv'):
        for prefix, scope in [('selection','selection'), ('audit','audit'), ('validation','complete_validation')]:
            original = lookup[(r['candidate_id'], r['source_artifact'], scope)]
            if r[prefix+'_mae'] != original['mae'] or r[prefix+'_top_decile_mae'] != original['top_decile_mae']:
                errors.append('advanced pivot changed saved value')
    condition = json.loads((root/'results/regression/six_condition_check.json').read_text())
    if condition['conditions_passed'] != 5 or condition['conditions_total'] != 6:
        errors.append('six-condition conclusion changed')
    if {r['condition'] for r in condition['conditions'] if r['status']=='FAIL'} != {'C2'}:
        errors.append('C2 failure absent')
    overall={r['model_id']:r for r in rows(root/'results/regression/iid_overall_metrics.csv')}
    for r in rows(root/'results/regression/iid_body_tail_metrics.csv'):
        field={'body_bottom90':'bottom_90_mae','top_decile':'top_decile_mae','top5':'top_five_percent_mae'}[r['band']]
        if r['mae']!=overall[r['model_id']][field]:
            errors.append('body/tail export changed saved band metric')
    for r in validation + rows(root/'results/classification/final_test_results_6.csv'):
        if r['denial_positive_ap']!=r['pr_auc'] or r['positive_class']!='denial':
            errors.append('AP alias/orientation mismatch')
    return errors


def link_errors(path, root):
    text = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
    errors = []
    for target in re.findall(r'(?<!!)\[[^\]\n]*\]\(([^)]+)\)', text):
        target = target.strip().split(' "',1)[0].strip('<>')
        if urlparse(target).scheme or target.startswith('#'):
            continue
        target = unquote(target.split('#',1)[0])
        resolved = (path.parent/target).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            errors.append(f'{path.relative_to(root)}: broken internal link {target}')
    return errors


def links(root=ROOT):
    return [e for p in public_files(root) if p.suffix=='.md' for e in link_errors(p, root)]


def manifest(root=ROOT):
    errors=[]
    csv_path=root/'data/manifests/public_artifact_manifest.csv'
    json_path=root/'data/manifests/public_artifact_manifest.json'
    if not csv_path.exists() or not json_path.exists():
        return ['public manifest missing']
    entries=rows(csv_path)
    j=json.loads(json_path.read_text(encoding='utf-8'))
    if json.loads(json.dumps(entries)) != j['artifacts']:
        errors.append('CSV/JSON manifest mismatch')
    actual={p.relative_to(root).as_posix():p for p in public_files(root) if p.relative_to(root).as_posix() not in MANIFESTS}
    expected={r['relative_path']:r for r in entries}
    if len(expected)!=len(entries):
        errors.append('duplicate manifest path')
    if set(actual)!=set(expected):
        errors.append('manifest coverage mismatch: missing='+str(sorted(set(actual)-set(expected)))+' stale='+str(sorted(set(expected)-set(actual))))
    required={'relative_path','sha256','size_bytes','artifact_role','experiment_generation','evaluation_scope','scientific_status','privacy_classification','source_evidence','copied_or_transformed','transformation_description'}
    for rel in set(actual)&set(expected):
        row=expected[rel];p=actual[rel]
        if not required.issubset(row) or any(not row[k] for k in required):
            errors.append(rel+': incomplete manifest metadata')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256'] or str(p.stat().st_size)!=row['size_bytes']:
            errors.append(rel+': artifact hash/size mismatch')
        if row['scientific_status'] not in STATUSES or row['evaluation_scope'] not in SCOPES:
            errors.append(rel+': invalid manifest scope/status')
        if row['privacy_classification'] not in {'aggregate_non_row_level','source_code_no_data','documentation','synthetic_validation','artifact_metadata'}:
            errors.append(rel+': unapproved privacy class')
    return errors


def run(name, root=ROOT):
    functions={'safety':safety,'scopes':scopes,'traceability':traceability,'links':links,'manifest':manifest}
    errors=functions[name](root)
    print(name.upper()+(': FAIL' if errors else ': PASS'))
    for error in errors:
        print(error)
    return not errors
