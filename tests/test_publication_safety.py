import json
import pytest
from package_checks import scan_file,notebook_errors,safety

@pytest.mark.parametrize('filename,content,expected',[
    ('model.joblib','synthetic','forbidden extension'),
    ('notes.md','D'+':'+chr(92)+'private','personal_drive_path'),
    ('notes.md','C'+':'+chr(92)+'Users'+chr(92)+'person','personal_drive_path'),
    ('notes.md','/ho'+'me/person','personal_home_path'),
    ('notes.md','gh'+'p_'+'A'*30,'github_credential'),
    ('notes.md','s'+'k-'+'A'*30,'api_credential'),
    ('test_ids.csv','id\n1\n','forbidden data/model filename'),
    ('synthetic.csv','row_'+'id,y_true\n1,2\n','applicant-level column'),
])
def test_bad_artifact_rejected(tmp_path,filename,content,expected):
    p=tmp_path/filename;p.write_text(content)
    assert expected in scan_file(p,tmp_path)

def test_applicant_notebook_outputs_rejected():
    assert notebook_errors({'cells':[{'outputs':[{'data':{'text/plain':'synthetic applicant preview'}}]}]})

def test_clean_notebook_accepted():
    assert notebook_errors({'cells':[{'outputs':[],'execution_count':None}]})==[]

def test_no_forbidden_packaged_content():
    assert safety()==[]
