# Action-taken mapping

| HMDA action | Binary classification | Regression V2 |
|---|---|---|
| 1: loan originated | favorable (`loan_approved=1`) | retained |
| 2: approved but not accepted | favorable (`loan_approved=1`) | retained |
| 3: application denied | denial (`loan_approved=0`) | excluded |
| 7: preapproval request denied | denial (`loan_approved=0`) | excluded |
| 8: preapproval request approved but not accepted | favorable (`loan_approved=1`) | retained |
| 4/5/6 and other codes | excluded from binary population | excluded |

The historical notebook explicitly maps 1/2/8 to one and 3/7 to zero. The later classification adapter uses `target_denied = 1 - loan_approved`, so positive class 1 means denial. Favorable action does not imply loan origination, customer acceptance, or disbursement. In particular, codes 2 and 8 are not accepted/originated loans.

Source evidence (relative to the read-only source collection):
- `main/DATA_CLEANING_FOR_14M.ipynb`
- `regresionpart2/regression_v2/src/data_cleaning.py`
- `HMDA_pipeline_review/project/new new project/src/data/schema.py`
