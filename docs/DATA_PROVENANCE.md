# Data provenance

The historical cleaning notebook is now available. It names the nationwide labelled HMDA 2017 file and documents the binary action mapping and historical sampler. This corrects earlier claims that no label-construction implementation was available. A saved source report for Regression V2 records 14,285,496 raw rows and 78 original columns. The historical classification notebook records the same stated source; a complete immutable chain from that historical notebook run to the retained classification extract is not proven here.

Classification uses the supplied transformed 500,000-row, 29-column extract. Its recorded file SHA-256 is `b9c8373a9234f0a6083d0b92e87574f37c0c2be858812b2e1cbde66ae5af63fe`. Later exact full-row deduplication removes 260 excess copies before splitting, leaving 499,740 rows. The original class counts are 401,226 favorable and 98,774 denial records. Source code recovery is stronger provenance evidence than the old flattened audit, but does not establish national representativeness or authenticate every historical intermediate file.

Regression V2 independently documents the labelled nationwide 2017 source: recorded SHA-256 `dd35f6a877c5882bbe7260ce65ca842b18ca6aa16514cba256feeeb316d4b7c3`, size 11,237,068,086 bytes. This is a saved identity, not a new hash of raw data. It retains favorable-action records, removes exact duplicates, contradictory-target groups, and overlap with the legacy regression sample before selecting 575,000 records. Raw files and individual membership remain excluded.

See [filters](DATA_FILTERS.md), [sampling](SAMPLING_DESIGN.md), and [target definitions](POPULATION_AND_TARGETS.md).

Source evidence (relative to the read-only source collection):
- `main/DATA_CLEANING_FOR_14M.ipynb`
- `HMDA_pipeline_review/project/new new project/reports/DATASET_REPORT.md`
- `regresionpart2/regression_v2/config.json`
- `regresionpart2/regression_v2/outputs/reports/prompt1a_source_report.json`
