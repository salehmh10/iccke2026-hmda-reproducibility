# Static source transformations

Reference implementation files preserve recovered logic for inspection. Personal absolute paths were replaced by the placeholder SOURCE_ROOT and text uses UTF-8/LF with trailing whitespace removed. Scientific modules were parsed as Python syntax, never imported or executed. Internal references to excluded data/model files remain as documentation of the historical workflow; they are not supplied artifacts or runnable reproduction promises.

The only executable validation helper outside scripts/tests is src/common/contracts.py, which performs simple scalar/tiny-synthetic orientation, AP, and routing checks. It cannot open data or model files. All scientific source trees and archived scientific tests are outside pytest's testpaths.
