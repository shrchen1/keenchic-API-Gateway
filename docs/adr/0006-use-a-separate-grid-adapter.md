# Use a Separate Grid Adapter

`ocr/meter-table-grid` will use a dedicated `MeterTableGridAdapter` that invokes `procd_table_L.proc`, while `ocr/meter-table` keeps its existing `MeterTableAdapter` and public behavior. The adapters may share private model-loading and CUDA lifecycle code, but they will keep independent accepted fields, result mapping, tests, registry entries, and build descriptors so Grid Inspection cannot change the Cell Inspection contract.

`MeterTableGridAdapter` will be a shallow subclass of `MeterTableAdapter`. It overrides accepted fields, Grid request validation, imports of `procd_table_L.proc`, and payload construction while reusing backend selection and fallback, model loading and unloading, CUDA context management, and diagnostic-image encoding. This limits changes to the established Cell Inspection lifecycle without duplicating the full adapter.
