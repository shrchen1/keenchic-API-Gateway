# Use Client-Chosen Subfolders for Report Groups

Taimide report uploads use an optional client-chosen `subfolder` instead of a batch-specific field such as `batch_number`. The domain concept is a Report Group, currently used to collect Excel reports for the same batch across different instruments, while the generic wire field preserves flexibility for future customer grouping requirements. The server treats the value as one validated directory name under `reports/`; omitting it preserves the original upload contract.
