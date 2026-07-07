"""Arrow-based data normalization for training ingest.

Converts CSV/Parquet/JSON uploads into training_examples rows via PyArrow.
Handles column detection, type inference, and label column mapping.
"""

import io
from typing import Optional

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.json as pa_json


SUPPORTED_FORMATS = {"csv", "parquet", "json"}


def detect_format(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ("csv", "tsv", "txt"):
        return "csv"
    if ext in ("parquet", "pq"):
        return "parquet"
    if ext in ("json", "jsonl", "ndjson"):
        return "json"
    raise ValueError(f"Unsupported file format: {ext or 'unknown'}")


def normalize(data: bytes, fmt: str) -> pa.Table:
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{fmt}'. Supported: {', '.join(SUPPORTED_FORMATS)}")

    buf = io.BytesIO(data)

    if fmt == "csv":
        table = pa_csv.read_csv(buf)
    elif fmt == "parquet":
        table = pa.parquet.read_table(buf)
    elif fmt == "json":
        table = pa_json.read_json(buf)

    return table


def table_to_examples(
    table: pa.Table,
    label_column: Optional[str] = None,
) -> list[dict]:
    rows = []
    columns = table.column_names

    for batch in table.to_batches(max_chunksize=1000):
        for i in range(batch.num_rows):
            input_data = {}
            label = None
            metadata = {"row_index": len(rows)}

            for col in columns:
                value = batch.column(col)[i].as_py()
                if col == label_column:
                    label = str(value) if value is not None else None
                else:
                    input_data[col] = value

            row = {
                "input_data": input_data,
                "original_label": label,
                "example_metadata": metadata,
            }
            rows.append(row)

    return rows
