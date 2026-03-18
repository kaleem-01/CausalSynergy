from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Tuple
from dataclasses import asdict, is_dataclass
import pandas as pd


def _json_dump(obj: Any) -> Any:
    """Best-effort JSON serialization for df.attrs."""
    try:
        import networkx as nx  # type: ignore
        if isinstance(obj, nx.Graph):
            return {
                "nodes": [str(n) for n in obj.nodes()],
                "edges": [(str(u), str(v)) for u, v in obj.edges()],
                "directed": obj.is_directed(),
            }
    except Exception:
        pass

    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def save_csv_with_attrs(df: pd.DataFrame, csv_path: str, *, attrs: Dict[str, Any]) -> None:
    """
    Saves:
      - <csv_path>
      - <csv_path>.meta.json   (df.attrs as JSON)
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    df.attrs.update(attrs)
    df.to_csv(csv_path, index=False)

    sidecar = csv_path + ".meta.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(df.attrs, f, indent=2, default=_json_dump)



def load_csv_with_attrs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    sidecar = csv_path + ".meta.json"
    if os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as f:
            df.attrs = json.load(f)
    else:
        df.attrs = {}

    return df


def _extract_id(fname: str) -> int:
    m = re.search(r"(\d+)", fname)
    if not m:
        raise ValueError(f"Could not extract numeric id from filename: {fname}")
    return int(m.group(1))


def read_data_files(
    data_dir: str,
    csv_pattern: str = "Data_Graph_",
    metadata_pattern: str = "Metadata_",
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    data_files = [f for f in files if f.startswith(csv_pattern)]
    meta_files = [f for f in files if f.startswith(metadata_pattern)]

    data_by_id_fname: Dict[int, str] = {_extract_id(f): f for f in data_files}
    meta_by_id_fname: Dict[int, str] = {_extract_id(f): f for f in meta_files}

    common_ids = data_by_id_fname.keys() & meta_by_id_fname.keys()
    if not common_ids:
        raise RuntimeError("No id appears in both data and metadata files.")

    # Load dataframes (data gets attrs; metadata is a plain df)
    df_by_id: Dict[int, pd.DataFrame] = {}
    metadata_by_id: Dict[int, pd.DataFrame] = {}

    for i in sorted(common_ids):
        # print(f"Loading data and metadata for id={i}...")
        df_path = os.path.join(data_dir, data_by_id_fname[i])
        meta_path = os.path.join(data_dir, meta_by_id_fname[i])

        df_by_id[i] = load_csv_with_attrs(df_path)
        metadata_by_id[i] = pd.read_csv(meta_path)

    return df_by_id, metadata_by_id




