import os
import sys
import pickle
import traceback
from typing import Dict, List, Optional, Tuple

import polars as pl
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))
try:
    from dataset.impl.Exceptions import KnownException
    from config import ProjectConfig, ProjectConst
except ImportError:
    from dataset.impl.Exceptions import KnownException
    from config import ProjectConfig, ProjectConst


# -------------------------
# Config
# -------------------------
max_workers = 24
code_tokenizer_name = "microsoft/unixcoder-base"
tokenizer_path = f"{ProjectConfig.pretrain_path}/{code_tokenizer_name}"
code_tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    cache_dir=f"{ProjectConfig.pretrain_path}/{code_tokenizer_name}",
)


def chunked_rows(rows: List[dict], size: int) -> List[List[dict]]:
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def _encode_one(code: Optional[str]) -> Optional[List[int]]:
    if code is None:
        return None
    if not isinstance(code, str):
        code = str(code)
    code = code.strip()
    if not code:
        return None
    try:
        # batch_encode_plus expects list/tuple of strings; we encode single
        enc = code_tokenizer.batch_encode_plus((code,)).encodings
        if not enc:
            return None
        return enc[0].ids
    except KnownException:
        return None
    except Exception:
        return None


def process_batch_for_one_col(batch_rows: List[dict], col_name: str) -> Dict[str, object]:

    out = {}
    for r in batch_rows:
        _id = r.get("_id")
        out[_id] = _encode_one(r.get(col_name))
    return out


def _list_code_columns(df: pl.DataFrame) -> List[str]:

    return [c for c in df.columns if c not in ("_id", "docstring")]


def run_exported_parquet(lang: str, parquet_path: str, split_name: str, *, batch_size: int = 256):
    print(f"Building code tokens for {parquet_path}")
    df = pl.read_parquet(parquet_path)

    if "_id" not in df.columns:
        raise ValueError("Missing required column: _id")
    if "code" not in df.columns:
        raise ValueError("Missing required column: code")

    code_columns = _list_code_columns(df)
    print(f"Columns to tokenize ({len(code_columns)}): {code_columns[:8]}{'...' if len(code_columns) > 8 else ''}")

    base_dir = os.path.join(ProjectConfig.path_compiled_dataset, lang)

    for col_name in code_columns:
        if col_name != ProjectConst.llm_type_deepseek_v3:
            continue
        out_dir = os.path.join(base_dir, col_name)
        os.makedirs(out_dir, exist_ok=True)

        out_pkl = os.path.join(out_dir, f"{split_name}_tokens_{code_tokenizer_name.replace('/','-')}.pkl")
        if os.path.exists(out_pkl):
            os.remove(out_pkl)

        sub = df.select(["_id", col_name])
        rows = sub.to_dicts()
        total = len(rows)
        batches = chunked_rows(rows, batch_size)

        result_map: Dict[str, object] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_batch_for_one_col, b, col_name) for b in batches]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name}:{col_name}"):
                try:
                    result_map.update(fut.result())
                except Exception as e:
                    print(f"Future error: {e}")
                    print(traceback.format_exc())

        with open(out_pkl, "wb") as f:
            pickle.dump(result_map, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved: {out_pkl} (items={len(result_map)})")


def main():
    lang = "java"
    base = f"{ProjectConfig.path_origin_dataset}/{lang}"

    run_exported_parquet(lang, f"{base}/train.parquet", "train")
    run_exported_parquet(lang, f"{base}/valid.parquet", "val")
    run_exported_parquet(lang, f"{base}/test.parquet", "test")


if __name__ == "__main__":
    main()
