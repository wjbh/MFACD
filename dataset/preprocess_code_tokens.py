import os
import sys
import pickle
import traceback
from typing import Dict, List, Optional

import polars as pl
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from transformers import AutoTokenizer


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
try:
    from dataset.impl.Exceptions import KnownException
    from dataset.impl.JavaCodeExtractor import JavaCodeExtractor
    from config import ProjectConfig, ProjectConst
except ImportError:
    from dataset.impl.Exceptions import KnownException
    from dataset.impl.JavaCodeExtractor import JavaCodeExtractor
    from config import ProjectConfig, ProjectConst


# -------------------------
# Config
# -------------------------
max_workers = 24
batch_size_default = 256

code_tokenizer_name = "microsoft/unixcoder-base"
tokenizer_path = f"{ProjectConfig.pretrain_path}/{code_tokenizer_name}"
tokenizer_cache_dir = f"{ProjectConfig.pretrain_path}/{code_tokenizer_name}"


# -------------------------
# Helpers
# -------------------------
def chunked_rows(rows: List[dict], size: int) -> List[List[dict]]:
    return [rows[i:i + size] for i in range(0, len(rows), size)]


def _sanitize_type(t: Optional[str]) -> str:

    if t is None:
        return "unk"
    t = str(t).strip()
    if not t:
        return "unk"
    safe = []
    for ch in t:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _init_worker_tokenizer():

    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=tokenizer_cache_dir,
    )


def _encode_one(code: Optional[str], extractor_cls) -> Optional[List[int]]:
    if code is None:
        return None
    if not isinstance(code, str):
        code = str(code)
    code = code.strip()
    if not code:
        return None
    try:
        txt = extractor_cls(code).get_codes()
        if not txt:
            return None


        tok = globals().get("_WORKER_TOKENIZER", None)
        if tok is None:

            tok = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=tokenizer_cache_dir)

        enc = tok.batch_encode_plus((txt,)).encodings
        if not enc:
            return None
        return enc[0].ids
    except KnownException:
        return None
    except Exception:
        return None


def process_batch_for_type(batch_rows: List[dict], extractor_cls) -> Dict[str, object]:
    """
    batch_rows: [{"id": ..., "code": ...}, ...]
    return: {id: token_ids_or_None}
    """
    out: Dict[str, object] = {}
    for r in batch_rows:
        _id = r.get("id")
        out[_id] = _encode_one(r.get("code"), extractor_cls)
    return out


# -------------------------
# Main
# -------------------------
def run_exported_parquet(lang: str, parquet_path: str, split_name: str, *, batch_size: int = batch_size_default):
    extractor_cls = JavaCodeExtractor if lang == "java" else None
    if extractor_cls is None:
        raise ValueError(f"Unsupported language: {lang}")

    print(f"Building code tokens for {parquet_path}")
    df = pl.read_parquet(parquet_path)


    required = {"id", "type", "code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    out_dir = ProjectConfig.path_compiled_dataset
    os.makedirs(out_dir, exist_ok=True)


    types = df.select(pl.col("type").cast(pl.Utf8)).unique().to_series().to_list()

    types_sorted = sorted(types, key=lambda x: str(x) if x is not None else "")

    print(f"Found types ({len(types_sorted)}): {types_sorted[:20]}{'...' if len(types_sorted) > 20 else ''}")

    tok_slug = code_tokenizer_name.replace("/", "-")


    for tval in types_sorted:
        t_safe = _sanitize_type(tval)

        out_pkl = os.path.join(
            out_dir,
            f"{lang}_{split_name}_code_tokens_{tok_slug}_{t_safe}.pkl",
        )
        if os.path.exists(out_pkl):
            os.remove(out_pkl)

        sub = df.filter(pl.col("type") == tval).select(["id", "code"])
        rows = sub.to_dicts()
        total = len(rows)

        if total == 0:
            print(f"[skip] type={tval} -> empty")
            continue

        batches = chunked_rows(rows, batch_size)
        result_map: Dict[str, object] = {}

        with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker_tokenizer,
        ) as ex:
            futures = [ex.submit(process_batch_for_type, b, extractor_cls) for b in batches]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name}:{t_safe}"):
                try:
                    result_map.update(fut.result())
                except Exception as e:
                    print(f"Future error: {e}")
                    print(traceback.format_exc())

        with open(out_pkl, "wb") as f:
            pickle.dump(result_map, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved: {out_pkl} (type={tval}, items={len(result_map)})")


def main():
    lang = "java"
    base = f"{ProjectConfig.path_origin_dataset}"

    run_exported_parquet(lang, f"{base}/step1_{lang}_train.parquet", "train")
    run_exported_parquet(lang, f"{base}/step1_{lang}_val.parquet", "val")
    run_exported_parquet(lang, f"{base}/step1_{lang}_test.parquet", "test")


if __name__ == "__main__":
    main()
