import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple, Iterable

import polars as pl
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


sys.setrecursionlimit(20000)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
try:
    from dataset.impl.Exceptions import KnownException
    from dataset.impl.JavaMetricsExtractor import JavaMetricsExtractor
    from config import ProjectConfig, ProjectConst
except ImportError:
    from dataset.impl.Exceptions import KnownException
    from dataset.impl.JavaMetricsExtractor import JavaMetricsExtractor
    from config import ProjectConfig, ProjectConst



_WORKER_EXTRACTOR_CLS = None


def _init_worker(extractor_cls):

    global _WORKER_EXTRACTOR_CLS
    _WORKER_EXTRACTOR_CLS = extractor_cls


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


def min_max_normalize(
    feature_vectors: List[np.ndarray],
    *,
    eps: float = 1e-8,
    global_min: np.ndarray,
    global_max: np.ndarray,
) -> np.ndarray:
    if not feature_vectors:
        return np.zeros((0, global_min.shape[0]), dtype=np.float32)
    all_features = np.vstack(feature_vectors).astype(np.float32, copy=False)
    denom = (global_max - global_min + eps).astype(np.float32, copy=False)
    normalized = (all_features - global_min) / denom
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32, copy=False)


def calculate_mean(features: List[np.ndarray]) -> np.ndarray:
    if not features:
        return np.array([], dtype=np.float32)
    all_features = np.vstack(features).astype(np.float32, copy=False)
    m = np.mean(all_features, axis=0)
    m[~np.isfinite(m)] = 0.0
    return m.astype(np.float32, copy=False)


def _extract_metrics(code: Optional[str]) -> Optional[np.ndarray]:

    global _WORKER_EXTRACTOR_CLS
    if _WORKER_EXTRACTOR_CLS is None:
        raise RuntimeError("Worker extractor class not initialized. Did you pass initializer?")

    if code is None:
        return None
    if not isinstance(code, str):
        code = str(code)
    code = code.strip()
    if not code:
        return None
    try:
        feat = _WORKER_EXTRACTOR_CLS(code).generate_feature_list()  # 每条样本独立创建 extractor 实例
        if feat is None:
            return None
        arr = np.asarray(feat, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return arr
    except KnownException:
        return None

    except BaseException: 

        print("\n[Warning] One sample failed due to recursion/parse error.", flush=True)
        return None


def _iter_batches(ids: List, types: List, codes: List, batch_size: int) -> Iterable[Tuple[List, List, List]]:
    n = len(ids)
    for i in range(0, n, batch_size):
        yield ids[i:i + batch_size], types[i:i + batch_size], codes[i:i + batch_size]


def _worker_process_batch(args: Tuple[List, List, List]) -> Dict[Tuple[str, str], Optional[np.ndarray]]:

    ids_batch, types_batch, codes_batch = args
    out: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    for _id, _t, _code in zip(ids_batch, types_batch, codes_batch):
        key = (str(_id), str(_t) if _t is not None else None)
        out[key] = _extract_metrics(_code)
    return out


def extract_all_features_once(
    df: pl.DataFrame,
    extractor_cls,
    batch_size: int = 1024,
    max_workers: Optional[int] = None,
) -> Dict[Tuple[str, str], Optional[np.ndarray]]:
    ids = df["id"].to_list()
    types = df["type"].to_list()
    codes = df["code"].to_list()
    total_rows = len(ids)

    if not (len(types) == total_rows and len(codes) == total_rows):
        raise RuntimeError("Column length mismatch: id/type/code")

    batches = list(_iter_batches(ids, types, codes, batch_size))
    print(f"Extracting all features once (rows={total_rows}, batches={len(batches)}, batch_size={batch_size})...")

    if max_workers is None:

        max_workers = min(32, os.cpu_count() or 8)

    feature_map: Dict[Tuple[str, str], Optional[np.ndarray]] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(extractor_cls,),
    ) as ex:

        chunksize = 2
        it = ex.map(_worker_process_batch, batches, chunksize=chunksize)
        for r in tqdm(it, total=len(batches), desc="Extracting features"):
            feature_map.update(r)


    missing = 0
    for _id, _t in zip(ids, types):
        key = (str(_id), str(_t) if _t is not None else None)
        if key not in feature_map:
            missing += 1
    if missing:
        print(f"Warning: {missing} rows missing feature extraction results (unexpected).")

    return feature_map


def run_exported_parquet(lang: str, parquet_path: str, split_name: str, *, batch_size: int = 1024):
    extractor_cls = JavaMetricsExtractor if lang == "java" else None
    if extractor_cls is None:
        raise ValueError(f"Unsupported language: {lang}")

    print(f"Building metrics for {parquet_path}")
    df = pl.read_parquet(parquet_path)

    required = {"id", "type", "code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")


    out_dir = ProjectConfig.path_compiled_dataset
    os.makedirs(out_dir, exist_ok=True)


    feature_map = extract_all_features_once(df, extractor_cls, batch_size=batch_size)


    all_valid_feats = [f for f in feature_map.values() if f is not None]
    if not all_valid_feats:
        raise ValueError("No valid features found in the entire dataset!")
    all_valid_features = np.vstack(all_valid_feats).astype(np.float32, copy=False)


    global_stats_path = os.path.join(out_dir, f"{lang}_global_stats.npz")
    if split_name == "train":
        global_min = np.min(all_valid_features, axis=0).astype(np.float32, copy=False)
        global_max = np.max(all_valid_features, axis=0).astype(np.float32, copy=False)
        global_mean = np.mean(all_valid_features, axis=0).astype(np.float32, copy=False)

        np.savez(global_stats_path, global_min=global_min, global_max=global_max, global_mean=global_mean)
        print(f"Saved global stats to: {global_stats_path}")
    else:
        if not os.path.exists(global_stats_path):
            raise FileNotFoundError(f"Global stats file not found (train split required first): {global_stats_path}")
        train_stats = np.load(global_stats_path, allow_pickle=True)
        global_min = train_stats["global_min"].astype(np.float32, copy=False)
        global_max = train_stats["global_max"].astype(np.float32, copy=False)
        global_mean = train_stats["global_mean"].astype(np.float32, copy=False)
        print(f"Loaded global stats from train split: {global_stats_path}")

    # types
    types = df.select(pl.col("type").cast(pl.Utf8)).unique().to_series().to_list()
    types_sorted = sorted(types, key=lambda x: str(x) if x is not None else "")
    print(f"Found types ({len(types_sorted)}): {types_sorted[:20]}{'...' if len(types_sorted) > 20 else ''}")

    for tval in types_sorted:
        t_safe = _sanitize_type(tval)
        out_npz = os.path.join(out_dir, f"{lang}_{split_name}_metrics_{t_safe}.npz")
        if os.path.exists(out_npz):
            os.remove(out_npz)

        sub_df = df.filter(pl.col("type") == tval)
        ids_in_order = sub_df["id"].to_list()
        total = len(ids_in_order)

        if total == 0:
            print(f"[skip] type={tval} -> empty")
            continue

        t_key = str(tval) if tval is not None else None
        feats = [feature_map.get((str(_id), t_key), None) for _id in ids_in_order]

        ok_mask = [f is not None for f in feats]
        ok_feats = [f for f in feats if f is not None]

        if not ok_feats:
            type_mean = np.array([], dtype=np.float32)
            feat_dim = int(global_min.shape[0])
            normalized_all = np.zeros((total, feat_dim), dtype=np.float32)
        else:
            normalized_ok = min_max_normalize(ok_feats, global_min=global_min, global_max=global_max, eps=1e-8)
            type_mean = calculate_mean(ok_feats)

            feat_dim = int(normalized_ok.shape[-1])
            normalized_all = np.zeros((total, feat_dim), dtype=np.float32)

            j = 0
            for i, is_ok in enumerate(ok_mask):
                if is_ok:
                    normalized_all[i] = normalized_ok[j]
                    j += 1
                else:
                    normalized_all[i] = 0.0

        np.savez(
            out_npz,
            ids=np.array(ids_in_order, dtype=object),
            features=normalized_all.astype(np.float32, copy=False),
            ok_mask=np.array(ok_mask, dtype=np.bool_),
            global_min=global_min,
            global_max=global_max,
            global_mean=global_mean,
            type_mean=type_mean,
        )

        print(f"Saved: {out_npz} (type={tval}, items={total}, ok={sum(ok_mask)}/{total})")


def main():
    lang = "java"
    base = f"{ProjectConfig.path_origin_dataset}"

    run_exported_parquet(lang, f"{base}/step1_{lang}_train.parquet", "train", batch_size=2048)
    run_exported_parquet(lang, f"{base}/step1_{lang}_val.parquet", "val", batch_size=2048)
    run_exported_parquet(lang, f"{base}/step1_{lang}_test.parquet", "test", batch_size=2048)


if __name__ == "__main__":

    main()