import os
import polars as pl

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import ProjectConfig, ProjectConst
except ImportError:
    from ..config import ProjectConfig, ProjectConst

language_type = "java"
HVSAIC_PATH = os.path.join(ProjectConfig.path_dataset_home, f"{language_type}_dataset.jsonl")
OUT_DIR = ProjectConfig.path_dataset_home
# 42
seed = 42

df = pl.read_ndjson(HVSAIC_PATH)


df = df.rename({"hm_index": "id"}).drop("docstring")


code_cols = ["human_code", "chatgpt_code", "dsc_code", "qwen_code"]

parts = []
for c in code_cols:
    t = c[:-5] if c.endswith("_code") else c  
    parts.append(
        df.select([
            pl.col("id"),
            pl.lit(t).alias("type"),
            pl.col(c).alias("code"),
        ])
    )

long_df = pl.concat(parts)


long_df = long_df.filter((~pl.col("code").is_null()) & (pl.col("code") != ""))


long_df = long_df.with_row_index("rid").with_columns(pl.col("rid").cast(pl.Int64))


rid_type = long_df.select(["rid", "type"])
groups = rid_type.group_by("type").agg(pl.col("rid"))

train_parts, val_parts, test_parts = [], [], []

for rid_list in groups["rid"].to_list():

    s = pl.Series("rid", rid_list).cast(pl.Int64).sample(
        n=len(rid_list), shuffle=True, seed=seed
    )

    n = len(rid_list)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_parts.append(pl.DataFrame({"rid": s[:n_train]}))
    val_parts.append(pl.DataFrame({"rid": s[n_train:n_train + n_val]}))
    test_parts.append(pl.DataFrame({"rid": s[n_train + n_val:]}))

train_rid_df = pl.concat(train_parts) if train_parts else pl.DataFrame({"rid": []})
val_rid_df   = pl.concat(val_parts)   if val_parts   else pl.DataFrame({"rid": []})
test_rid_df  = pl.concat(test_parts)  if test_parts  else pl.DataFrame({"rid": []})


train_rid_df = train_rid_df.with_columns(pl.col("rid").cast(pl.Int64))
val_rid_df   = val_rid_df.with_columns(pl.col("rid").cast(pl.Int64))
test_rid_df  = test_rid_df.with_columns(pl.col("rid").cast(pl.Int64))

train_df = train_rid_df.join(long_df, on="rid", how="inner").select(["id", "type", "code"])
val_df   = val_rid_df.join(long_df, on="rid", how="inner").select(["id", "type", "code"])
test_df  = test_rid_df.join(long_df, on="rid", how="inner").select(["id", "type", "code"])


os.makedirs(OUT_DIR, exist_ok=True)
train_df.write_parquet(os.path.join(OUT_DIR, f"step1_{language_type}_train.parquet"))
val_df.write_parquet(os.path.join(OUT_DIR, f"step1_{language_type}_val.parquet"))
test_df.write_parquet(os.path.join(OUT_DIR, f"step1_{language_type}_test.parquet"))

print(f"[{language_type}] total={long_df.height} train={train_df.height} val={val_df.height} test={test_df.height}")

check = pl.concat([
    train_df.with_columns(pl.lit("train").alias("split")),
    val_df.with_columns(pl.lit("val").alias("split")),
    test_df.with_columns(pl.lit("test").alias("split")),
]).group_by(["type", "split"]).count().sort(["type", "split"])
print(check)
