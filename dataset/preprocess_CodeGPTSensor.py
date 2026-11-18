import csv
import os
import pickle
import sys
import json
import polars as pl
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import traceback

from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))
try:
    from dataset.impl.Exceptions import KnownException
    from dataset.impl.JavaCommentExtractor import JavaCommentExtractor
    from config import ProjectConfig, ProjectConst
except ImportError:
    from dataset.impl.Exceptions import KnownException
    from dataset.impl.JavaCommentExtractor import JavaCommentExtractor
    from config import ProjectConfig, ProjectConst

# label=0 human;  label=1 GPT
max_workers = 24
# comment_tokenizer_name = "bert-base-uncased"
code_tokenizer_name = "microsoft/unixcoder-base"
tokenizer_path = f"{ProjectConfig.pretrain_path}/{code_tokenizer_name}"
code_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                               cache_dir=f"{ProjectConfig.pretrain_path}/{code_tokenizer_name}", )


# TODO:


def chunked(lst, size):

    return [lst[i:i + size] for i in range(0, len(lst), size)]


def process_batch_with_encode(batch_items):
    results = {}
    for item in batch_items:
        index = item["index"]
        code = item["code"]
        contrast_code = item["contrast"]
        label = item["label"]
        try:


            if label == 0:
                code_false = code
                code_true = contrast_code
            else:
                code_false = contrast_code
                code_true = code
            encoded_list = code_tokenizer.batch_encode_plus((code_false, code_true)).encodings
            ids = (encoded_list[0].ids, encoded_list[1].ids)

            results[index] = ids
        except KnownException:
            continue
        except Exception as e:
            print(traceback.format_exc())
    return results


def run(lang, file):
    file_path = f"{ProjectConfig.path_origin_dataset}/{lang}/{file}"
    print(f"Building metrics for {file_path}")
    schema = {
        "index": pl.Utf8,
        "code": pl.Utf8,
        "contrast": pl.Utf8,
        "label": pl.Int32
    }
    if file.endswith("jsonl"):
        df = pl.read_ndjson(file_path, schema=schema)
        items = df.select(["index", "code", "contrast", "label"]).to_dicts()
    elif file.endswith("parquet"):
        df = pl.read_parquet(file_path, schema=schema)
        items = df.select(["index", "code", "contrast", "label"]).to_dicts()


    batch_size = 256
    batches = chunked(items, batch_size)


    preprocessed_data = {}

    print("Processing with encoding...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_batch_with_encode,
                batch,
            )
            for batch in batches
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing & Encoding"):
            try:
                batch_result = future.result()
                preprocessed_data.update(batch_result)
            except Exception as e:
                print(f"Error occurred in future: {e}")


    save_dir = f"{ProjectConfig.path_compiled_dataset}/{lang}"
    os.makedirs(save_dir, exist_ok=True)
    save_filepath = f"{save_dir}/{file.split('.')[0]}_unixcoder_only.pkl"
    print(f"Saving to {save_filepath}")
    with open(save_filepath, "wb+") as f:
        pickle.dump(preprocessed_data, f)


def main():
    run(ProjectConst.java, ProjectConst.train_parquet)
    run(ProjectConst.java, ProjectConst.test_parquet)


if __name__ == '__main__':
    main()
