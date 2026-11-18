import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d:%(filename)s:%(lineno)d:%(levelname)s:\t%(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S', )


class __ProjectConfig:
    def __init__(self):
        self.root = ""
        self.path_dataset_home = os.path.join(self.root, "data")
        self.path_origin_dataset = self.path_dataset_home
        self.path_compiled_dataset = os.path.join(self.path_dataset_home, "Compiled")
        self.pretrain_path = os.path.join(self.root, "pretrain")
        self.pretrain_bert_path = os.path.join(self.pretrain_path, "bert-base-uncased")
        self.pretrain_tokenizer_path = os.path.join(self.pretrain_bert_path, "tokenizer.json")
        self.train_log_dir = os.path.join(self.root, "Model")


class __ProjectConst:
    def __init__(self):
        self.java = "java"
        self.python = "python"
        self.train_parquet = "train.parquet"
        self.valid_parquet = "valid.parquet"
        self.test_parquet = "test.parquet"
        self.train_no_comment_jsonl = "train_no_comment.jsonl"
        self.valid_no_comment_jsonl = "valid_no_comment.jsonl"
        self.test_no_comment_jsonl = "test_no_comment.jsonl"
        self.llm_type_origin = "human"
        self.llm_type_deepseek_v3 = "dsc"
        self.llm_type_qwen3_coder_plus = "qwen"
        self.llm_type_chatgpt = "chatgpt"


ProjectConfig = __ProjectConfig()
ProjectConst = __ProjectConst()
