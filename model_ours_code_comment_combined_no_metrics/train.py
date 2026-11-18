import os
import sys
import json
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig, ProjectConst
    from model_ours_code_comment_combined_no_graph.train_model import ContrastiveLearningTrainModel
    from dataset.preprocessed_dataset_combined import ContrastivePairDataset, custom_collate_fn_pair, \
        NormalClassificationDataset, custom_collate_fn_normal
except ImportError:
    from ..config import ProjectConfig, ProjectConst
    from train_model import ContrastiveLearningTrainModel
    from dataset.preprocessed_dataset_combined import ContrastivePairDataset, custom_collate_fn_pair, \
        NormalClassificationDataset, custom_collate_fn_normal

token_len = 128
devices = [2]
batch_size = 20
max_epochs = 1
accumulate_grad_batches = 4


def train(lang, llm_types, code_tokenizer_name, comment_tokenizer_name):
    model_root_path = f"{ProjectConfig.train_log_dir}"


    result_dir = os.path.join(model_root_path, "test_results")
    os.makedirs(result_dir, exist_ok=True)


    train_dataset = ContrastivePairDataset(
        file_prefix="train",
        pos_llm_types=llm_types,
        neg_llm_types=[ProjectConst.llm_type_origin],
        lang=lang,
        token_len=token_len,
        code_tokenizer_name=code_tokenizer_name,
    )
    val_dataset = ContrastivePairDataset(
        file_prefix="val",
        pos_llm_types=llm_types,
        neg_llm_types=[ProjectConst.llm_type_origin],
        lang=lang,
        token_len=token_len,
        code_tokenizer_name=code_tokenizer_name,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn_pair,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size * 6,
        shuffle=False,
        collate_fn=custom_collate_fn_pair,
        num_workers=0,
    )

    contrastive_learning_model = ContrastiveLearningTrainModel()

    checkpoint_callback = ModelCheckpoint(
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    logger = TensorBoardLogger(save_dir=model_root_path, name="training_logs")

    trainer = pl.Trainer(
        accelerator="cuda",
        devices=devices,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        precision=16,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=False,
        benchmark=True,
    )

    trainer.fit(contrastive_learning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


    all_test_results = {}

    # test
    for test_llm_type in [ProjectConst.llm_type_origin] + llm_types:
        if test_llm_type == ProjectConst.llm_type_origin:
            test_dataset = NormalClassificationDataset(
                file_prefix="test",
                pos_llm_types=None,
                neg_llm_types=[test_llm_type],
                lang=lang,
                token_len=token_len,
                code_tokenizer_name=code_tokenizer_name,
            )
        else:
            test_dataset = NormalClassificationDataset(
                file_prefix="test",
                pos_llm_types=[test_llm_type],
                neg_llm_types=None,
                lang=lang,
                token_len=token_len,
                code_tokenizer_name=code_tokenizer_name,
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 16,
            shuffle=False,
            collate_fn=custom_collate_fn_normal,
            num_workers=0,
        )
        res = trainer.test(contrastive_learning_model, test_loader)
        print(test_llm_type)
        print(res)


        all_test_results[test_llm_type] = res[0] if res else {}


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lang_name = lang if isinstance(lang, str) else str(lang)
    json_filename = f"test_results_{lang_name}_{timestamp}.json"
    json_filepath = os.path.join(result_dir, json_filename)

    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_test_results, f, ensure_ascii=False, indent=4)

    print(f"\nresult path: {json_filepath}")
    print(f"model: {list(all_test_results.keys())}")


def main():
    lang = ProjectConst.java

    llm_types = [
        ProjectConst.llm_type_deepseek_v3,
        ProjectConst.llm_type_qwen3_coder_plus,
        ProjectConst.llm_type_chatgpt,
    ]

    code_tokenizer_name = "microsoft/unixcoder-base"
    comment_tokenizer_name = "bert-base-uncased"

    train(lang, llm_types, code_tokenizer_name, comment_tokenizer_name)


if __name__ == "__main__":
    main()