from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Batch

from config import ProjectConst
try:
    from config import ProjectConfig, ProjectConst
    from dataset.preprocessed_dataset_combined import ContrastivePairDataset, custom_collate_fn_pair, \
    NormalClassificationDataset, custom_collate_fn_normal, _metrics_is_empty
except ImportError:
    from ..config import ProjectConfig, ProjectConst
    from dataset.preprocessed_dataset_combined import ContrastivePairDataset, custom_collate_fn_pair, \
        NormalClassificationDataset, custom_collate_fn_normal


def custom_collate_fn_normal(batch):
    """
    batch item:
      metrics: (D,)
      graph: Data
      code: (L,)
      comment: (L,)
      label: scalar
    return:
      metrics_batch: (B,D)
      structure_batch: PyG Batch with B graphs
      code_batch: (B,L)
      comment_batch: (B,L)
      label_batch: (B,)
    """
    metrics_list, graphs, codes, comments, labels = [], [], [], [], []
    for metrics, graph, code, comment, label in batch:
        metrics_list.append(metrics)
        graphs.append(graph)
        codes.append(code)
        comments.append(comment)
        labels.append(label)

    metrics_batch = torch.stack(metrics_list, dim=0)  # (B,D)
    structure_batch = Batch.from_data_list(graphs)  # num_graphs = B
    code_batch = torch.stack(codes, dim=0)  # (B,L)
    comment_batch = torch.stack(comments, dim=0)  # (B,L)
    label_batch = torch.stack(labels, dim=0).view(-1)  # (B,)

    return metrics_batch, structure_batch, code_batch, comment_batch, label_batch


def calculate_metrics_averages(dataset) -> Dict[str, np.ndarray]:
    """Calculate element-wise average metrics for each LLM type in the dataset"""
    averages = {}
    for llm_type, bundle in dataset.bundles.items():
        # Get all non-empty metrics arrays for this LLM type
        metrics_list = [metrics for metrics in bundle["metrics"].values() if not _metrics_is_empty(metrics)]
        if not metrics_list:
            averages[llm_type] = np.array([])  # Empty array if no valid metrics
            continue
        # Stack arrays and compute element-wise mean
        metrics_array = np.stack(metrics_list)
        averages[llm_type] = np.mean(metrics_array, axis=0)
    return averages



def main():
    code_tokenizer_name = "microsoft/unixcoder-base"
    comment_tokenizer_name = "bert-base-uncased"
    llm_types = [ProjectConst.llm_type_chatgpt]
    ds = ContrastivePairDataset(
        file_prefix="train",
        pos_llm_types=llm_types,
        neg_llm_types=[ProjectConst.llm_type_origin],
        lang="java",
        token_len=128,
        code_tokenizer_name=code_tokenizer_name,
    )

    model_averages = calculate_metrics_averages(ds)
    for llm_type, avg_metrics in model_averages.items():
        print(f"\nLLM Type: {llm_type}")
        print(f"Sample count: {len(ds.bundles[llm_type]['metrics'])}")
        print(f"Average metrics shape: {avg_metrics.shape}")
        print("Average metrics values:")
        print(avg_metrics)


if __name__ == "__main__":
    main()