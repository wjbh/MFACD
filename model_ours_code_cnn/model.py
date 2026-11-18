# bert_model_name='bert-base-uncased'
import os
import sys

import torch
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
from torch_geometric.nn import GATv2Conv, global_mean_pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from config import ProjectConfig
except ImportError:
    from ..config import ProjectConfig


class MLP(torch.nn.Module):
    def __init__(self, dim_in=100, dim_out=100, activate_func=torch.nn.functional.relu):
        super().__init__()
        self.activate_func = activate_func
        t = dim_in + dim_out
        self.linear1 = torch.nn.Linear(dim_in, t)
        self.linear3 = torch.nn.Linear(t, dim_out)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(self.activate_func(x))
        x = self.linear3(x)
        return x


class LazySimpleGate(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.LazyLinear(dims),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)


class FusionModel(torch.nn.Module):
    """
    Fusion:
      - matrix (metrics) branch: outer-product -> flatten -> MLP -> metrics_dim
      - structure branch: node-type embedding -> GATv2Conv -> global_mean_pool -> structure_embedding_dim
      - code branch: UniXcoder / BERT / RoBERTa -> CLS -> code_hidden
      - comment branch: BERT -> CLS -> comment_hidden
      - fuse: [matrix_vec, structure_vec, code_vec + comment_vec] -> linear -> logits(2)

    Inputs:
      matrix:        (B, matrix_dim_in)   # e.g. 21
      structure:     PyG Batch(Data) with:
                       - x: (N,1) or (N,) long node_type_ids
                       - edge_index: (2, E)
                       - batch: (N,) graph ids
      code_tokens:   (B, L)
      comment_tokens:(B, L)
    Outputs:
      vec:    (B, cat_dims)
      logits: (B, 2)
    """

    def __init__(
            self,
            code_model_name: str = "microsoft/unixcoder-base",
            comment_model_name: str = "bert-base-uncased",
            matrix_dim_in: int = 18,
            metrics_dim: int = 100,
            gat_dim: int = 128,
            structure_embedding_dim: int = 100,
            seq_len: int = 256,  # kept for compatibility; not directly used
            logits_dim: int = 128,  # kept for compatibility; not directly used
            num_node_types: int = 77,
            local_files_only: bool = True,
    ):
        super().__init__()

        # ----------------------
        # Matrix branch
        # ----------------------
        self.matrix_dim_in = matrix_dim_in
        self.matrix_proj = MLP(matrix_dim_in * matrix_dim_in, metrics_dim)

        # ----------------------
        # Structure branch (GNN)
        # ----------------------
        self.node_type_embedding = torch.nn.Embedding(
            num_embeddings=num_node_types + 1,
            embedding_dim=gat_dim
        )
        # ✅ use GATv2Conv to avoid "Static graphs not supported in 'GATConv'"
        self.gat = GATv2Conv(gat_dim, structure_embedding_dim, heads=1, concat=False)

        # ----------------------
        # Code encoder
        # ----------------------
        self._code_model_kind = "unknown"
        self._code_is_unixcoder = False

        if code_model_name == "microsoft/unixcoder-base":
            try:
                from pretrain_model.unixcoder import UniXcoder
            except Exception:
                from ..pretrain_model.unixcoder import UniXcoder

            code_model = UniXcoder(
                code_model_name,
                cache_dir=f"{ProjectConfig.pretrain_path}/{code_model_name}",
                local_files_only=local_files_only,
            )
            self._code_model_kind = "unixcoder"
            self._code_is_unixcoder = True
            code_hidden = code_model.config.hidden_size

        elif code_model_name == "bert-base-uncased":
            code_model = BertModel.from_pretrained(
                code_model_name,
                cache_dir=f"{ProjectConfig.pretrain_path}/{code_model_name}",
                local_files_only=local_files_only,
            )
            self._code_model_kind = "bert"
            code_hidden = code_model.config.hidden_size

        else:
            code_model = RobertaModel.from_pretrained(
                code_model_name,
                cache_dir=f"{ProjectConfig.pretrain_path}/{code_model_name}",
                local_files_only=local_files_only,
            )
            self._code_model_kind = "roberta"
            code_hidden = code_model.config.hidden_size

        self.add_module("code_module", code_model)
        self.code_bert_model = code_model

        # ----------------------
        # Fusion head
        # ----------------------
        # original logic: code_vec + comment_vec (must match dims)
        # if you ever use different hidden sizes, replace with proj layers.
        self.code_align = None
        self.comment_align = None

        self.cat_dims = metrics_dim + structure_embedding_dim + code_hidden

        self.classify_proj = torch.nn.AdaptiveAvgPool1d(2)

    def _encode_code(self, code_tokens: torch.Tensor) -> torch.Tensor:
        """
        returns: (B, H)
        """
        if self._code_is_unixcoder:
            out = self.code_bert_model(input_ids=code_tokens)
            # UniXcoder may return (B, L, H) or already pooled.
            if isinstance(out, torch.Tensor):
                code_vec = out
            elif hasattr(out, "last_hidden_state"):
                code_vec = out.last_hidden_state
            else:
                code_vec = out
        else:
            mask = (code_tokens != 0).float()
            out = self.code_bert_model(input_ids=code_tokens, attention_mask=mask)
            code_vec = out.last_hidden_state

        if code_vec.dim() == 3:
            code_vec = code_vec[:, 0, :]  # CLS
        return code_vec

    def forward(self, matrix, structure, code_tokens):
        # ----------------------
        # matrix branch
        # ----------------------
        # matrix: (B, matrix_dim_in)
        # outer-product: (B, D, D)
        m = torch.matmul(matrix.unsqueeze(2), matrix.unsqueeze(1))
        matrix_vec = self.matrix_proj(m.flatten(start_dim=1))  # (B, metrics_dim)

        # ----------------------
        # structure branch
        # ----------------------
        # structure.x: (N,1) or (N,) -> (N,)
        node_type_ids = structure.x.view(-1).long()
        node_emb = self.node_type_embedding(node_type_ids)  # (N, gat_dim)

        node_out = self.gat(node_emb, structure.edge_index)  # (N, structure_embedding_dim)

        # graph pooling -> (B, structure_embedding_dim)
        structure_vec = global_mean_pool(node_out, structure.batch)

        # ----------------------
        # text branches
        # ----------------------
        code_vec = self._encode_code(code_tokens)

        # text_vec = code_vec + comment_vec

        # ----------------------
        # fuse + classify
        # ----------------------
        vec = torch.cat([matrix_vec, structure_vec, code_vec], dim=1)
        logits = self.classify_proj(vec)
        return vec, logits


if __name__ == "__main__":
    # demo (note: structure must be a PyG Batch in real usage)
    from torch_geometric.data import Data, Batch

    B = 4
    matrix_input = torch.randn(B, 21)
    code_tokens = torch.randint(0, 10000, (B, 256))
    comment_tokens = torch.randint(0, 10000, (B, 256))

    # build 4 tiny graphs
    graphs = []
    for _ in range(B):
        x = torch.randint(0, 10, (5, 1), dtype=torch.long)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index))
    structure_batch = Batch.from_data_list(graphs)

    model = FusionModel(local_files_only=True)
    vec, logits = model(matrix_input, structure_batch, code_tokens, comment_tokens)
    print(vec.shape, logits.shape)  # (B, cat_dims), (B, 2)

if __name__ == '__main__':
    #  batch_size = 4
    matrix_input = torch.randn(4, 21)  # [batch_size, seq_len, feature_dim]
    code_tokens = torch.randint(0, 10000, (4, 50))  # [batch_size, code_seq_len]
    comment_tokens = torch.randint(0, 10000, (4, 50))  # [batch_size, comment_seq_len]
    model = FusionModel()

    output = model(matrix_input, code_tokens, comment_tokens)
    print(output.shape)  # torch.Size([4, 1])
