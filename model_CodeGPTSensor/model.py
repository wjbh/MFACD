import os
import sys

import torch
from transformers import BertModel, RobertaModel
from torch_geometric.nn import GATConv
from torch_geometric.utils import unbatch

# from transformers import RobertaModel, RobertaTokenizerFast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))
try:
    from config import ProjectConfig
    from pretrain_model.unixcoder import UniXcoder
except ImportError:
    from ..config import ProjectConfig
    from ..pretrain_model.unixcoder import UniXcoder


# bert_model_name='bert-base-uncased'
class FusionModel(torch.nn.Module):
    def __init__(self, code_model_name="microsoft/unixcoder-base",
                 matrix_dim_in=21, metrics_dim=100, gat_dim=128,
                 structure_embedding_dim=100, seq_len=256, logits_dim=128, num_node_types=77, local_files_only=True):
        super(FusionModel, self).__init__()
        self.code_bert_model = UniXcoder(code_model_name,
                                         cache_dir=f"{ProjectConfig.pretrain_path}/{code_model_name}",
                                         local_files_only=local_files_only)
        self.add_module('code_module', self.code_bert_model)
        code_bert_model_hidden_size = self.code_bert_model.config.hidden_size

        # self.fusion_proj = torch.nn.Linear(code_bert_model_hidden_size, logits_dim)
        self.classify_proj = torch.nn.Linear(code_bert_model_hidden_size, 2)

    def forward(self, _, __, code_tokens):
        """

            matrix (Tensor): shape [batch_size, seq_len, feature_dim]
            code_tokens (Tensor): shape [batch_size, code_seq_len]
            comment_tokens (Tensor): shape [batch_size, comment_seq_len]
        return:
            Tensor: shape [batch_size, output_dim]
        """
        mask = code_tokens.ne(0)
        code_vec = self.code_bert_model(
            input_ids=code_tokens, attention_mask=mask)
        if len(code_vec.shape) == 3:
            code_vec = code_vec[:, 0, :]
        vec = code_vec
        # vec = self.fusion_proj(vec)

        logits = self.classify_proj(vec)  # [batch_size, 2]
        return vec, logits


if __name__ == '__main__':
    #  batch_size = 4
    matrix_input = torch.randn(4, 21)  # [batch_size, seq_len, feature_dim]
    code_tokens = torch.randint(0, 10000, (4, 50))  # [batch_size, code_seq_len]
    comment_tokens = torch.randint(0, 10000, (4, 50))  # [batch_size, comment_seq_len]
    model = FusionModel(code_model_name="bert-base-uncased")

    output = model(matrix_input, code_tokens, comment_tokens)
    print(output.shape)  # : torch.Size([4, 1])
