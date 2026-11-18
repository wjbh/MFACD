import traceback
import concurrent.futures
import os
import sys
import cachetools.func
import pytorch_lightning as pl
import torch
from torch_optimizer import Lamb
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))

try:
    from model_ours_code_comment_combined_no_token.model import FusionModel
except ImportError:
    from .model import FusionModel


def get_contrast_loss(vec, contrast):
    label_new = torch.full((vec.size(0),), -1, device=vec.device)
    loss = torch.nn.functional.cosine_embedding_loss(vec, contrast, label_new)
    return loss


def get_kl_loss(p, q, pad_mask=None):
    p_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(p, dim=-1),
                                        torch.nn.functional.softmax(q, dim=-1), reduction='none')
    q_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(q, dim=-1),
                                        torch.nn.functional.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = p_loss + q_loss
    return loss


def nt_xent_loss(z1, z2, temperature=0.5):

    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    logits = z1 @ z2.t() * torch.exp(torch.tensor(temperature).log())
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


class ContrastiveLearningTrainModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FusionModel()

    def forward(self, metrics, structure, code):
        vec, logits = self.model(metrics, structure, code)
        return vec, logits

    def training_step(self, batch, batch_idx):
        metrics_tensor, structure_info, code_tensor, labels = batch
        labels = labels.transpose(-1, -2).reshape(-1)

        metrics_pos = metrics_tensor[:, 0]  # shape: (B, seq_len, feature_dim)
        metrics_neg = metrics_tensor[:, 1]

        code_pos = code_tensor[:, 0]  # shape: (B, L_code)
        code_neg = code_tensor[:, 1]


        all_metrics = torch.cat([metrics_pos, metrics_neg], dim=0)  # (2B, seq_len, feature_dim)
        all_code = torch.cat([code_pos, code_neg], dim=0)  # (2B, L_code)


        vec1, logits1 = self(all_metrics, structure_info, all_code)  # (2B, D)
        vec_t, logits2 = self(all_metrics, structure_info, all_code)

        vec2, vec3 = torch.chunk(vec_t, chunks=2, dim=0)  # (B, D), (B, D)
        pos_logits2, neg_logits2 = torch.chunk(logits2, chunks=2, dim=0)  # (B, D), (B, D)


        contrast_loss = get_contrast_loss(vec2, vec3)
        kl_loss = get_kl_loss(logits1, logits2)
        classify_loss = torch.nn.functional.cross_entropy(logits1, labels)
        loss = classify_loss + 0.05 * kl_loss + 0.2 * contrast_loss
        self.log("train_loss", loss, prog_bar=True)
        preds = torch.argmax(logits1, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        metrics_tensor, structure_info, code_tensor, labels = batch
        labels = labels.transpose(-1, -2).reshape(-1)

        metrics_pos = metrics_tensor[:, 0]  # shape: (B, seq_len, feature_dim)
        metrics_neg = metrics_tensor[:, 1]

        code_pos = code_tensor[:, 0]  # shape: (B, L_code)
        code_neg = code_tensor[:, 1]


        all_metrics = torch.cat([metrics_pos, metrics_neg], dim=0)  # (2B, seq_len, feature_dim)
        all_code = torch.cat([code_pos, code_neg], dim=0)  # (2B, L_code)


        vec1, logits1 = self(all_metrics, structure_info, all_code)  # (2B, D)
        vec_t, logits2 = self(all_metrics, structure_info, all_code)

        vec2, vec3 = torch.chunk(vec_t, chunks=2, dim=0)  # (B, D), (B, D)
        pos_logits2, neg_logits2 = torch.chunk(logits2, chunks=2, dim=0)  # (B, D), (B, D)


        contrast_loss = get_contrast_loss(vec2, vec3)
        kl_loss = get_kl_loss(logits1, logits2)
        classify_loss = torch.nn.functional.cross_entropy(logits1, labels)
        loss = classify_loss + 0.05 * kl_loss + 0.2 * contrast_loss
        self.log("val_loss", loss, prog_bar=True)
        preds = torch.argmax(logits1, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        metrics_tensor, structure_info, code_tensor, labels = batch

        vec1, logits1 = self(metrics_tensor, structure_info, code_tensor)  # (2B, D)

        preds = torch.argmax(logits1, dim=1)
        acc = (preds == labels).float().mean()
        self.log("acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Lamb(self.parameters(), lr=1e-3, weight_decay=1e-5, eps=1e-11)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer
