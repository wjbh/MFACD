import pickle
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

try:
    from config import ProjectConfig, ProjectConst
except ImportError:
    from config import ProjectConfig, ProjectConst


def truncate_or_pad_1d(seq: torch.Tensor, token_len: int, pad_value: int = 0) -> torch.Tensor:
    if seq.numel() > token_len:
        return seq[:token_len]
    if seq.numel() < token_len:
        pad_size = token_len - seq.numel()
        return torch.cat([seq, torch.full((pad_size,), pad_value, dtype=seq.dtype)], dim=0)
    return seq


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _npz_to_id_map(npz_path: str) -> Dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    if "indices" in d.files:
        ids = d["indices"]
    elif "ids" in d.files:
        ids = d["ids"]
    else:
        raise ValueError(f"npz missing ids/indices: {npz_path}")

    feats = d["features"]
    if feats.ndim == 2:
        pass
    elif feats.ndim == 3:
        feats = feats.mean(axis=1)
    else:
        raise ValueError(f"unexpected features shape {feats.shape} in {npz_path}")

    return dict(zip(ids.tolist(), feats))


def _tuple_graph_to_pyg(graph_tuple):
    if graph_tuple is None:
        return Data(x=torch.zeros((1, 1), dtype=torch.long), edge_index=torch.zeros((2, 0), dtype=torch.long))
    try:
        nodes_dict, edges = graph_tuple
    except Exception:
        return Data(x=torch.zeros((1, 1), dtype=torch.long), edge_index=torch.zeros((2, 0), dtype=torch.long))
    if not nodes_dict:
        return Data(x=torch.zeros((1, 1), dtype=torch.long), edge_index=torch.zeros((2, 0), dtype=torch.long))

    node_ids = sorted(nodes_dict.keys())
    id2i = {nid: i for i, nid in enumerate(node_ids)}
    node_types = [int(nodes_dict[nid]) for nid in node_ids]
    x = torch.tensor(node_types, dtype=torch.long).view(-1, 1)

    src, dst = [], []
    if edges:
        for u, v in edges:
            if u in id2i and v in id2i:
                src.append(id2i[u])
                dst.append(id2i[v])
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def _load_one_type_bundle(
        *,
        base_dir: str,
        llm_type: str,
        file_prefix: str,
        code_postfix: str,
):
    t_dir = f"{base_dir}/{llm_type}"

    code_path = f"{t_dir}/{file_prefix}_tokens_{code_postfix}.pkl"
    code_map = _load_pickle(code_path)

    return code_map


def _metrics_is_empty(v: Any) -> bool:
    if v is None:
        return True
    try:
        a = np.asarray(v)
    except Exception:
        return True
    if a.size == 0:
        return True
    if hasattr(a, "shape") and any(int(s) == 0 for s in a.shape):
        return True
    return False


class ContrastivePairDataset(Dataset):


    def __init__(
            self,
            *,
            file_prefix: str,
            neg_llm_types: List[str],
            pos_llm_types: List[str],
            lang: str = ProjectConst.java,
            token_len: int = 256,
            code_tokenizer_name: str = "microsoft/unixcoder-base",
            comment_tokenizer_name: str = "bert-base-uncased",
    ):
        assert neg_llm_types and pos_llm_types
        self.file_prefix = file_prefix
        self.token_len = token_len
        self.neg_llm_types = list(neg_llm_types)
        self.pos_llm_types = list(pos_llm_types)

        code_postfix = code_tokenizer_name.replace("/", "-")
        base_dir = f"{ProjectConfig.path_compiled_dataset}/{lang}"

        # load bundles for all involved types
        self.bundles: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for t in set(self.neg_llm_types + self.pos_llm_types):
            c = _load_one_type_bundle(
                base_dir=base_dir,
                llm_type=t,
                file_prefix=file_prefix,
                code_postfix=code_postfix)
            self.bundles[t] = dict(code=c)


        all_ids = set()
        for t in self.bundles:
            all_ids |= set(self.bundles[t]["code"].keys())

        self.ids: List[str] = []
        self.id_to_negs: List[List[str]] = []
        self.id_to_pos: List[List[str]] = []
        self.id_combo_counts: List[int] = [] 

        for _id in sorted(all_ids):
            avail_negs = []
            for t in self.neg_llm_types:
                    if self.bundles[t]["code"]:
                        avail_negs.append(t)

            avail_pos = []
            for t in self.pos_llm_types:
                if _id in self.bundles[t]["code"]:
                    avail_pos.append(t)

            if len(avail_negs) == 0 or len(avail_pos) == 0:
                continue

            self.ids.append(_id)
            self.id_to_negs.append(avail_negs)
            self.id_to_pos.append(avail_pos)
            self.id_combo_counts.append(len(avail_negs) * len(avail_pos))


        self.prefix = np.zeros((len(self.id_combo_counts) + 1,), dtype=np.int64)
        if self.id_combo_counts:
            self.prefix[1:] = np.cumsum(np.asarray(self.id_combo_counts, dtype=np.int64))

    def __len__(self):
        return int(self.prefix[-1])

    def _locate(self, idx: int) -> Tuple[int, int]:

        i = int(np.searchsorted(self.prefix, idx, side="right") - 1)
        offset = int(idx - self.prefix[i])
        return i, offset

    def __getitem__(self, idx: int):
        i, offset = self._locate(idx)
        _id = self.ids[i]

        neg_types = self.id_to_negs[i]
        pos_types = self.id_to_pos[i]
        npos = len(pos_types)


        neg_i = offset // npos
        pos_j = offset % npos

        neg_t = neg_types[neg_i]
        pos_t = pos_types[pos_j]

        bneg = self.bundles[neg_t]
        bpos = self.bundles[pos_t]

        # code (2,L)
        c_neg = bneg["code"][_id]
        c_pos = bpos["code"][_id]
        code_neg = torch.zeros((self.token_len,), dtype=torch.long) if c_neg is None else truncate_or_pad_1d(
            torch.tensor(c_neg, dtype=torch.long), self.token_len, 0
        )
        code_pos = torch.zeros((self.token_len,), dtype=torch.long) if c_pos is None else truncate_or_pad_1d(
            torch.tensor(c_pos, dtype=torch.long), self.token_len, 0
        )
        code_2 = torch.stack([code_neg, code_pos], dim=0)
        label_2 = torch.tensor([0, 1], dtype=torch.long)

        return code_2, label_2


def custom_collate_fn_pair(batch):
    """
    batch item:
      metrics_2: (2,D)
      graph_views: [Data_neg, Data_pos]
      code_2: (2,L)
      comment_2: (2,L)
      label_2: (2,)
    return:
      metrics_batch: (B,2,D)
      structure_batch: PyG Batch with B*2 graphs (sample-major)
      code_batch: (B,2,L)
      comment_batch: (B,2,L)
      label_batch: (B,2)
    """
    code_list, label_list = [], []

    for code_2, label_2 in batch:
        code_list.append(code_2)
        label_list.append(label_2)

    code_batch = torch.stack(code_list, dim=0)         # (B,2,L)
    label_batch = torch.stack(label_list, dim=0)       # (B,2)

    return code_batch, label_batch


# ---- demo ----
def main():
    ds = ContrastivePairDataset(
        file_prefix="train",
        lang=ProjectConst.java,
        token_len=256,
        neg_llm_types=[
            ProjectConst.llm_type_origin,
            # other neg types...
        ],
        pos_llm_types=[
            ProjectConst.llm_type_deepseek_v3,
            # other pos types...
        ],
        code_tokenizer_name="microsoft/unixcoder-base",
        comment_tokenizer_name="bert-base-uncased",
    )

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=custom_collate_fn_pair)

    for code_batch, label_batch in dl:
        print(code_batch.shape)             # (B,2,L)
        break


if __name__ == "__main__":
    main()
