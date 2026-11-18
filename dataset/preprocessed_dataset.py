import os

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
        lang: str,             
        llm_type: str,
        file_prefix: str,
        code_postfix: str,
        comment_postfix: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:

    metrics_path = f"{base_dir}/{lang}_{file_prefix}_metrics_{llm_type}.npz"
    graph_path = f"{base_dir}/{lang}_{file_prefix}_graph_{llm_type}.pkl"
    code_path = f"{base_dir}/{lang}_{file_prefix}_code_tokens_{code_postfix}_{llm_type}.pkl"
    comment_path = f"{base_dir}/{lang}_{file_prefix}_comment_tokens_{comment_postfix}_{llm_type}.pkl"

    metrics_map = _npz_to_id_map(metrics_path)
    graph_map = _load_pickle(graph_path)
    code_map = _load_pickle(code_path)

    # comment_map = _load_pickle(comment_path)


    if os.path.exists(comment_path):
        comment_map = _load_pickle(comment_path)
    else:
        print(f"[WARNING] Comment file not found: {comment_path}, using None for all samples")
        comment_map = {k: None for k in metrics_map.keys()}

    common = set(metrics_map.keys()) & set(graph_map.keys()) & set(code_map.keys()) & set(comment_map.keys())
    metrics_map = {k: metrics_map[k] for k in common}
    graph_map = {k: graph_map[k] for k in common}
    code_map = {k: code_map[k] for k in common}
    comment_map = {k: comment_map[k] for k in common}
    return metrics_map, graph_map, code_map, comment_map


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
        comment_postfix = comment_tokenizer_name.replace("/", "-")



        base_dir = ProjectConfig.path_compiled_dataset

        # load bundles for all involved types
        self.bundles: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for t in set(self.neg_llm_types + self.pos_llm_types):
            m, g, c, com = _load_one_type_bundle(
                base_dir=base_dir,
                lang=lang,
                llm_type=t,
                file_prefix=file_prefix,
                code_postfix=code_postfix,
                comment_postfix=comment_postfix,
            )
            self.bundles[t] = dict(metrics=m, graph=g, code=c, comment=com)


        all_ids = set()
        for t in self.bundles:
            all_ids |= set(self.bundles[t]["metrics"].keys())

        self.ids: List[str] = []
        self.id_to_negs: List[List[str]] = []
        self.id_to_pos: List[List[str]] = []
        self.id_combo_counts: List[int] = [] 

        for _id in sorted(all_ids):
            avail_negs = []
            for t in self.neg_llm_types:
                bm = self.bundles[t]["metrics"].get(_id)
                if bm is not None and (not _metrics_is_empty(bm)):
                    
                    if _id in self.bundles[t]["graph"] and _id in self.bundles[t]["code"] and _id in self.bundles[t][
                        "comment"]:
                        avail_negs.append(t)

            avail_pos = []
            for t in self.pos_llm_types:
                bm = self.bundles[t]["metrics"].get(_id)
                if bm is not None and (not _metrics_is_empty(bm)):
                    if _id in self.bundles[t]["graph"] and _id in self.bundles[t]["code"] and _id in self.bundles[t][
                        "comment"]:
                        avail_pos.append(t)

            if len(avail_negs) == 0 or len(avail_pos) == 0:
                continue

            self.ids.append(_id)
            self.id_to_negs.append(avail_negs)
            self.id_to_pos.append(avail_pos)
            self.id_combo_counts.append(len(avail_negs) * len(avail_pos))


        # prefix[i] = sum(counts[:i])，总长度 = prefix[N]
        self.prefix = np.zeros((len(self.id_combo_counts) + 1,), dtype=np.int64)
        if self.id_combo_counts:
            self.prefix[1:] = np.cumsum(np.asarray(self.id_combo_counts, dtype=np.int64))

    def __len__(self):
        return int(self.prefix[-1])

    def _locate(self, idx: int) -> Tuple[int, int]:

        # i :prefix[i] <= idx < prefix[i+1]
        i = int(np.searchsorted(self.prefix, idx, side="right") - 1)
        offset = int(idx - self.prefix[i])
        return i, offset

    def __getitem__(self, idx: int):
        i, offset = self._locate(idx)
        _id = self.ids[i]

        neg_types = self.id_to_negs[i]
        pos_types = self.id_to_pos[i]
        nneg = len(neg_types)
        npos = len(pos_types)


        # neg_i = offset // npos, pos_j = offset % npos
        neg_i = offset // npos
        pos_j = offset % npos

        neg_t = neg_types[neg_i]
        pos_t = pos_types[pos_j]

        bneg = self.bundles[neg_t]
        bpos = self.bundles[pos_t]

        # metrics (2,D)
        m_neg = torch.tensor(bneg["metrics"][_id], dtype=torch.float32)
        m_pos = torch.tensor(bpos["metrics"][_id], dtype=torch.float32)
        metrics_2 = torch.stack([m_neg, m_pos], dim=0)

        # graph (len=2)
        g_neg = _tuple_graph_to_pyg(bneg["graph"][_id])
        g_pos = _tuple_graph_to_pyg(bpos["graph"][_id])
        graph_views = [g_neg, g_pos]

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

        # comment (2,L)
        com_neg = bneg["comment"][_id]
        com_pos = bpos["comment"][_id]
        comment_neg = torch.zeros((self.token_len,), dtype=torch.long) if com_neg is None else truncate_or_pad_1d(
            torch.tensor(com_neg, dtype=torch.long), self.token_len, 0
        )
        comment_pos = torch.zeros((self.token_len,), dtype=torch.long) if com_pos is None else truncate_or_pad_1d(
            torch.tensor(com_pos, dtype=torch.long), self.token_len, 0
        )
        comment_2 = torch.stack([comment_neg, comment_pos], dim=0)

        label_2 = torch.tensor([0, 1], dtype=torch.long)

        return metrics_2, graph_views, code_2, comment_2, label_2


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
    metrics_list, code_list, comment_list, label_list = [], [], [], []
    graphs_flat = []

    for metrics_2, graph_views, code_2, comment_2, label_2 in batch:
        metrics_list.append(metrics_2)
        code_list.append(code_2)
        comment_list.append(comment_2)
        label_list.append(label_2)
        graphs_flat.extend(graph_views)  # 2 graphs per sample

    metrics_batch = torch.stack(metrics_list, dim=0)  # (B,2,D)
    code_batch = torch.stack(code_list, dim=0)  # (B,2,L)
    comment_batch = torch.stack(comment_list, dim=0)  # (B,2,L)
    label_batch = torch.stack(label_list, dim=0)  # (B,2)

    structure_batch = Batch.from_data_list(graphs_flat)  # num_graphs = B*2
    return metrics_batch, structure_batch, code_batch, comment_batch, label_batch


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

    for metrics_batch, structure_batch, code_batch, comment_batch, label_batch in dl:
        print(metrics_batch.shape)  # (B,2,D)
        print(code_batch.shape)  # (B,2,L)
        print(structure_batch.num_graphs)  # B*2
        break


class NormalClassificationDataset(Dataset):


    def __init__(
            self,
            *,
            file_prefix: str,
            neg_llm_types: Optional[List[str]] = None,
            pos_llm_types: Optional[List[str]] = None,
            lang: str,
            token_len: int = 256,
            code_tokenizer_name: str = "microsoft/unixcoder-base",
            comment_tokenizer_name: str = "bert-base-uncased",
    ):
        neg_llm_types = list(neg_llm_types or [])
        pos_llm_types = list(pos_llm_types or [])
        llm_types = neg_llm_types + pos_llm_types
        if not llm_types:
            raise ValueError("Both neg_llm_types and pos_llm_types are empty. Provide at least one llm_type.")

        self.file_prefix = file_prefix
        self.lang = lang
        self.token_len = token_len

        self.neg_set = set(neg_llm_types)
        self.pos_set = set(pos_llm_types)

        code_postfix = code_tokenizer_name.replace("/", "-")
        comment_postfix = comment_tokenizer_name.replace("/", "-")


        base_dir = ProjectConfig.path_compiled_dataset

        # load bundles for all requested types
        self.bundles: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for t in set(llm_types):
            m, g, c, com = _load_one_type_bundle(
                base_dir=base_dir,
                lang=lang,
                llm_type=t,
                file_prefix=file_prefix,
                code_postfix=code_postfix,
                comment_postfix=comment_postfix,
            )
            self.bundles[t] = dict(metrics=m, graph=g, code=c, comment=com)

        # build samples list: [(llm_type, _id), ...]
        self.samples: List[Tuple[str, str]] = []
        for t in llm_types:
            b = self.bundles.get(t)
            if b is None:
                continue
            # b["metrics"] keys already satisfy four-tuple intersection from _load_one_type_bundle
            for _id, met in b["metrics"].items():
                if _metrics_is_empty(met):
                    continue
                self.samples.append((t, _id))

    def __len__(self):
        return len(self.samples)

    def _label_of(self, llm_type: str) -> int:

        if llm_type in self.pos_set:
            return 1
        if llm_type in self.neg_set:
            return 0

        return 1

    def __getitem__(self, idx: int):
        llm_type, _id = self.samples[idx]
        b = self.bundles[llm_type]

        metrics = torch.tensor(b["metrics"][_id], dtype=torch.float32)  # (D,)
        graph = _tuple_graph_to_pyg(b["graph"][_id])  # Data
        code_tokens = b["code"][_id]
        comment_tokens = b["comment"][_id]

        if code_tokens is None:
            code = torch.zeros((self.token_len,), dtype=torch.long)
        else:
            code = truncate_or_pad_1d(torch.tensor(code_tokens, dtype=torch.long), self.token_len, pad_value=0)

        if comment_tokens is None:
            comment = torch.zeros((self.token_len,), dtype=torch.long)
        else:
            comment = truncate_or_pad_1d(torch.tensor(comment_tokens, dtype=torch.long), self.token_len, pad_value=0)

        label = torch.tensor(self._label_of(llm_type), dtype=torch.long)  # scalar

        return metrics, graph, code, comment, label


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


if __name__ == "__main__":
    main()
