import jsonlines
from sentence_classifier import SentenceClassifier, SentencePairClassifier
import torch
from torch.utils.data import Dataset
import random
import numpy as np


def convert_to_tensor(sample, single_model, paired_model):
    a = sample['Case_A']
    b = sample['Case_B']
    rel = sample["relation"]
    rationa = sample["Case_A_rationales"]
    rationb = sample["Case_B_rationales"]

    # get single sentence embeddings
    ra = []
    rb = []
    for i in range(len(a)):
        ra.append(int(i in rationa))
    for j in range(len(b)):
        rb.append(int(j in rationb))

    a_embedding = single_model.encode(a)
    b_embedding = single_model.ecnode(b)

    # get sentence pair embeddings
    rel_trues = []
    paired_sents = []
    for ia, senta in enumerate(a):
        for ib, sentb in enumerate(b):
            paired_sents.append([senta, sentb])
            rel_trues.append(int((ia, ib) in rel))
    paired_embedding = paired_model.encode(paired_sents)
    res = {
        "emb_a": a_embedding,
        "ra": np.array(ra),
        "emb_b": b_embedding,
        "rb": np.array(rb),
        "emb_ab": paired_embedding,
        "rel_ab": np.array(rel_trues),
        "label": sample.get("label"),
        "id": sample.get("id")
    }
    return res




if __name__ == "__main__":
    pass