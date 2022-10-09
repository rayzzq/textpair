from doc_encoder import DocModel
import pickle
import numpy as np
import torch
from tqdm import tqdm
import os
import pandas as pd
from collections import defaultdict


def infer_stage1_label(logits_ab, logits_a, logits_b):
    def ration_sent(logits):
        label = np.argmax(logits, axis=-1)
        label_idx = np.where(label == 1)[0].tolist()
        return label_idx

    def sigmoid(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    label_a = ration_sent(logits_a)
    label_b = ration_sent(logits_b)

    len_a = logits_a.shape[0]
    len_b = logits_b.shape[0]

    logits_ab = np.argmax(logits_ab, axis=-1)
    idxs = np.where(logits_ab == 1)[0].tolist()
    paired_ab = [[idx // len_b, idx % len_b] for idx in idxs]
    
    # logits_ab = sigmoid(logits_ab)
    # logits_ab = logits_ab[:, 1]
    # logits_ab = logits_ab.reshape(len_a, len_b)

    # row_mask = logits_ab.max(axis=-1) > 0.5
    # tmp_ab = []
    # for i, is_paired in enumerate(row_mask):
    #     if is_paired and i in label_a:
    #         j = np.argmax(logits_ab[i, :], axis=-1)
    #         tmp_ab.append([i, j])

    # tmp_dict = defaultdict(list)
    # for i, j in tmp_ab:
    #     tmp_dict[j].append(i)

    # paired_ab = []

    # for j, v in tmp_dict.items():
    #     if len(v) > 1:
    #         max_socre = 0
    #         for i in v:
    #             if logits_ab[i, j] > max_socre:
    #                 max_socre = logits_ab[i, j]
    #                 max_i = i
    #         paired_ab.append([max_i, j])
    #     else:
    #         paired_ab.append([v[0], j])

    return {"Case_A_rationales": label_a, "Case_B_rationales": label_b, "relation": paired_ab}


def sample_to_tensor(sample, model):

    # get stage_1 label:
    # Case_A_rationales
    # Case_B_rationales
    # relation
    res = infer_stage1_label(sample["logits_ab"], sample["logits_a"], sample["logits_b"])

    # get total document label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    emb_ab = sample["emb_ab"]
    logits_ab = sample["logits_ab"]

    emb_ab = torch.tensor(emb_ab)
    emb_ab = emb_ab.unsqueeze(0)

    logits = torch.tensor(sample.get("logits_ab"))
    logits = torch.softmax(logits, dim=-1)
    logits = logits[:, 1]
    logits = logits.unsqueeze(0)

    emb_ab = emb_ab.to(device)
    logits = logits.to(device)

    doc_seqs, doc_logits = model(emb_ab, logits)
    doc_label = torch.argmax(doc_logits, dim=-1)[0].tolist()
    res["label"] = doc_label
    res["id"] = sample["id"]

    return res


def stage2_inference(file_path, output_path, model):

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    data = pickle.load(open(file_path, "rb"))

    res = []
    for sample in tqdm(data):
        res.append(sample_to_tensor(sample, model))

    res = pd.DataFrame(res)

    res.to_csv(output_path, index=False, header=True, sep="\t")


if __name__ == "__main__":
    model_path = r"/home/wanghao/zzq/textpair/model/doced/2022-10-05-14-04/doc-step=40950-valid_acc_epoch=0.6133.ckpt"
    sub_file = r"/home/wanghao/zzq/textpair/data/val_stage1.pkl"

    model = DocModel.load_from_checkpoint(model_path)
    output_path = sub_file.split(".")[0] + ".final_resutls.csv"
    stage2_inference(sub_file, output_path, model)
