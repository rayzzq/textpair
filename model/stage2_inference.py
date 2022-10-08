import jsonlines
from doc_encoder import DocModel, Collate
from torch.utils.data import Dataset
import pickle
import numpy as np
import torch
from tqdm import tqdm
import os
import pandas as pd


def infer_stage1_label(logits_ab, logits_a, logits_b):
    def ration_sent(logits):
        label = np.argmax(logits, axis=-1)
        label_idx = np.where(label == 1)[0].tolist()
        return label_idx
    
    def sigmoid(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        print(e_x.shape)
        return e_x / e_x.sum(axis=-1, keepdims=True)
    
    label_a = ration_sent(logits_a)
    label_b = ration_sent(logits_b)

    len_a = logits_a.shape[0]
    len_b = logits_b.shape[0]


    logits_ab = sigmoid(logits_ab)
    logits_ab = logits_ab[:, 1]
    logits_ab = logits_ab.reshape(len_a, len_b)

    row_mask = logits_ab.max(axis=-1) > 0.5
    
    paired_ab = []
    for i, is_paired in enumerate(row_mask):
        if is_paired and i in label_a:
            j = np.argmax(logits_ab[i,:], axis=-1)
            paired_ab.append([i, j])
            
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
    res["doc_label"] = doc_label

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
    model_path = r""
    sub_files_stage1 = r""
    sub_files_stage2 = r""

    model = DocModel.load_from_checkpoint(model_path)
    for file in (sub_files_stage1, sub_files_stage2):
        output_path = file.split(".")[0] + "_final_resutls.csv"
        stage2_inference(file, output_path, model)
