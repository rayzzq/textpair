import jsonlines
from sentence_classifier import SentenceClassifier, SentencePairClassifier
from torch.utils.data import Dataset
import pickle
import numpy as np
import torch
from tqdm import tqdm


def sample_to_tensor(sample, single_model, paired_model):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    single_model.to(device)
    paired_model.to(device)

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

    a_embedding, a_logits = single_model.encode(a, device=device)
    b_embedding, b_logits = single_model.encode(b, device=device)

    a_embedding = a_embedding.cpu().detach().numpy()
    b_embedding = b_embedding.cpu().detach().numpy()
    a_logits = a_logits.cpu().detach().numpy()
    b_logits = b_logits.cpu().detach().numpy()

    # get sentence pair embeddings
    rel_trues = []
    paired_sents = []
    for ia, senta in enumerate(a):
        for ib, sentb in enumerate(b):
            paired_sents.append([senta, sentb])
            rel_trues.append(int((ia, ib) in rel))

    paired_embedding, paired_logits = paired_model.encode(paired_sents, device=device)
    paired_embedding = paired_embedding.cpu().detach().numpy()
    paired_logits = paired_logits.cpu().detach().numpy()

    res = {
        "emb_a": a_embedding,
        "logits_a": a_logits,
        "ra": np.array(ra),
        "emb_b": b_embedding,
        "logits_b": b_logits,
        "rb": np.array(rb),
        "emb_ab": paired_embedding,
        "rel_ab": np.array(rel_trues),
        "logits_ab": paired_logits,
        "label": sample.get("label"),
        "id": sample.get("id")
    }

    return res


def stage1_inference(file_path, output_path, single_model, paired_model):
    with jsonlines.open(file_path, "r") as reader:
        data = list(reader)

    res = []
    for sample in tqdm(data):
        res.append(sample_to_tensor(sample, single_model, paired_model))

    print("inference done, dumping to {}".format(output_path))
    pickle.dump(res, open(output_path, "wb"))

    return res


if __name__ == "__main__":
    single_ckp_path = "/home/wanghao/zzq/textpair/model/single/2022-10-04-11-54/simcse-chinese-roberta-wwm-ext-step=4562-valid_acc_epoch=0.7661.ckpt"
    paired_ckp_paht = "/home/wanghao/zzq/textpair/model/paired/2022-10-04-11-50/simcse-chinese-roberta-wwm-ext-step=2828-valid_acc_epoch=0.8516.ckpt"
    data_root_dir = "/home/wanghao/zzq/textpair/data"

    single_model = SentenceClassifier.load_from_checkpoint(single_ckp_path)
    paired_model = SentencePairClassifier.load_from_checkpoint(paired_ckp_paht)

    for t in ("train", "val"):
        file_path = f"{data_root_dir}/{t}.jsonl"
        output_path = f"{data_root_dir}/{t}_stage1.jsonl"
        stage1_inference(file_path, output_path, single_model, paired_model)
