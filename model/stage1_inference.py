import jsonlines
from sentence_classifier import SentenceClassifier, SentencePairClassifier
from torch.utils.data import Dataset
import pickle
import numpy as np
import torch
from tqdm import tqdm
import os


def sample_to_tensor(sample, single_model, paired_model):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    single_model.to(device)
    paired_model.to(device)
    single_model.eval()
    paired_model.eval()

    a = sample.get('Case_A')
    b = sample.get('Case_B')
    rel = sample.get("relation", None)
    rationa = sample.get("Case_A_rationales", None)
    rationb = sample.get("Case_B_rationales", None)

    # get single sentence embeddings
    ra = []
    rb = []
    if not (rationa is None or rationb is None):
        for i in range(len(a)):
            ra.append(int(i in rationa))
        for j in range(len(b)):
            rb.append(int(j in rationb))

    a_embedding, a_logits = single_model.encode(a, device=device, batch_size=128)
    b_embedding, b_logits = single_model.encode(b, device=device, batch_size=128)

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
            if rel is not None:
                rel_trues.append(int((ia, ib) in rel))

    paired_embedding, paired_logits = paired_model.encode(paired_sents, device=device, batch_size=64)
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

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with jsonlines.open(file_path, "r") as reader:
        data = list(reader)
    res = []
    for idx, sample in enumerate(tqdm(data)):
        res.append(sample_to_tensor(sample, single_model, paired_model))

    print("inference done, dumping to {}".format(output_path))
    pickle.dump(res, open(output_path, "wb"))

    return res


if __name__ == "__main__":
    single_ckp_path = "/home/wanghao/zzq/textpair/model/single/2022-10-04-13-02/simcse-chinese-roberta-wwm-ext-step=2281-valid_acc_epoch=0.8539.ckpt"
    paired_ckp_paht = "/home/wanghao/zzq/textpair/model/paired/2022-10-04-13-03/simcse-chinese-roberta-wwm-ext-step=1414-valid_acc_epoch=0.8933.ckpt"

    single_model = SentenceClassifier.load_from_checkpoint(single_ckp_path)
    paired_model = SentencePairClassifier.load_from_checkpoint(paired_ckp_paht)

    # data_root_dir = "/home/wanghao/zzq/textpair/data"
    # for t in ("train", "val"):
    #     file_path = f"{data_root_dir}/{t}.jsonl"
    #     output_path = f"{data_root_dir}/{t}_stage1.pkl"
    #     stage1_inference(file_path, output_path, single_model, paired_model)

    data_root_dir = "/home/wanghao/zzq/textpair/data"
    sub_files_stage1 = r"competition_stage_1_test.json"
    sub_files_stage2 = r"competition_stage_2_test.json"

    for file in (sub_files_stage1, sub_files_stage2):
        file_path = f"{data_root_dir}/raw/{file}"
        output_path = f"{data_root_dir}/submission/{file}.infer-stage1.pkl"
        stage1_inference(file_path, output_path, single_model, paired_model)
