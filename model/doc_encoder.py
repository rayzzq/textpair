
import torch
import torch.nn.functional as F
import torch.nn as nn

import pickle
import hydra

from transformers import AutoModel, AutoTokenizer


class ScaleAttention(nn.Module):
    def __init__(self, temp, p=0.2):
        super(ScaleAttention, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.temp = temp

    def forward(self, q, k, v, mask):
        # b, seq, n, dv
        atten = torch.matmul(q / self.temp, k.transpose(-1, -2))
        if mask is not None:
            atten.masked_fill(mask, 1e-9)

        print("-"*10 + "debug" + "-"*10)
        print(q.shape)
        print(atten.shape)
        print(mask.shape)
        print("-"*10 + "debug" + "-"*10)

        atten = F.softmax(atten, dim=-1)
        atten = self.dropout(atten)
        out = torch.matmul(atten, v)
        return out, atten


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k=1536, d_v=1536, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaleAttention(temp=d_k ** 0.5)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        # q k v: [b, seq, d_model]
        # mask: [b, seq]

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.transpose(1, 2).contiguous().view(-1, len_q, d_k)
        k = k.transpose(1, 2).contiguous().view(-1, len_k, d_k)
        v = v.transpose(1, 2).contiguous().view(-1, len_v, d_v)

        if mask is not None:
            mask = mask.view(sz_b, 1, len_q, 1)
        output, attention = self.attention(q, k, v, mask=mask)

        output = output.view(sz_b, n_head, len_q, d_v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attention


class DocAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attention = MultiHeadAttention(d_model=cfg.hidden_size,
                                            n_head=cfg.num_heads,
                                            dropout=cfg.dropout)

        self.ln = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.activation = nn.GELU()
        self.fc = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x, mask=None):
        residual = x
        x, _ = self.attention(x, x, x, mask=mask)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        x = self.ln(x + residual)
        return x


class DocTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size)
        self.blocks = nn.ModuleList([DocAttentionBlock(cfg) for _ in range(cfg.num_layers)])

    def forward(self, x, mask=None):
        x = self.fc(x)
        residual = x
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x + residual)
        return x


class DocEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.doc_encoder = DocTransformer(cfg)
        self.classifier = nn.Sequential(nn.Dropout(cfg.dropout),
                                        nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
                                        nn.GELU(),
                                        nn.Linear(cfg.hidden_size // 2, cfg.num_classes),)

        self.doc_cls_embedding = nn.Parameter(torch.randn(cfg.hidden_size * 2))
        self.doc_pad_embedding = nn.Parameter(torch.randn(cfg.hidden_size * 2))

    def forward(self, doc_seqs, doc_scores):
        doc_seqs, doc_scores = self._pad_to_longest(doc_seqs, doc_scores)
        doc_seqs = self.doc_encoder(doc_seqs, doc_scores)
        cls_emb = doc_seqs[:, 0, ...]
        logits = self.classifier(cls_emb)
        return doc_seqs, logits

    def _pad_cls_seqs(self, tensor, max_seq_len):
        cls_emb = self.doc_cls_embedding.unsqueeze(0)
        pad_emb = self.doc_pad_embedding.unsqueeze(0)
        padded_seq = [cls_emb]
        padded_seq.append(tensor)
        padded_seq.extend([pad_emb] * (max_seq_len - tensor.shape[0]))
        padded_seq = torch.cat(padded_seq, dim=0)
        return padded_seq

    def _pad_cls_socres(self, doc_score, max_seq_len):
        scores = [torch.ones(1)]
        scores.append(doc_score)
        scores.append(torch.zeros(max_seq_len - doc_score.shape[0]))
        padded_scores = torch.cat(scores, dim=0)
        return padded_scores

    def _pad_to_longest(self, doc_seqs, doc_scores):
        doc_lens = [len(seq) for seq in doc_seqs]
        max_seq_len = max(doc_lens)
        for i in range(len(doc_seqs)):
            padded_seq = self._pad_cls_seqs(doc_seqs[i], max_seq_len)
            padded_socres = self._pad_cls_socres(doc_scores[i], max_seq_len)
            doc_scores[i] = padded_socres.unsqueeze(0)
            doc_seqs[i] = padded_seq.unsqueeze(0)

        doc_seqs = torch.vstack(doc_seqs)
        doc_socres = torch.vstack(doc_scores)

        return doc_seqs, doc_socres


def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    emb_ab = []
    rel_ab = []
    for sample in data[0:5]:
        emb_ab.append(torch.tensor(sample['emb_ab']))
        rel_ab.append(torch.tensor(sample['rel_ab']))

    return emb_ab, rel_ab


@hydra.main(config_path='./', config_name='config', version_base=None)
def test_forward(cfg):
    cfg = cfg.doc_encoder
    model = DocEncoder(cfg)
    emb_ab, rel_ab = load_data(cfg.data_path)
    doc_seqs, logits = model(emb_ab, rel_ab)

    print(doc_seqs.shape)
    print(logits.shape)


if __name__ == "__main__":
    test_forward()
