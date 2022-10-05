import time
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, RandomSampler

import pickle
import hydra

import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
import math 

class ScaleAttention(nn.Module):
    def __init__(self, temp, dropout=0.2, depth=1):
        super(ScaleAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.temp = temp
        self.depth = depth
        self.r = math.exp(4 - 2 * self.depth) / (math.exp(4 - 2 * self.depth) + 1)
        
    def forward(self, q, k, v, mask):
        # b, seq, n, dv
        atten = torch.matmul(q / self.temp, k.transpose(-1, -2))

        # TODO: is this correct to use mask label
        if mask is not None:
            atten.masked_fill_(mask.bool(), 1e-9)
        
        new_mask = mask.bool().float()
        new_mask = (1 - self.r) * new_mask + self.r * mask 
        # print("-"*10 + "debug" + "-"*10)
        # print(q.shape)
        # print(atten.shape)
        # print(mask.shape)
        # print("-"*10 + "debug" + "-"*10)

        atten = F.softmax(atten, dim=-1)
        atten = atten * new_mask
        atten = self.dropout(atten)
        out = torch.matmul(atten, v)

        return out, atten


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k=1536, d_v=1536, dropout=0.2, depth=1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaleAttention(temp=d_k ** 0.5, dropout=dropout, depth=depth)

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
            mask = mask.repeat(n_head, 1, 1).view(-1, len_q, 1)

        output, attention = self.attention(q, k, v, mask=mask)

        output = output.view(sz_b, n_head, len_q, d_v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attention


class DocAttentionBlock(nn.Module):
    def __init__(self, cfg, depth=1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model=cfg.hidden_size,
                                            n_head=cfg.num_heads,
                                            dropout=cfg.dropout,
                                            depth = depth)

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
        self.blocks = nn.ModuleList([DocAttentionBlock(cfg, depth=i) for i in range(cfg.num_layers)])

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

        self.doc_pad_embedding = nn.Parameter(torch.randn(cfg.hidden_size * 2))

    def forward(self, doc_seqs, doc_scores):
        doc_seqs = self.doc_encoder(doc_seqs, doc_scores)
        doc_seqs = doc_seqs.transpose(1, 2)
        cls_emb = torch.avg_pool1d(doc_seqs, kernel_size=doc_seqs.shape[-1]).squeeze(-1)
        logits = self.classifier(cls_emb)
        doc_seqs = doc_seqs.transpose(1, 2).contiguous()
        return doc_seqs, logits


class DocModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.hparams.update(cfg.train_args)
        self.save_hyperparameters()
        self.model = DocEncoder(cfg)
        self.tag_classifier = nn.Sequential(nn.Dropout(cfg.dropout),
                                            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
                                            nn.GELU(),
                                            nn.Linear(cfg.hidden_size // 2, 2),)
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

        
        
    def forward(self, doc_seqs, doc_scores):
        doc_seqs, logits = self.model(doc_seqs, doc_scores)
        return doc_seqs, logits


    def _tag_loss(self, seqs, tags, mask):
        # seq b * s * h
        # tag b * s 
        # mask b * s
        logit = self.tag_classifier(seqs).view(-1, 2)
        tags = tags.view(-1).long()
        mask = mask.bool().view(-1)
        loss = F.cross_entropy(logit[mask], tags[mask], reduction='mean')
        return loss
        
        
    def training_step(self, batch, batch_idx):
        emb_ab = batch.get("emb_ab")
        logits_ab = batch.get("logits_ab")
        label = batch.get("label")
        rel_ab = batch.get("rel_ab")
        doc_seqs, logits = self(emb_ab, logits_ab)
        doc_loss = F.cross_entropy(logits, label)
        tag_loss = self._tag_loss(doc_seqs, rel_ab, logits_ab)
        loss = 0.8 * doc_loss + 0.2 * tag_loss
        self.log("train_loss", loss, on_step=True)
        self.train_acc.update(logits, label)
        return loss

    def validation_step(self, batch, batch_idx):
        emb_ab = batch.get("emb_ab")
        logits_ab = batch.get("logits_ab")
        label = batch.get("label")
        
        doc_seqs, logits = self.model(emb_ab, logits_ab)
        self.valid_acc.update(logits, label)

    def training_epoch_end(self, outputs):
        self.log('train_acc_epoch',
                 self.train_acc.compute(),
                 sync_dist=True)
        self.train_acc.reset()

    def validation_epoch_end(self, outputs):
        self.log('valid_acc_epoch',
                 self.valid_acc.compute(),
                 sync_dist=True)
        self.valid_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def load_dataloader(self, cfg):
        collate_fn = Collate.from_model(self.model)
        
        train_data = pickle.load(open(cfg.train_data_path, "rb"))
        valid_data = pickle.load(open(cfg.valid_data_path, "rb"))

        train_dl = DataLoader(train_data,
                              batch_size=cfg.train_args.batch_size,
                              sampler=RandomSampler(train_data),
                              collate_fn=collate_fn,
                              num_workers=4)

        val_dl = DataLoader(valid_data,
                            batch_size=cfg.train_args.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=4)

        self.train_dl = train_dl
        self.val_dl = val_dl



class Collate(object):
    def __init__(self, model):
        super().__init__()
        self.model = model 
        self.device = self.model.doc_pad_embedding.device
        
    def __call__(self, batch):
        label = []
        emb_ab = []
        logits_ab = []
        rel_ab = []
        for sample in batch:
            emb_ab.append(torch.tensor(sample.get("emb_ab")))

            logits = torch.tensor(sample.get("logits_ab"))
            logits = torch.softmax(logits, dim=-1)
            logits = logits[:, 1]
            logits_ab.append(logits)
            label.append(sample.get("label"))
            
            rel_ab.append(torch.tensor(sample.get("rel_ab")))
        
        emb_ab, logits_ab, rel_ab = self._pad_to_longest(emb_ab, logits_ab, rel_ab)
        
        return {"emb_ab": emb_ab, "logits_ab": logits_ab, "label": torch.tensor(label), "rel_ab": rel_ab}
    
    
    def _pad_emb_seqs(self, tensor, max_seq_len):
        device = self.device
        pad_emb = self.model.doc_pad_embedding.unsqueeze(0)
        padded_seq = []
        padded_seq.append(tensor.to(device))
        padded_seq.extend([pad_emb] * (max_seq_len - tensor.shape[0]))
        padded_seq = torch.cat(padded_seq, dim=0)
        return padded_seq
    
    def _pad_tag_seqs(self, tag, max_seq_len):
        device = self.device
        padded_tag = []
        padded_tag.append(tag.to(device))
        padded_tag.append(torch.zeros(max_seq_len - tag.shape[0]).to(device))
        padded_tag = torch.cat(padded_tag, dim=0)
        return padded_tag
    
    
    def _pad_to_longest(self, doc_seqs, doc_scores, doc_tags):
        doc_lens = [len(seq) for seq in doc_seqs]
        max_seq_len = max(doc_lens)
        
        for i in range(len(doc_seqs)):
            padded_seq = self._pad_emb_seqs(doc_seqs[i], max_seq_len)
            padded_scores = self._pad_tag_seqs(doc_scores[i], max_seq_len)
            padded_tag = self._pad_tag_seqs(doc_tags[i], max_seq_len)
            
            doc_scores[i] = padded_scores.unsqueeze(0)
            doc_seqs[i] = padded_seq.unsqueeze(0)
            doc_tags[i] = padded_tag.unsqueeze(0)
            
            
        doc_seqs = torch.vstack(doc_seqs)
        doc_socres = torch.vstack(doc_scores)
        doc_tags = torch.vstack(doc_tags)
        
        return doc_seqs, doc_socres, doc_tags
    
    @classmethod
    def from_model(cls, model):
        return cls(model)


@hydra.main(config_path="./", config_name="config", version_base="1.2")
def train_doc_encoder(cfg):
    cfg = cfg.doc_encoder

    pl.seed_everything(cfg.seed)
    model = DocModel(cfg)
    model.load_dataloader(cfg)

    train_args = cfg.train_args
    output_dir = cfg.output_dir

    time_now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    ckp_path = os.path.join(output_dir, time_now)
    ckp_name = "doc-{step}-{valid_acc_epoch:.4f}"

    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)

    checkpoint = pl.callbacks.ModelCheckpoint(
        mode="max",
        dirpath=ckp_path,
        filename=ckp_name,
        monitor="valid_acc_epoch",
    )

    logger = TensorBoardLogger(output_dir)

    trainer = pl.Trainer(
        accelerator=train_args.accelerator,
        max_epochs=train_args.max_epochs,
        devices=train_args.devices,
        gradient_clip_val=train_args.gradient_clip_val,
        callbacks=[checkpoint],
        logger=logger
    )

    trainer.fit(model)


# def load_data(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     emb_ab = []
#     rel_ab = []
#     for sample in data[0:5]:
#         emb_ab.append(torch.tensor(sample['emb_ab']))
#         # rel_ab.append(torch.tensor(sample['rel_ab']))
#         soft_rel = torch.tensor(sample['logits_ab'])[:,1]
#         rel_ab.append(soft_rel)

#     return emb_ab, rel_ab


# @hydra.main(config_path='./', config_name='config', version_base=None)
# def test_forward(cfg):
#     cfg = cfg.doc_encoder
#     model = DocEncoder(cfg)
#     emb_ab, rel_ab = load_data(cfg.data_path)
#     doc_seqs, logits = model(emb_ab, rel_ab)
#     print(doc_seqs.shape)
#     print(logits.shape)
if __name__ == "__main__":
    train_doc_encoder()
