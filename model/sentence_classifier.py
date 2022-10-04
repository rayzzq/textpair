import jsonlines
import random
import time
import os

import hydra
from transformers import AutoModel, AutoTokenizer

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim

import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger


class DataReader(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self._read_data()

        self.single_data = self._gen_single_data()
        self.paired_data = self._gen_paired_data()

    def get_data(self):
        return {"single": self.single_data, "paired": self.paired_data}

    def _read_data(self):
        with jsonlines.open(self.data_path, 'r') as f:
            data = list(f)
        return data

    def _gen_single_data(self):
        data = self.data
        pos_sents = []
        neg_sents = []

        def select_pos_sent(sents, labels):
            for i in range(len(sents)):
                if i in labels:
                    pos_sents.append(sents[i])
                else:
                    neg_sents.append(sents[i])

        for sample in data:
            select_pos_sent(sample["Case_A"], sample["Case_A_rationales"])
            select_pos_sent(sample["Case_B"], sample["Case_B_rationales"])
        random.shuffle(neg_sents)
        neg_sents = neg_sents[:int(len(pos_sents) * 1.2)]

        res = []
        for pos in pos_sents:
            res.append({'text': pos, 'label': 1})
        for neg in neg_sents:
            res.append({'text': neg, 'label': 0})
        return res

    def _gen_paired_data(self):
        data = self.data
        res = []
        for sample in data:
            res.extend(self._get_paired_samples(sample["Case_A"], sample["Case_B"], sample["relation"]))
        return res

    def _get_paired_samples(self, ca, cb, rel):
        rel = list(map(tuple, rel))
        pos = []
        for r in rel:
            ia, ib = r
            pos.append({'texta': ca[ia], 'textb': cb[ib], 'label': 1})

        neg = []
        while len(neg) < int(1.5 * len(pos)):
            ia = random.randint(0, len(ca) - 1)
            ib = random.randint(0, len(cb) - 1)
            if (ia, ib) not in rel:
                neg.append({'texta': ca[ia], 'textb': cb[ib], 'label': 0})
        return pos + neg


class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_batch(self, sents):
        encoded = self.tokenizer(sents, padding="longest", truncation=True, return_tensors='pt', max_length=512)
        return encoded

    def collate_fn(self, batch):
        text = []
        label = []
        for sample in batch:
            text.append(sample['text'])
            label.append(sample["label"])
        encoded = self.encode_batch(text)

        return {"input": encoded, "label": torch.tensor(label)}


class SentenceClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.hparams.update(cfg.train_args)
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained(cfg.pretrain_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrain_model_name_or_path)
        self.classifier = torch.nn.Sequential(nn.Dropout(0.2),
                                              nn.Linear(self.bert.config.hidden_size,
                                                        int(self.bert.config.hidden_size / 2)),
                                              nn.GELU(),
                                              nn.Linear(int(self.bert.config.hidden_size / 2), 2))

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def encode(self, texts, device="cuda:0"):

        self.to(device)

        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, list), "sinlge_model input must be batch of raw texts"

        def batch_tokenize(sents):
            encoded = self.tokenizer(sents,
                                     padding="longest",
                                     truncation=True,
                                     return_tensors='pt',
                                     max_length=512)
            return encoded

        encoded_sents = batch_tokenize(texts)
        for k in encoded_sents:
            encoded_sents[k].to(device)

        emb, logits = self(**encoded_sents)
        return emb, logits

    def forward(self, *args, **kwargs):
        ouput = self.bert(**kwargs)
        last = ouput.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
        emb = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        logits = self.classifier(emb)
        return emb, logits

    def training_step(self, batch, batch_idx):
        input = batch.get("input")
        _, logits = self(**input)
        label = batch.get("label")
        loss = F.cross_entropy(logits, label)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.train_acc.update(logits, label)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc_epoch',
                 self.train_acc.compute(),
                 sync_dist=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        input = batch.get("input")
        _, logits = self(**input)
        label = batch.get("label")
        self.valid_acc.update(logits, label)

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
        return self.train_data

    def val_dataloader(self):
        return self.val_data

    def set_dataloader(self, train_dataloader, val_dataloader):
        self.train_data = train_dataloader
        self.val_data = val_dataloader


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_batch(self, sents):
        encoded = self.tokenizer(sents, padding="longest", truncation=True, return_tensors='pt', max_length=512)
        return encoded

    def collate_fn(self, batch):
        sep_token = self.tokenizer.sep_token
        ab = []
        ba = []
        label = []
        for sample in batch:
            ab.append(sample['texta'] + sep_token + sample['textb'])
            ba.append(sample['textb'] + sep_token + sample['texta'])
            label.append(sample["label"])
        ab = self.encode_batch(ab)
        ba = self.encode_batch(ba)
        return {"input_ab": ab, "input_ba": ba, "label": torch.tensor(label)}


class SentencePairClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.hparams.update(cfg.train_args)
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained(cfg.pretrain_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrain_model_name_or_path)
        self.classifier = torch.nn.Sequential(nn.Dropout(0.2),
                                              nn.Linear(self.bert.config.hidden_size * 2,
                                                        int(self.bert.config.hidden_size / 2)),
                                              nn.GELU(),
                                              nn.Linear(int(self.bert.config.hidden_size / 2), 2))

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, input_ab, input_ba):
        out1 = self.bert(**input_ab).last_hidden_state.transpose(1, 2)
        out2 = self.bert(**input_ba).last_hidden_state.transpose(1, 2)
        # print(out1.shape, out2.shape)

        emb1 = torch.avg_pool1d(out1, kernel_size=out1.shape[-1]).squeeze(-1)  # [batch, 768]
        emb2 = torch.avg_pool1d(out2, kernel_size=out1.shape[-1]).squeeze(-1)  # [batch, 768]
        # print(emb1.shape, emb2.shape)

        emb = torch.cat([emb1, emb2], dim=-1)
        logits = self.classifier(emb)
        return emb, logits

    def encode(self, paired_text, device='cuda:0'):

        self.to(device)

        assert isinstance(paired_text, list)
        assert isinstance(paired_text[0], list)
        assert isinstance(paired_text[0][0], str)

        sep_token = self.tokenizer.sep_token

        def batch_tokenize(sents):
            encoded = self.tokenizer(sents,
                                     padding="longest",
                                     truncation=True,
                                     return_tensors='pt',
                                     max_length=512)
            return encoded

        ab = []
        ba = []
        for ta, tb in paired_text:
            ab.append(ta + sep_token + tb)
            ba.append(tb + sep_token + ta)

        encoded_ab = batch_tokenize(ab)

        for k in encoded_ab:
            encoded_ab[k].to(device)
        encoded_ba = batch_tokenize(ba)

        for k in encoded_ba:
            encoded_ba.to(device)

        emb, logits = self(encoded_ab, encoded_ba)

        return emb, logits

    def training_step(self, batch, batch_idx):
        input_ab = batch.get("input_ab")
        input_ba = batch.get("input_ba")
        _, logits = self(input_ab, input_ba)
        label = batch.get("label")
        loss = F.cross_entropy(logits, label)
        self.log("train_loss", loss, on_step=True)
        self.train_acc.update(logits, label)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ab = batch.get("input_ab")
        input_ba = batch.get("input_ba")
        _, logits = self(input_ab, input_ba)
        label = batch.get("label")
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
        return self.train_data

    def val_dataloader(self):
        return self.val_data

    def set_dataloader(self, train_dataloader, val_dataloader):
        self.train_data = train_dataloader
        self.val_data = val_dataloader


def build_dataloader(cfg, tokenizer=None):
    task_type = cfg.task
    data_dir = cfg.data_dir
    train_args = cfg.train_args

    train_file = os.path.join(data_dir, "train.jsonl")
    val_file = os.path.join(data_dir, "val.jsonl")

    train = DataReader(train_file).get_data()
    val = DataReader(val_file).get_data()

    if task_type == "single":
        train = train["single"]
        val = val["single"]
        train_dataset = SingleDataset(train, tokenizer)
        val_dataset = SingleDataset(val, tokenizer)

    elif task_type == "paired":
        train = train["paired"]
        val = val["paired"]
        train_dataset = PairedDataset(train, tokenizer)
        val_dataset = PairedDataset(val, tokenizer)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_args.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  collate_fn=train_dataset.collate_fn)

    val_dataset = DataLoader(val_dataset,
                             batch_size=train_args.batch_size,
                             shuffle=False,
                             num_workers=4,
                             collate_fn=val_dataset.collate_fn)

    return train_dataloader, val_dataset


def train_classifier(cfg, sent_cls):
    pl.seed_everything(cfg.seed)

    model = sent_cls(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrain_model_name_or_path)

    model_name = cfg.pretrain_model_name_or_path.split("/")[-1]

    train_dl, val_dl = build_dataloader(cfg, tokenizer)
    model.set_dataloader(train_dl, val_dl)

    train_args = cfg.train_args
    output_dir = cfg.output_dir

    time_now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    ckp_path = os.path.join(output_dir, time_now)
    ckp_name = model_name + "-{step}-{valid_acc_epoch:.4f}"

    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=ckp_path,
        filename=ckp_name,
        every_n_train_steps=train_args.save_steps,
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


@hydra.main(config_path="./", config_name="config", version_base="1.2")
def main(cfg):
    if cfg.task == "single":
        train_classifier(cfg.sentence_classifier, SentenceClassifier)
    elif cfg.task == "paired":
        train_classifier(cfg.sentence_pair_classifier, SentencePairClassifier)


if __name__ == "__main__":
    main()
