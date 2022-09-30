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
            res.append(self._get_paired_samples(sample["Case_A"], sample["Case_B"], sample["relation"]))
        return res

    def _get_paired_samples(self, ca, cb, rel):
        rel = list(map(tuple, rel))
        pos = []
        for r in rel:
            ia, ib = r
            pos.append({'texta': ca[ia], 'textb': cb[ib], 'label': 1})

        neg = []
        while len(neg) < int(1.5 * len(pos)):
            ia = random.randint(0, len(ca)-1)
            ib = random.randint(0, len(cb)-1)
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
        a = []
        b = []
        ab = []
        ba = []
        label = []
        for sample in batch:
            ab.append(sample['texta'] + sep_token + sample['textb'])
            ba.append(sample['textb'] + sep_token + sample['texta'])
            label.append(sample["label"])
        ab = self.encode_batch(ab)
        ba = self.encode_batch(ba)
        return {"input_ab": ab, "input_ba": ba, "labels": torch.tensor(label)}


class SentenceClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.hparams.update(cfg.train_args)
        self.bert = AutoModel.from_pretrained(cfg.pretrain_model_name_or_path)
        self.classifer = torch.nn.Sequential(nn.Dropout(0.2),
                                             nn.Linear(self.bert.config.hidden_size,
                                                       int(self.bert.config.hidden_size / 2)),
                                             nn.GELU(),
                                             nn.Linear(int(self.bert.config.hidden_size / 2), 2))

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, *args, **kwargs):
        ouput = self.bert(**kwargs)
        last = ouput.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
        sent_emb = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
        logits = self.classifer(sent_emb)
        return sent_emb, logits

    def training_step(self, batch, batch_idx):
        input = batch.get("input")
        _, logits = self(**input)
        label = batch.get("label")
        loss = F.cross_entropy(logits, label)
        return loss

    def training_epoch_end(self, outputs):
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        input = batch.get("input")
        _, logits = self(**input)
        label = batch.get("label")
        self.valid_acc.update(logits, label)

    def validation_epoch_end(self, outputs):
        self.log('valid_acc_epoch', self.valid_acc.compute())
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


class SentencePairClassifier(pl.LightningModule):
    pass


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


def train_sentence_classifier(cfg):
    pl.seed_everything(cfg.seed)

    model = SentenceClassifier(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrain_model_name_or_path)

    train_dl, val_dl = build_dataloader(cfg, tokenizer)
    model.set_dataloader(train_dl, val_dl)

    train_args = cfg.train_args
    output_dir = cfg.output_dir

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="{valid_acc_epoch_:02d}",
        save_weights_only=True,
        save_on_train_epoch_end=True,
    )

    logger = TensorBoardLogger(f"{output_dir}/logs")

    print(train_args)
    
    
    trainer = pl.Trainer(
        accelerator=train_args.accelerator,
        max_epochs=train_args.max_epochs,
        devices=train_args.devices,
        gradient_clip_val = train_args.gradient_clip_val,
        strategy = train_args.strategy,
        val_check_interval = train_args.val_check_interval,
        
        callbacks=[
            checkpoint,
        ],
        logger=logger
    )

    trainer.fit(model)


@hydra.main(config_path="./", config_name="config", version_base="1.2")
def main(cfg):
    train_sentence_classifier(cfg.sentence_classifier)


if __name__ == "__main__":
    main()
