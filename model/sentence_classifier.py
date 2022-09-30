from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import jsonlines
import random
import torch
import hydra


class DataReader(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self._read_data()

        self.single_data = self._gen_single_data()
        self.paired_data = self._gen_paired_data()

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
        encoded['labels'] = torch.tensor(label)
        return encoded


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    cfg = cfg.sentence_classifier

    pretrain_model_name_or_path = cfg.pretrain_model_name_or_path
    data_dir = cfg.data_dir
    output_dir = cfg.output_dir

    model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_name_or_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name_or_path)

    train_single_data = DataReader(f"{data_dir}/train.jsonl").single_data
    test_single_data = DataReader(f"{data_dir}/val.jsonl").single_data

    train_single_dataset = SingleDataset(train_single_data, tokenizer)
    test_single_dataset = SingleDataset(test_single_data, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        weight_decay=0.01,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_single_dataset,
        eval_dataset=test_single_dataset,
        tokenizer=tokenizer,
        data_collator=train_single_dataset.collate_fn,
    )

    trainer.train()


if __name__ == "__main__":
    main()
