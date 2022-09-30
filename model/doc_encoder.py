from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch

import hydra


class SentEncoder(nn.Module):
    def __init__(self, cfg):
        super(SentEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(cfg.pretrain_model_name_or_path)
        self.pooling_type = cfg.pooling_type

    def forward(self, *input, **kwargs):

        out = self.encoder(*input, **kwargs, output_hidden_states=True)

        if self.pooling_type == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling_type == 'pooler':
            return out.pooler_output            # [batch, 768]

        if self.pooling_type == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)


class DocAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=cfg.hidden_size,
                                               num_heads=cfg.num_heads,
                                               dropout=cfg.dropout,
                                               bias=False,
                                               batch_first=True)

        self.ln = nn.LayerNorm(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        residual = x
        x = self.attention(x, x, x, key_padding_mask=mask)[0]
        x = self.dropout(x)
        x += residual
        x = self.ln(x)
        return x


class DocTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation = nn.GELU()
        self.blocks = nn.ModuleList([DocAttentionBlock(cfg) for _ in range(cfg.num_layers)])

    def forward(self, x, mask):
        residual = x
        for block in self.blocks:
            x = block(x, mask)
            x = self.activation(x)
        x += residual
        return x


class DocEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sent_encoder = SentEncoder(cfg.sent_encoder)
        self.doc_encoder = DocTransformer(cfg)

        self.doc_cls_embedding = nn.Parameter(torch.randn(cfg.hidden_size))
        self.doc_pad_embedding = nn.Parameter(torch.randn(cfg.hidden_size))

    def forward(self, docs):
        # batch_size, max_doc_seq_len, hidden_size
        doc_seqs, doc_mask = self._get_doc_embeds(docs)
        doc_seqs = self.doc_encoder(doc_seqs, doc_mask)
        return doc_seqs, doc_mask

    def _add_cls_pad(self, tensor, max_seq_len):
        cls_emb = self.doc_cls_embedding
        pad_emb = self.doc_pad_embedding
        padded_seq = [cls_emb]
        padded_seq.append(tensor)
        padded_seq.extend([pad_emb] * (max_seq_len - tensor.shape[0]))
        padded_seq = torch.vstack(padded_seq)
        return padded_seq

    def _get_mask(self, doc_lens):
        max_seq_len = max(doc_lens)
        mask = torch.ones(len(doc_lens), max_seq_len + 1)
        for i, l in enumerate(doc_lens):
            mask[i, :l + 1] = 0
        return mask.bool()

    def _pad_doc_seq_longest(self, doc_seqs, doc_lens):
        max_seq_len = max(doc_lens)
        for i in range(len(doc_seqs)):
            padded_seq = self._add_cls_pad(doc_seqs[i], max_seq_len)
            doc_seqs[i] = padded_seq.unsqueeze(0)
        return torch.vstack(doc_seqs)

    def _get_doc_embeds(self, docs):
        doc_seqs = []
        doc_lens = []
        for doc in docs:
            sents_embed = self.sent_encoder(**doc)
            doc_lens.append(sents_embed.shape[0])
            doc_seqs.append(sents_embed)

        doc_seqs = self._pad_doc_seq_longest(doc_seqs, doc_lens)
        doc_mask = self._get_mask(doc_lens)
        return doc_seqs, doc_mask


@hydra.main(config_path='./', config_name='config', version_base=None)
def test_forward(cfg):

    def encode_doc(sents, tokenizer):
        encoded = tokenizer(sents, padding="longest", truncation=True, return_tensors='pt', max_length=512)
        return encoded

    model = DocEncoder(cfg.doc_encoder)
    doc = [r"公诉机关指控,2018年8月底,被告人XXX付给被告人朱德印人民币1000元费用,后XXX印制作了XXX的伪造身份证,并安排学生XXX另案处理于2018年10月27日、28日在城阳五中考点代替XXX参加2018年成人高等学校招生全国统一考试,后被查获。",
           r"被告人XXX于2018年11月9日主动至投案;被告人XXX于2018年10月29日主动至投案。",
           r"  经审理查明的事实与公诉机关的指控一致。",
           r"另查明,根据黑龙江省哈尔滨市道外区司法局调查评估,被告人XXX适用社区矫正,该已缴纳罚金人民币五千元;根据山东省青岛市黄岛区司法局调查评估,被告人XXX适用社区矫正,该已缴纳罚金人民币五千元。",
           r"经审理查明的事实与公诉机关的指控一致。",
           r"另查明,根据黑龙江省哈尔滨市道外区司法局调查评估,被告人XXX适用社区矫正,该已缴纳罚金人民币五千元;根据山东省青岛市黄岛区司法局调查评估,被告人XXX适用社区矫正,该已缴纳罚金人民币五千元。",
           r"本院认为,被告人XXX让他人代替自己参加法律规定的国家考试,其行为构成代替考试罪,依法应予惩处;被告人XXX印在法律规定的国家考试中,策划、安排他人进行作弊,其行为构成组织考试作弊罪,依法应予惩处。",
           r"公诉机关指控被告人犯罪的事实清楚,证据确实、充分,罪名成立,量刑建议适当,本院予以采纳。",
           r"二被告人系自首,且自愿认罪认罚,依法可从轻处罚;二人均主动缴纳罚金,可酌情从轻处罚。",
           r"辩护人所提与上述情节相关的辩护意见,本院予以采纳,其他辩护意见酌情予以考虑"]

    tokenizer = AutoTokenizer.from_pretrained("cyclone/simcse-chinese-roberta-wwm-ext")
    docs = [encode_doc(doc[:i + 2], tokenizer) for i in range(4)]
    doc_seqs, doc_mask = model(docs)
    print(doc_seqs.shape)
    print(doc_mask.shape)


if __name__ == "__main__":
    test_forward()
