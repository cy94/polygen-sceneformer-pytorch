import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

# from warmup_scheduler import GradualWarmupScheduler

from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

import numpy as np

from models.misc import get_lr


def sample_top_p(logits, top_p=0.9, filter_value=-float("Inf")):
    """
    logits: single array of logits (N,)
    top_p: top cumulative probability to select

    return: new array of logits, same shape as logits (N,)
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    # dont modify the original logits
    sampled = logits.clone()
    sampled[indices_to_remove] = filter_value

    return sampled


class scene_transformer(LightningModule):
    def __init__(self, cfg):

        super(scene_transformer, self).__init__()
        self.hparams = cfg
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.cat_emb = nn.Embedding(
            cfg["model"]["start_token"] + 1, cfg["model"]["emb_dim"]
        )
        self.pos_emb = nn.Embedding(
            cfg["model"]["max_seq_len"], cfg["model"]["emb_dim"]
        )
        # self.obj_emb = nn.Embedding(cfg['model']['max_obj_num'], cfg['model']['emb_dim'])
        # self.token_class_emb = nn.Embedding()
        d_layer = nn.TransformerEncoderLayer(
            d_model=cfg["model"]["emb_dim"],
            nhead=cfg["model"]["num_heads"],
            dim_feedforward=cfg["model"]["dim_fwd"],
            dropout=cfg["model"]["dropout"],
        )
        self.generator = nn.TransformerEncoder(d_layer, cfg["model"]["num_blocks"])
        self.output = nn.Linear(cfg["model"]["emb_dim"], cfg["model"]["start_token"])
        self.decoder_seq_len = cfg["model"]["max_seq_len"]
        self.start_token = cfg["model"]["start_token"]
        self.stop_token = cfg["model"]["stop_token"]
        self.pad_token = cfg["model"]["pad_token"]

    def forward(self, seq):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')

        cat_emb, pos_emb = self.get_embedding(seq)  # , obj_emb

        joint_emb = cat_emb + pos_emb  # + obj_emb

        tgt_padding_mask = self.get_padding_mask(seq)[:, :-1].to(device)
        tgt_mask = self.generate_square_subsequent_mask(seq.shape[1] - 1).to(device)

        tgt = joint_emb.transpose(1, 0)[:-1, :, :]
        out_embs = self.generator(tgt, tgt_mask, tgt_padding_mask)
        out_embs = out_embs.transpose(1, 0)
        seq_logprobs = F.log_softmax(self.output(out_embs), dim=-1)

        return seq_logprobs

    def greedy_decode(self, probabilistic=False, nucleus=True):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generate_seq_len = self.decoder_seq_len - 1

        start_seq = (
            torch.LongTensor([self.start_token] + generate_seq_len * [0])
            .view(1, -1)
            .to(device)
        )
        cat_emb, pos_emb = self.get_embedding(start_seq)  # , obj_emb
        joint_emb = cat_emb + pos_emb  # + obj_emb

        tgt = joint_emb.transpose(1, 0)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[0]).to(device)

        gen_seq = [self.start_token]

        for out_ndx in range(generate_seq_len):
            out_embs = self.generator(tgt, tgt_mask)
            logits = self.output(out_embs)[out_ndx][0]

            if probabilistic and nucleus:
                logits = sample_top_p(logits)

            probs = F.softmax(logits, dim=-1)
            if probabilistic:
                next_token = Categorical(probs=probs).sample()
            else:
                _, next_token = torch.max(probs, dim=0)
            gen_seq.append(next_token)

            if next_token == self.stop_token:  # - 1:
                # output seq contains the stop token, not its class index
                # gen_seq[-1] += 1
                break

            curr_seq = gen_seq + (self.decoder_seq_len - len(gen_seq)) * [0]
            curr_seq = torch.LongTensor(curr_seq).view(1, -1).to(device)

            c, p = self.get_embedding(curr_seq)  # , v
            tgt = (c + p).transpose(1, 0)  # + v

        return torch.LongTensor(gen_seq)

    def get_embedding(self, seq):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cat_emb = self.cat_emb(seq)
        seq_len = seq.shape[1]

        pos_seq = torch.arange(0, seq_len).to(device)
        pos_emb = self.pos_emb(pos_seq)

        # ndx = np.arange(seq_len).reshape((1, -1))
        # ndx[0, 1:] = (ndx[0, 1:] - 1) // 9 + 1
        # obj_seq = torch.LongTensor(ndx)#.to(device)
        # obj_emb = self.obj_emb(obj_seq)

        return cat_emb, pos_emb  # , obj_emb

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def get_padding_mask(self, seq):
        mask = torch.ByteTensor(np.zeros(seq.shape, dtype=np.uint8))
        mask[seq == self.cfg["model"]["pad_token"]] = 1

        return mask.bool()

    # def greedy_decode(self, probabilistic=False):
    #
    #     return seq

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.cfg["train"]["lr"])
        return optim

    def general_step(self, batch):
        loss = 0
        seq = batch["seq"]
        logprobs = self.forward(seq)
        for ndx, preds in enumerate(logprobs):

            gt = seq[ndx].clone()
            gt = gt[1:]
            # gt[gt == self.cfg['model']['pad_token']] -= 1
            loss += F.nll_loss(preds, gt, ignore_index=self.pad_token)
        loss /= seq.shape[0]
        # gt = seq.clone()
        # loss = F.nll_loss(logprobs.transpose(1, 2), gt[:, 1:])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        log = {"loss": {"train_loss": loss}}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        log = {"loss": {"val": avg_loss}}

        return {"val_loss": avg_loss, "log": log}
