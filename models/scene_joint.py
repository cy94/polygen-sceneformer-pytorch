import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from warmup_scheduler import GradualWarmupScheduler

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
            cfg["model"]["cat"]["start_token"] + 1,
            cfg["model"]["emb_dim"],
            padding_idx=cfg["model"]["cat"]["pad_token"],
        )
        self.pos_emb = nn.Embedding(
            cfg["model"]["max_seq_len"], cfg["model"]["emb_dim"]
        )
        self.x_coor_emb = nn.Embedding(
            cfg["model"]["coor"]["start_token"] + 1,
            cfg["model"]["emb_dim"],
            padding_idx=cfg["model"]["coor"]["pad_token"],
        )
        self.y_coor_emb = nn.Embedding(
            cfg["model"]["coor"]["start_token"] + 1,
            cfg["model"]["emb_dim"],
            padding_idx=cfg["model"]["coor"]["pad_token"],
        )
        self.z_coor_emb = nn.Embedding(
            cfg["model"]["coor"]["start_token"] + 1,
            cfg["model"]["emb_dim"],
            padding_idx=cfg["model"]["coor"]["pad_token"],
        )
        self.orient_emb = nn.Embedding(
            cfg["model"]["orient"]["start_token"] + 1,
            cfg["model"]["emb_dim"],
            padding_idx=cfg["model"]["orient"]["pad_token"],
        )

        d_layer = nn.TransformerEncoderLayer(
            d_model=cfg["model"]["emb_dim"],
            nhead=cfg["model"]["num_heads"],
            dim_feedforward=cfg["model"]["dim_fwd"],
            dropout=cfg["model"]["dropout"],
        )
        self.generator = nn.TransformerEncoder(d_layer, cfg["model"]["num_blocks"])
        # Todo 这里要改一波
        self.output_cat = nn.Linear(
            cfg["model"]["emb_dim"], cfg["model"]["cat"]["start_token"]
        )
        self.output_x = nn.Linear(
            cfg["model"]["emb_dim"], cfg["model"]["coor"]["start_token"]
        )
        self.output_y = nn.Linear(
            cfg["model"]["emb_dim"], cfg["model"]["coor"]["start_token"]
        )
        self.output_z = nn.Linear(
            cfg["model"]["emb_dim"], cfg["model"]["coor"]["start_token"]
        )
        self.output_orient = nn.Linear(
            cfg["model"]["emb_dim"], cfg["model"]["orient"]["start_token"]
        )

        self.decoder_seq_len = cfg["model"]["max_seq_len"]

    def forward(self, cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cat_emb, pos_emb, x_emb, y_emb, z_emb, ori_emb = self.get_embedding(
            cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq
        )  # ,obj_emb

        joint_emb = cat_emb + pos_emb + x_emb + y_emb + z_emb + ori_emb  # + obj_emb

        tgt_padding_mask = self.get_padding_mask(cat_seq)[:, :-1].to(device)
        tgt_mask = self.generate_square_subsequent_mask(cat_seq.shape[1] - 1).to(device)

        tgt = joint_emb.transpose(1, 0)[:-1, :, :]
        out_embs = self.generator(tgt, tgt_mask, tgt_padding_mask)
        out_embs = out_embs.transpose(1, 0)

        out_cat = self.output_cat(out_embs)
        out_x = self.output_x(out_embs)
        out_y = self.output_y(out_embs)
        out_z = self.output_z(out_embs)
        out_orient = self.output_orient(out_embs)

        logprobs_cat = F.log_softmax(out_cat, dim=-1)
        logprobs_x = F.log_softmax(out_x, dim=-1)
        logprobs_y = F.log_softmax(out_y, dim=-1)
        logprobs_z = F.log_softmax(out_z, dim=-1)
        logprobs_orient = F.log_softmax(out_orient, dim=-1)

        return logprobs_cat, logprobs_x, logprobs_y, logprobs_z, logprobs_orient

    def greedy_decode(self, probabilistic=False, nucleus=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generate_seq_len = self.decoder_seq_len - 1

        cat_start_seq = (
            torch.LongTensor(
                [self.cfg["model"]["cat"]["start_token"]] + generate_seq_len * [0]
            )
            .view(1, -1)
            .to(device)
        )
        x_start_seq = (
            torch.LongTensor(
                [self.cfg["model"]["coor"]["start_token"]] + generate_seq_len * [0]
            )
            .view(1, -1)
            .to(device)
        )
        y_start_seq = (
            torch.LongTensor(
                [self.cfg["model"]["coor"]["start_token"]] + generate_seq_len * [0]
            )
            .view(1, -1)
            .to(device)
        )
        z_start_seq = (
            torch.LongTensor(
                [self.cfg["model"]["coor"]["start_token"]] + generate_seq_len * [0]
            )
            .view(1, -1)
            .to(device)
        )
        orient_start_seq = (
            torch.LongTensor(
                [self.cfg["model"]["orient"]["start_token"]] + generate_seq_len * [0]
            )
            .view(1, -1)
            .to(device)
        )

        cat_emb, pos_emb, x_emb, y_emb, z_emb, ori_emb = self.get_embedding(
            cat_start_seq, x_start_seq, y_start_seq, z_start_seq, orient_start_seq
        )
        joint_emb = cat_emb + pos_emb + x_emb + y_emb + z_emb + ori_emb

        tgt = joint_emb.transpose(1, 0)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[0]).to(device)

        cat_gen_seq = [self.cfg["model"]["cat"]["start_token"]]
        x_gen_seq = [self.cfg["model"]["coor"]["start_token"]]
        y_gen_seq = [self.cfg["model"]["coor"]["start_token"]]
        z_gen_seq = [self.cfg["model"]["coor"]["start_token"]]
        orient_gen_seq = [self.cfg["model"]["orient"]["start_token"]]

        for out_ndx in range(generate_seq_len):
            out_embs = self.generator(tgt, tgt_mask)

            logits_cat = self.output_cat(out_embs)[out_ndx][0]
            logits_x = self.output_x(out_embs)[out_ndx][0]
            logits_y = self.output_y(out_embs)[out_ndx][0]
            logits_z = self.output_z(out_embs)[out_ndx][0]
            logits_orient = self.output_orient(out_embs)[out_ndx][0]

            if probabilistic and nucleus:
                logits_cat = sample_top_p(logits_cat)
                logits_x = sample_top_p(logits_x)
                logits_y = sample_top_p(logits_y)
                logits_z = sample_top_p(logits_z)
                logits_orient = sample_top_p(logits_orient)

            probs_cat = F.softmax(logits_cat, dim=-1)
            probs_x = F.softmax(logits_x, dim=-1)
            probs_y = F.softmax(logits_y, dim=-1)
            probs_z = F.softmax(logits_z, dim=-1)
            probs_orient = F.softmax(logits_orient, dim=-1)

            if probabilistic:
                cat_next_token = Categorical(probs=probs_cat).sample()
                x_next_token = Categorical(probs=probs_x).sample()
                y_next_token = Categorical(probs=probs_y).sample()
                z_next_token = Categorical(probs=probs_z).sample()
                orient_next_token = Categorical(probs=probs_orient).sample()

            else:
                _, cat_next_token = torch.max(probs_cat, dim=0)
                _, x_next_token = torch.max(probs_x, dim=0)
                _, y_next_token = torch.max(probs_y, dim=0)
                _, z_next_token = torch.max(probs_z, dim=0)
                _, orient_next_token = torch.max(probs_orient, dim=0)

            cat_gen_seq.append(cat_next_token)
            x_gen_seq.append(x_next_token)
            y_gen_seq.append(y_next_token)
            z_gen_seq.append(z_next_token)
            orient_gen_seq.append(orient_next_token)

            if (
                cat_next_token == self.cfg["model"]["cat"]["stop_token"]
                or x_next_token == self.cfg["model"]["coor"]["stop_token"]
                or y_next_token == self.cfg["model"]["coor"]["stop_token"]
                or z_next_token == self.cfg["model"]["coor"]["stop_token"]
                or orient_next_token == self.cfg["model"]["orient"]["stop_token"]
            ):
                # output seq contains the stop token, not its class index
                break

            curr_cat_seq = cat_gen_seq + (self.decoder_seq_len - len(cat_gen_seq)) * [0]
            curr_cat_seq = torch.LongTensor(curr_cat_seq).view(1, -1).to(device)

            curr_x_seq = x_gen_seq + (self.decoder_seq_len - len(x_gen_seq)) * [0]
            curr_x_seq = torch.LongTensor(curr_x_seq).view(1, -1).to(device)

            curr_y_seq = y_gen_seq + (self.decoder_seq_len - len(y_gen_seq)) * [0]
            curr_y_seq = torch.LongTensor(curr_y_seq).view(1, -1).to(device)

            curr_z_seq = z_gen_seq + (self.decoder_seq_len - len(z_gen_seq)) * [0]
            curr_z_seq = torch.LongTensor(curr_z_seq).view(1, -1).to(device)

            curr_orient_seq = orient_gen_seq + (
                self.decoder_seq_len - len(orient_gen_seq)
            ) * [0]
            curr_orient_seq = torch.LongTensor(curr_orient_seq).view(1, -1).to(device)

            cat_emb, pos_emb, x_emb, y_emb, z_emb, ori_emb = self.get_embedding(
                curr_cat_seq, curr_x_seq, curr_y_seq, curr_z_seq, curr_orient_seq
            )  # , v
            tgt = (cat_emb + pos_emb + x_emb + y_emb + z_emb + ori_emb).transpose(
                1, 0
            )  # + v

        return cat_gen_seq, x_gen_seq, y_gen_seq, z_gen_seq, orient_gen_seq

    def get_embedding(self, cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cat_emb = self.cat_emb(cat_seq)
        seq_len = cat_seq.shape[1]

        x_emb = self.x_coor_emb(x_loc_seq)
        y_emb = self.y_coor_emb(y_loc_seq)
        z_emb = self.z_coor_emb(z_loc_seq)

        ori_emb = self.orient_emb(orient_seq)

        pos_seq = torch.arange(0, seq_len).to(device)
        pos_emb = self.pos_emb(pos_seq)

        return cat_emb, pos_emb, x_emb, y_emb, z_emb, ori_emb

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
        mask[seq == self.cfg["model"]["cat"]["pad_token"]] = 1

        return mask.bool()

    def configure_optimizers(self):
        self.optim = Adam(
            self.parameters(),
            lr=self.cfg["train"]["lr"],
            weight_decay=self.cfg["train"]["l2"],
        )
        self.sched = CosineAnnealingLR(
            self.optim, T_max=self.cfg["train"]["lr_restart"]
        )
        self.warmup = GradualWarmupScheduler(
            self.optim,
            multiplier=1,
            total_epoch=self.cfg["train"]["warmup"],
            after_scheduler=self.sched,
        )

        return [self.optim], [self.sched]

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        optimizer.step()
        self.warmup.step()
        self.sched.step()
        optimizer.zero_grad()

    def general_step(self, batch):
        loss = 0
        cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq = (
            batch["cat_seq"],
            batch["x_loc_seq"],
            batch["y_loc_seq"],
            batch["z_loc_seq"],
            batch["orient_seq"],
        )
        (
            logprobs_cat,
            logprobs_x,
            logprobs_y,
            logprobs_z,
            logprobs_orient,
        ) = self.forward(cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq)

        loss_cat = F.nll_loss(
            logprobs_cat.transpose(1, 2),
            batch["cat_seq"][:, 1:],
            ignore_index=self.cfg["model"]["cat"]["pad_token"],
        )
        loss_x = F.nll_loss(
            logprobs_x.transpose(1, 2),
            batch["x_loc_seq"][:, 1:],
            ignore_index=self.cfg["model"]["coor"]["pad_token"],
        )
        loss_y = F.nll_loss(
            logprobs_y.transpose(1, 2),
            batch["y_loc_seq"][:, 1:],
            ignore_index=self.cfg["model"]["coor"]["pad_token"],
        )
        loss_z = F.nll_loss(
            logprobs_z.transpose(1, 2),
            batch["z_loc_seq"][:, 1:],
            ignore_index=self.cfg["model"]["coor"]["pad_token"],
        )
        loss_orient = F.nll_loss(
            logprobs_orient.transpose(1, 2),
            batch["orient_seq"][:, 1:],
            ignore_index=self.cfg["model"]["orient"]["pad_token"],
        )

        loss = loss_cat + (loss_x + loss_y + loss_z) + loss_orient

        acc_cat, acc_loc = self.eval_model(
            cat_seq,
            x_loc_seq,
            y_loc_seq,
            z_loc_seq,
            orient_seq,
            logprobs_cat,
            logprobs_x,
            logprobs_y,
            logprobs_z,
            logprobs_orient,
        )
        # gt = seq.clone()
        # loss = F.nll_loss(logprobs.transpose(1, 2), gt[:, 1:])
        return loss, acc_cat, acc_loc

    def eval_model(
        self,
        cat_seq,
        x_loc_seq,
        y_loc_seq,
        z_loc_seq,
        orient_seq,
        logprobs_cat,
        logprobs_x,
        logprobs_y,
        logprobs_z,
        logprobs_orient,
    ):
        correct_cat, total_cat = 0, 0
        for ndx, preds in enumerate(logprobs_cat):
            gt = cat_seq[ndx].clone()
            # remove start tokens from gt - we dont want to predict these finally
            gt = gt[1:]
            mask = gt != self.cfg["model"]["cat"]["pad_token"]

            gt_real = gt[mask]
            correct_cat += (preds.argmax(-1) == gt).float()[mask].sum()

            total_cat += gt_real.shape[0]
        acc_cat = correct_cat / total_cat

        correct_x, total_x = 0, 0
        for ndx, preds in enumerate(logprobs_x):
            gt = x_loc_seq[ndx].clone()
            # remove start tokens from gt - we dont want to predict these finally
            gt = gt[1:]
            mask = gt != self.cfg["model"]["coor"]["pad_token"]
            gt_real = gt[mask]
            correct_x += (preds.argmax(-1) == gt).float()[mask].sum()
            total_x += gt_real.shape[0]
        acc_x = correct_x / total_x

        correct_y, total_y = 0, 0
        for ndx, preds in enumerate(logprobs_y):
            gt = y_loc_seq[ndx].clone()
            # remove start tokens from gt - we dont want to predict these finally
            gt = gt[1:]
            mask = gt != self.cfg["model"]["coor"]["pad_token"]
            gt_real = gt[mask]
            correct_y += (preds.argmax(-1) == gt).float()[mask].sum()
            total_y += gt_real.shape[0]
        acc_y = correct_y / total_y

        correct_z, total_z = 0, 0
        for ndx, preds in enumerate(logprobs_z):
            gt = z_loc_seq[ndx].clone()
            # remove start tokens from gt - we dont want to predict these finally
            gt = gt[1:]
            mask = gt != self.cfg["model"]["coor"]["pad_token"]
            gt_real = gt[mask]
            correct_z += (preds.argmax(-1) == gt).float()[mask].sum()
            total_z += gt_real.shape[0]
        acc_z = correct_z / total_z

        acc_loc = (acc_x + acc_y + acc_z) / 3

        return acc_cat, acc_loc

    def training_step(self, batch, batch_idx):
        loss, acc_cat, acc_loc = self.general_step(batch)
        lr = get_lr(self.optim)
        log = {
            "loss": {"train_loss": loss},
            "acc": {"train_acc_cat": acc_cat, "train_acc_loc": acc_loc},
            "lr": lr,
        }
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        loss, acc_cat, acc_loc = self.general_step(batch)
        return {"val_loss": loss, "acc": {"acc_cat": acc_cat, "acc_loc": acc_loc}}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc_cat = torch.stack([x["acc"]["acc_cat"] for x in outputs]).mean()
        avg_acc_loc = torch.stack([x["acc"]["acc_loc"] for x in outputs]).mean()

        log = {
            "loss": {"val": avg_loss},
            "acc_cat": {"val": avg_acc_cat},
            "acc_loc": {"val": avg_acc_loc},
        }

        return {"val_loss": avg_loss, "log": log}
