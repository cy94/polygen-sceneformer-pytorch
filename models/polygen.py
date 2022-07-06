import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from warmup_scheduler import GradualWarmupScheduler

from pytorch_lightning.core.lightning import LightningModule

import numpy as np

from models.misc import get_lr
from models.transformer import CustomTransformerEncoder, CustomTransformerEncoderLayer


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


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class PolyGen(LightningModule):
    """
    The full model from the paper
    PolyGen: An Autoregressive Generative Model of 3D Meshes
    https://arxiv.org/pdf/2002.10880.pdf
    """

    def __init__(self, vtx_cfg, face_cfg, train_cfg=None):
        """
        vtx_cfg: configuration for the vertex model
        """
        super(PolyGen, self).__init__()

        self.vtx_stop_token = vtx_cfg["vertex_stop_token"]
        self.new_face_token = face_cfg["new_face_token"]

        self.vertex_model = VertexModel(vtx_cfg)
        self.face_model = FaceModel(face_cfg, vtx_cfg)

        self.vtx_cfg = vtx_cfg
        self.face_cfg = face_cfg
        self.train_cfg = train_cfg

    def forward(self, vertices, vertices_raw, faces, face_pos, in_face_pos):
        """
        Forward pass in vertex model and face model, independently
        """
        vtx_logprobs = self.vertex_model(vertices)
        face_logprobs = self.face_model(
            face_pos, in_face_pos, faces, vertices, vertices_raw
        )

        return vtx_logprobs, face_logprobs

    def greedy_decode(self, probabilistic=False):
        vertices, faces = None, None

        vtx_seq = self.vertex_model.greedy_decode(probabilistic=probabilistic)

        # check if the vertices are valid
        if self.vtx_stop_token in vtx_seq and (len(vtx_seq) - 2) % 3 == 0:
            # remove start and stop tokens
            vertices = vtx_seq[1:-1].numpy()
            # convert to (V, 3) array and change zyx to xyz
            vertices = vertices.reshape((-1, 3))[:, ::-1]
            # decode faces
            face_seq = self.face_model.greedy_decode(
                vertices, probabilistic=probabilistic
            ).numpy()
            # check if face seq is valid
            if len(face_seq) > 0:
                faces = self.face_model.faceseq_to_raw_faces(face_seq)

        return vertices, faces


class VertexModel(LightningModule):
    """
    The vertex generator model
    """

    def __init__(self, cfg, train_cfg=None, num_classes=0):
        """
        cfg: the vertex model cfg section
        train_cfg: training cfg section
        num_classes: 0 if unconditional model, else number of classes for
                    class conditional model
        """
        super(VertexModel, self).__init__()

        self.max_vertices = cfg["max_vertices"]
        self.pad_token = cfg["vertex_pad_token"]
        self.start_token = cfg["vertex_start_token"]
        self.stop_token = cfg["vertex_stop_token"]
        self.input_seq_len = cfg["max_seq_len"]

        self.cfg = cfg
        self.train_cfg = train_cfg

        self.class_conditional = bool(num_classes)

        emb_dim = cfg["emb_dim"]
        coord_min, coord_max = cfg["coordinate_range"]
        num_coord_vals = coord_max - coord_min + 1

        # embed x, y, z
        if cfg["emb_type"] == "continuous":
            # x, y, or z
            self.coordinate_emb = nn.Embedding(3, emb_dim)
            # position of the vertex in the sequence
            self.pos_emb = nn.Embedding(cfg["max_vertices"], emb_dim)
            # +2 for start and stop tokens
            self.value_emb = nn.Embedding(num_coord_vals + 3, emb_dim)
        elif emb_type == "discrete":
            raise NotImplementedError

        if self.class_conditional:
            self.class_emb = nn.Embedding(num_classes, emb_dim)

        d_layer = CustomTransformerEncoderLayer(
            d_model=emb_dim,
            nhead=cfg["num_heads"],
            dim_feedforward=cfg["dim_fwd"],
            dropout=cfg["dropout"],
        )
        self.generator = CustomTransformerEncoder(d_layer, cfg["num_blocks"])

        # predict coordinate value or stop token
        self.output = nn.Linear(emb_dim, num_coord_vals + 1)

        self.save_hyperparameters()

    def configure_optimizers(self):
        self.optim = Adam(
            self.parameters(),
            lr=self.train_cfg["vtx"]["lr"],
            weight_decay=self.train_cfg["vtx"]["l2"],
        )
        self.sched = CosineAnnealingLR(
            self.optim, T_max=self.train_cfg["vtx"]["lr_restart"]
        )
        self.warmup = GradualWarmupScheduler(
            self.optim,
            multiplier=1,
            total_epoch=self.train_cfg["vtx"]["warmup"],
            after_scheduler=self.sched,
        )

        return [self.optim], [self.warmup]

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        optimizer.step()
        self.warmup.step()
        optimizer.zero_grad()

    def validation_step(self, batch, batch_idx):
        bits_per_vertex, loss, acc = self._eval_common(batch)

        return {"val_loss": loss, "acc": acc, "bits_per_vertex": bits_per_vertex}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()
        avg_bpv = torch.stack([x["bits_per_vertex"] for x in outputs]).mean()

        log = {
            "vtxloss": {"val": avg_loss},
            "vtxacc": {"val": avg_acc},
            "bits_per_vertex": {"val": avg_bpv},
        }

        return {"val_loss": avg_loss, "log": log}

    def training_step(self, batch, batch_idx):
        bits_per_vertex, loss, acc = self._eval_common(batch)

        lr = get_lr(self.optim)
        log = {
            "vtxloss": {"train": loss},
            "vtxacc": {"train": acc},
            "bits_per_vertex": {"train": bits_per_vertex},
            "lr": lr,
        }

        return {"loss": loss, "log": log}

    def _eval_common(self, batch):
        vertices_raw, vertices = batch["vertices_raw"], batch["vertices"]
        classes = batch["class_ndx"] if self.class_conditional else None

        # total number of vertices in this batch
        num_vertices = sum([len(v) for v in vertices_raw])

        # forward pass
        logprobs = self(vertices, classes=classes)

        total_loss, loss_per_token, acc = self.eval_vtx_model(vertices, logprobs)
        # no need to divide by batch size, loss already divided by the number of tokens

        # bits per vertex loss
        # change from base e (natural log) to base 2
        bits_per_vertex = total_loss * np.log(2) / num_vertices

        return bits_per_vertex, loss_per_token, acc

    def eval_vtx_model(self, vtx_gt, vtx_logprobs):
        """
        TODO: eval over the whole GT and preds together, dont loop
        """
        loss = 0
        correct_tokens, total_tokens = 0, 0

        for ndx, preds in enumerate(vtx_logprobs):
            gt = vtx_gt[ndx].clone()
            # change the stop token into its class index (subtract 1)
            gt[gt == self.cfg["vertex_stop_token"]] -= 1
            # remove start tokens from gt - we dont want to predict these finally
            gt = gt[1:]

            # find the locations that are not padded
            non_padded_ndx = gt != self.pad_token
            # and the number of these locations
            non_padded_len = non_padded_ndx.sum()

            # dont compute loss at the positions where the GT has pad tokens
            # use ignore_index
            # add loss over each token
            loss += F.nll_loss(preds, gt, ignore_index=self.pad_token, reduction="sum")

            # dont compute accuracy at the padded locations
            correct_tokens += (preds.argmax(-1) == gt)[non_padded_ndx].float().sum()
            total_tokens += non_padded_len
        acc = correct_tokens / total_tokens
        # divide total token loss by number of tokens
        loss_per_token = loss / total_tokens

        return loss, loss_per_token, acc

    def get_padding_mask(self, vertices):
        """
        vertices: (batch size, seq len, emb dim) array

        create a mask which is True for padded tokens, False for the rest
        """
        mask = torch.ByteTensor(np.zeros(vertices.shape, dtype=np.uint8))
        mask[vertices == self.pad_token] = 1

        return mask.bool()

    def get_embedding(self, vertices):
        """
        vertices: (batch size, seq len) array

        return: coord, pos, value embeddings
        each is of size (batch size, seq len, emb dim)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        value_emb = self.value_emb(vertices)

        batch_size, seq_len = vertices.shape
        # coordinate emb - x, y, z
        # xx, 0, 1, 2, 0, 1, 2 ...
        ndx = np.arange(seq_len).reshape((1, -1))
        ndx_ref = np.arange(seq_len).reshape((1, -1))
        ndx[ndx_ref % 3 == 1] = 0
        ndx[ndx_ref % 3 == 2] = 1
        ndx[ndx_ref % 3 == 0] = 2
        ndx = torch.LongTensor(ndx).to(device)
        coord_emb = self.coordinate_emb(ndx).repeat(batch_size, 1, 1)

        # position emb: xx, 0, 0, 0, 1, 1, 1, 2, 2, 2...
        ndx = np.arange(seq_len).reshape((1, -1))
        ndx[0, 1:] = (ndx[0, 1:] - 1) // 3 + 1

        if self.training and self.stop_token in vertices:
            # during training, there is a stop token
            # all tokens after the stop token - convert them to max_vertices-1
            ndx_stop = (vertices == self.stop_token).nonzero()[0, 1]
            ndx[:, ndx_stop + 1 :] = self.max_vertices - 1
        elif not self.training:
            # during inference, there is no stop token
            # everything that is greater than max_vertices-1 -> set to max_vertices-1
            ndx[ndx > (self.max_vertices - 1)] = self.max_vertices - 1
        ndx = torch.LongTensor(ndx).to(device)
        pos_emb = self.pos_emb(ndx).repeat(batch_size, 1, 1)

        return coord_emb, pos_emb, value_emb

    def vertexseq_to_raw_vertices(self, vtx_seq):
        vertices = None
        if self.stop_token in vtx_seq and (len(vtx_seq) - 2) % 3 == 0:
            # remove start and stop tokens
            vertices = vtx_seq[1:-1].numpy()
            # convert to (V, 3) array and change zyx to xyz
            vertices = vertices.reshape((-1, 3))[:, ::-1]
        return vertices

    def greedy_decode(self, probabilistic=False, nucleus=None, class_ndx=None):
        """
        Generate an output sequence by starting with only a start token
        probabilistic: probabilistic decoding?
        nucleus (float/None): the top-p probability to use in nucleus sampling
        class_ndx (int/None): the index of the class to sample from 
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        out_seq_len = self.input_seq_len - 1

        start_seq = (
            torch.LongTensor([self.start_token] + out_seq_len * [0])
            .view(1, -1)
            .to(device)
        )
        coord_emb, pos_emb, value_emb = self.get_embedding(start_seq)
        joint_emb = coord_emb + pos_emb + value_emb  # N,L,E

        tgt = joint_emb.transpose(1, 0)  # L,N,E,

        tgt_mask = generate_square_subsequent_mask(len(tgt)).to(device)

        gen_seq = [self.start_token]
        # optionally create a class condition vector
        # need to do this only once
        cond_vec = (
            None
            if class_ndx is None
            else self.class_emb(torch.LongTensor([class_ndx]).to(device))
        )

        for out_ndx in range(out_seq_len):

            out_embs = self.generator(tgt, tgt_mask, cond_vec=cond_vec)
            logits = self.output(out_embs)[out_ndx][0]

            if probabilistic and nucleus:
                # pick the top scores that add up to 'nucleus'
                logits = sample_top_p(logits, top_p=nucleus)

            probs = F.softmax(logits, dim=-1)

            if probabilistic:
                next_token = Categorical(probs=probs).sample()
            else:
                _, next_token = torch.max(probs, dim=0)

            gen_seq.append(next_token)

            # the class index for the stop token is token-1
            if next_token == self.stop_token - 1:
                # output seq contains the stop token, not its class index
                gen_seq[-1] += 1
                break

            curr_seq = gen_seq + (self.input_seq_len - len(gen_seq)) * [0]
            curr_seq = torch.LongTensor(curr_seq).view(1, -1).to(device)

            c, p, v = self.get_embedding(curr_seq)
            tgt = (c + p + v).transpose(1, 0)

        return torch.LongTensor(gen_seq)

    def forward(self, vertices_gt, classes=None):
        """
        vertices: (batch_size, seq_len) array
        return: (out_seq_len, batch_size, n_classes)
        """
        # make a copy because we will modify this sequence during training
        vertices = vertices_gt.clone()
        # use "vertices" from here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = vertices.shape[1]
        out_seq_len = seq_len - 1

        tgt_padding_mask = self.get_padding_mask(vertices)[:, :-1].to(device)
        vertices[vertices == self.pad_token] = 0
        coord_emb, pos_emb, value_emb = self.get_embedding(vertices)
        joint_emb = coord_emb + pos_emb + value_emb
        tgt_mask = generate_square_subsequent_mask(seq_len - 1).to(device)
        tgt = joint_emb.transpose(1, 0)[:-1, :, :]

        # class condition vectors?
        cond_vec = None if not self.class_conditional else self.class_emb(classes)

        out_embs = self.generator(tgt, tgt_mask, tgt_padding_mask, cond_vec=cond_vec)
        out_embs = out_embs.transpose(1, 0)

        out = F.log_softmax(self.output(out_embs), dim=-1)
        return out


class FaceModel(LightningModule):
    """
    The face generator model
    """

    def __init__(self, cfg, vtx_cfg, train_cfg=None, num_classes=0):
        """
        cfg: the face model cfg section
        vtx_cfg: the vertex model cfg section
        train_cfg: training cfg section
        num_classes: 0 if unconditional model, else number of classes for
                    class conditional model
        """
        # pointer network
        # transformer
        super(FaceModel, self).__init__()

        self.cfg = cfg
        self.vtx_cfg = vtx_cfg
        self.train_cfg = train_cfg

        self.class_conditional = bool(num_classes)

        emb_dim = cfg["emb_dim"]
        self.emb_dim = cfg["emb_dim"]

        if self.class_conditional:
            self.class_emb = nn.Embedding(num_classes, emb_dim)

        # makes sure this moves to the GPU
        self.register_buffer("new_face_token", torch.Tensor([cfg["new_face_token"]]))
        self.register_buffer("face_stop_token", torch.Tensor([cfg["face_stop_token"]]))

        self.face_pad_token = cfg["face_pad_token"]
        self.max_face_seq_len = cfg["max_face_seq_len"]

        self.register_buffer(
            "vertex_stop_token", torch.Tensor([vtx_cfg["vertex_stop_token"]])
        )
        self.vertex_pad_token = vtx_cfg["vertex_pad_token"]
        self.max_vertices = vtx_cfg["max_vertices"]

        self.max_face_num = cfg["max_face_num"]
        self.max_vtx_per_face = cfg["max_vtx_per_face"]

        # assume the face_stop_token has the largest index
        self.pos_emb = nn.Embedding(cfg["max_face_num"], emb_dim)
        self.in_face_emb = nn.Embedding(cfg["max_vtx_per_face"], emb_dim)

        coord_min, coord_max = vtx_cfg["coordinate_range"]
        num_coord_vals = coord_max - coord_min + 1
        self.x_emb = nn.Embedding(num_coord_vals + 2, emb_dim)
        self.y_emb = nn.Embedding(num_coord_vals + 2, emb_dim)
        self.z_emb = nn.Embedding(num_coord_vals + 2, emb_dim)
        self.num_coord_vals = num_coord_vals

        d_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=cfg["num_heads"],
            dim_feedforward=cfg["dim_fwd"],
            dropout=cfg["dropout"],
        )
        self.vtx_encoder = nn.TransformerEncoder(d_layer, cfg["num_blocks"])

        d_layer_face = CustomTransformerEncoderLayer(
            d_model=emb_dim,
            nhead=cfg["num_heads_face"],
            dim_feedforward=cfg["dim_fwd_face"],
            dropout=cfg["dropout"],
        )

        self.face_encoder = CustomTransformerEncoder(
            d_layer_face, cfg["num_blocks_face"]
        )

        self.save_hyperparameters()

    def configure_optimizers(self):
        self.optim = Adam(
            self.parameters(),
            lr=self.train_cfg["face"]["lr"],
            weight_decay=self.train_cfg["face"]["l2"],
        )
        self.sched = CosineAnnealingLR(
            self.optim, T_max=self.train_cfg["face"]["lr_restart"]
        )
        self.warmup = GradualWarmupScheduler(
            self.optim,
            multiplier=1,
            total_epoch=self.train_cfg["face"]["warmup"],
            after_scheduler=self.sched,
        )

        return [self.optim], [self.warmup]

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        optimizer.step()
        self.warmup.step()
        optimizer.zero_grad()

    def validation_step(self, batch, batch_idx):
        bits_per_vertex, loss, acc = self._eval_common(batch)

        return {"val_loss": loss, "acc": acc, "bits_per_vertex": bits_per_vertex}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()
        avg_bpv = torch.stack([x["bits_per_vertex"] for x in outputs]).mean()

        log = {
            "faceloss": {"val": avg_loss},
            "faceacc": {"val": avg_acc},
            "bits_per_vertex": {"val": avg_bpv},
        }

        return {"val_loss": avg_loss, "log": log}

    def _eval_common(self, batch):
        """
        eval on a single batch in train or val mode
        """
        vertices_raw, faces_raw = batch["vertices_raw"], batch["faces_raw"]
        vertices, faces = batch["vertices"], batch["faces"]
        face_pos, in_face_pos = batch["face_pos"], batch["in_face_pos"]
        classes = batch["class_ndx"] if self.class_conditional else None

        # total number of vertices in this batch
        num_vertices = sum([len(v) for v in vertices_raw])

        # forward pass
        logprobs = self(
            face_pos, in_face_pos, faces, vertices, vertices_raw, classes=classes
        )

        total_loss, loss_per_token, acc = self.eval_face_model(faces, logprobs)
        # no need to divide by batch size, loss already divided by the number of tokens

        # bits per vertex loss
        # change from base e (natural log) to base 2
        bits_per_vertex = total_loss * np.log(2) / num_vertices

        return bits_per_vertex, loss_per_token, acc

    def training_step(self, batch, batch_idx):
        bits_per_vertex, loss, acc = self._eval_common(batch)

        lr = get_lr(self.optim)
        log = {
            "faceloss": {"train": loss},
            "faceacc": {"train": acc},
            "bits_per_vertex": {"train": bits_per_vertex},
            "lr": lr,
        }

        return {"loss": loss, "log": log}

    def eval_face_model(self, faces_gt, faces_logprobs):
        loss = 0
        correct_tokens, total_tokens = 0, 0

        for ndx, preds in enumerate(faces_logprobs):
            gt = faces_gt[ndx].clone()
            # remove start tokens from gt - we dont want to predict these finally
            gt = gt[1:]
            # -1 score is the stop face token
            gt[gt == self.cfg["face_stop_token"]] = preds.shape[1] - 1
            # -2 score is the new face token
            gt[gt == self.cfg["new_face_token"]] = preds.shape[1] - 2

            # find the locations that are not padded
            non_padded_ndx = gt != self.face_pad_token
            # and the number of these locations
            non_padded_len = non_padded_ndx.sum()

            # dont compute loss at the positions where the GT has pad tokens
            # add loss over all tokens
            loss += F.nll_loss(
                preds, gt, ignore_index=self.cfg["face_pad_token"], reduction="sum"
            )
            correct_tokens += (preds.argmax(-1) == gt)[non_padded_ndx].float().sum()
            total_tokens += non_padded_len

        acc = correct_tokens / total_tokens
        loss_per_token = loss / total_tokens
        return loss, loss_per_token, acc

    def encode_vertices(self, vertices_raw):
        """
        Encode the sequence N S 0 1 2 3 .. max_vertices

        vertices_raw: [(v1, 3), (v2, 3) .. arrays] of vertex coordinates
        
        return: (N, seq_len, emb_dim)
        """
        # get embedding for each coordinate
        x_emb, y_emb, z_emb = self.get_coordinate_emb(vertices_raw)
        emb = x_emb + y_emb + z_emb

        # pass through transformer
        vtx_emb = self.vtx_encoder(src=emb)

        return vtx_emb

    def get_embedding(self, vertices_raw, face_pos, in_face_pos, faces):
        """
        Embed a face sequence

        args:

        face_pos: 0 0 0 N 1 1 1 1 N 2 2 N 3
        in_face_pos: 1,2,3,N,1,2,3,4,N
        faces: the indices of vertices in a face, with new and stop tokens
                eg: 1 3 4 2 N 1 4 5 2 S
           
        First embed the vertex indices 0,1,2,3..V and N,S
        Then gather the vtx index embs by indexing into the vertex embedding
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dont embed start, stop tokens
        # the value embedding takes care of this
        # so we can have any embedding in these indices for the face_pos and in_face_pos
        # select index 0 because embedding[0] always exists
        ignore_ndx = (face_pos == self.face_stop_token) | (
            face_pos == self.new_face_token
        )
        # remove start, stop and new tokens
        face_pos_copy = face_pos.clone()
        face_pos_copy[ignore_ndx] = 0
        face_pos_emb = self.pos_emb(face_pos_copy)

        in_face_pos_copy = in_face_pos.clone()
        in_face_pos_copy[ignore_ndx] = 0
        in_face_pos_emb = self.in_face_emb(in_face_pos_copy)

        # encode the vertex coordinates along with N and S tokens
        vtx_emb = self.encode_vertices(vertices_raw)
        faces_copy = faces.clone()
        # -1 embedding is the stop face token
        faces_copy[faces == self.face_stop_token] = self.max_vertices + 1
        # -2 embedding is the new face token
        faces_copy[faces == self.new_face_token] = self.max_vertices
        select_ndx = faces_copy.unsqueeze(2).expand(
            faces_copy.size(0), faces_copy.size(1), vtx_emb.size(2)
        )

        value_emb = torch.gather(vtx_emb, 1, select_ndx)

        # add three embeddings
        assert face_pos_emb.shape == in_face_pos_emb.shape == value_emb.shape
        joint_emb = face_pos_emb + in_face_pos_emb + value_emb

        return joint_emb, vtx_emb

    def get_padding_mask(self, faces):
        """
        create a mask which is True for padded tokens, False for the rest
        """
        mask = torch.ByteTensor(np.zeros(faces.shape, dtype=np.uint8))
        mask[faces == self.face_pad_token] = 1

        return mask.bool()

    def get_coordinate_emb(self, vertices_raw):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # empty tensors to hold embeddings
        empty = torch.empty(
            vertices_raw.shape[0], self.max_vertices + 2, self.emb_dim
        ).to(device)
        x_emb, y_emb, z_emb = empty, empty.clone(), empty.clone()

        new_face_ndx = int(self.num_coord_vals)
        stop_face_ndx = int(self.num_coord_vals + 1)

        for i, vertices_orig in enumerate(vertices_raw):
            # calculate the padding for the vertex sequence
            n_pad = self.max_vertices - vertices_orig.shape[0]
            # conver to Long to allow new and stop tokens
            # make the numpy array contiguous before converting to tensor
            vertices = torch.LongTensor(vertices_orig - np.zeros_like(vertices_orig))
            # pad and create sequences
            # add 0 here, these dont get used anyway
            # pad left, right, top, bottom
            seq1 = F.pad(vertices, (0, 0, 0, n_pad), mode="constant", value=0)
            # add the new face token
            # coords are 0..255
            # new face embedded at index 256
            seq2 = F.pad(seq1, (0, 0, 0, 1), mode="constant", value=new_face_ndx)
            # add the face stop token
            # stop face embedded at index 257
            seq3 = F.pad(seq2, (0, 0, 0, 1), mode="constant", value=stop_face_ndx)
            # unpack into x, y, z
            x, y, z = seq3.T.to(device)

            x_emb[i] = self.x_emb(x)
            y_emb[i] = self.y_emb(y)
            z_emb[i] = self.z_emb(z)

        return x_emb, y_emb, z_emb

    def forward(
        self, face_pos, in_face_pos, faces_gt, vertices, vertices_raw, classes=None
    ):
        """
        face_pos: (batch size, seq_len) indicates the position of a vertex index 
                    in the full face sequence (0 0 0 N 1 1 1 N 2 2 2 )
        in_face_pos: (batch size, seq_len) indicates the position of a vertex index
                within the face (0 1 2 N 0 1 2 3 N 0 1 2 3 4 ..)
        faces: (batchsize, seq_len) indicates the vertex indices in a face
                (0 1 2 N 1 4 5 N 3 5 6 S)
        vertices: (batchsize, seq len) seq indicating vertex coordinates                 
        vertices_raw: [(v1, 3), (v2, 3) .. batchsize] raw vertex coordinates
        classes: class indices for class conditional model
        """
        # make a copy of the GT because we need to modify the sequence
        faces = faces_gt.clone()
        # use "faces" from here

        #  feed joint embedding into transformer decoder
        #  PointerNetwork, output distribution
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seq_len = faces.shape[1]
        # reduce seq len by 1
        tgt_mask = generate_square_subsequent_mask(seq_len - 1).to(device)
        # remove the last token, reduce seqlen by 1
        tgt_padding_mask = self.get_padding_mask(faces)[:, :-1].to(device)

        # remove padding tokens because we have the mask
        faces[faces == self.face_pad_token] = 0
        face_pos[face_pos == self.face_pad_token] = 0
        in_face_pos[in_face_pos == self.face_pad_token] = 0

        vertices_raw = np.array(vertices_raw)
        joint_emb, vtx_emb = self.get_embedding(
            vertices_raw, face_pos, in_face_pos, faces
        )

        # class condition vectors?
        cond_vec = None if not self.class_conditional else self.class_emb(classes)

        # remove the last token - input is shifted wrt the output
        tgt = joint_emb.transpose(1, 0)[:-1, :, :]
        pointer = self.face_encoder(tgt, tgt_mask, tgt_padding_mask, cond_vec=cond_vec)
        # align dimension
        pointer = pointer.transpose(0, 1)
        vtx_emb = vtx_emb.transpose(1, 2).squeeze()
        # get scores
        product = torch.matmul(pointer, vtx_emb)

        # for each sample, keep only V_i scores, V_i=number of vertices in sample i
        out_logprobs = []
        for (ndx, scores) in enumerate(product):
            num_vertices = vertices_raw[ndx].shape[0]
            # only 'V' scores + 2 scores for the new and stop tokens
            scores_reduced = torch.cat(
                [scores[:, :num_vertices], scores[:, -2:]], dim=-1
            )
            out_logprobs.append(F.log_softmax(scores_reduced, dim=-1))

        return out_logprobs

    def faceseq_to_raw_faces(self, face_seq):
        """
        Convert a PolyGen-type face sequence to the raw list of lists 
        """
        # get indices with the stop token
        outer = []
        inner = []
        for elem in face_seq:
            if elem == self.new_face_token or elem == self.face_stop_token:
                if inner:
                    outer.append(inner)
                inner = []
            else:
                inner.append(elem)
        if inner:
            outer.append(inner)
        # ignore the last stop token
        return outer

    def greedy_decode(
        self, vertices_raw, probabilistic=False, nucleus=None, class_ndx=None
    ):
        """
        vertices_raw: (V, 3) sequence, V is the number of vertices in the mesh
        probabilistic: use probabilistic inference
        nucleus (float/None): the top-p probability to use in nucleus sampling
        class_ndx (int/None): the index of the class to sample from 
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # use embeddings only for V vertices
        num_vertices = vertices_raw.shape[0]

        # sequence is 1 less than the max seq len
        # to allow for input shifting wrt the output
        out_seq_len = self.max_face_seq_len - 1

        # the first vertex in the first face has in_face_pos=0
        # rest of the tokens dont matter due to the mask
        in_face_pos = torch.zeros(1, out_seq_len, dtype=torch.long).to(device)
        # the first vertex in the first face has face_pos=0
        # rest of the tokens dont matter due to the mask
        face_pos = torch.zeros(1, out_seq_len, dtype=torch.long).to(device)
        # the first vertex in the first face has the lowest index
        # rest of the tokens dont matter due to the mask
        start_seq = torch.zeros(1, out_seq_len, dtype=torch.long).to(device)
        # embed everything together
        vertices_raw_batch = np.expand_dims(vertices_raw, 0)
        joint_emb, vtx_emb = self.get_embedding(
            vertices_raw_batch, face_pos, in_face_pos, start_seq
        )
        # embed (vertex indices+stop+new) only once
        vtx_emb = vtx_emb.transpose(1, 2).squeeze()
        # the initial face seq
        src = joint_emb.transpose(1, 0)  # L,N,E

        src_mask = generate_square_subsequent_mask(src.shape[0]).to(device)

        # output face seq
        gen_seq = [0]
        # output in_face_pos
        curr_in_face_pos = 0
        gen_in_face_pos_seq = [0]
        # output face_pos
        curr_face_pos = 0
        gen_face_pos_seq = [0]

        # optionally create a class condition vector
        # need to do this only once
        cond_vec = (
            None
            if class_ndx is None
            else self.class_emb(torch.LongTensor([class_ndx]).to(device))
        )

        # we already have one output token, get the rest
        for out_ndx in range(1, out_seq_len):
            # S,N,E -> N,S,E
            pointer = self.face_encoder(src, src_mask, cond_vec=cond_vec).transpose(
                0, 1
            )

            # get scores
            # (N,S,E) x ((V+2),E)
            product = torch.matmul(pointer, vtx_emb)
            scores = product[0]
            # take scores only for the vertices that we have, and the new and stop tokens
            # discard the rest
            scores_reduced = torch.cat(
                [scores[:, :num_vertices], scores[:, -2:]], dim=-1
            )
            # pick scores only for the index that we are currently decoding
            scores_reduced = scores_reduced[out_ndx - 1]

            if probabilistic and nucleus:
                # pick the top logits whose probs add up to 'nucleus'
                scores_reduced = sample_top_p(scores_reduced, top_p=nucleus)

            probs = F.softmax(scores_reduced, dim=-1)
            if probabilistic:
                next_token = Categorical(probs=probs).sample()
            else:
                _, next_token = torch.max(probs, dim=-1)

            next_token = next_token.item()

            # last but one embedding: corresponds to new_face_token
            if next_token == num_vertices:
                # add new face token to the output seq
                gen_seq.append(self.new_face_token.item())
                # move to the next face
                curr_face_pos += 1
                # pad the face_pos sequence
                gen_face_pos_seq.append(self.new_face_token.item())
                # reset the in_face position
                curr_in_face_pos = -1
                # pad the in_face_pos sequence
                gen_in_face_pos_seq.append(self.new_face_token.item())
            # last embedding: corresponds to face_stop_token
            elif next_token == num_vertices + 1:
                # add a face stop token to the output seq
                gen_seq.append(self.face_stop_token.item())
                # done decoding
                break
            # continue in the same face
            else:
                # keep the token as-is
                gen_seq.append(next_token)
                # increase the in_face_pos
                curr_in_face_pos += 1
                gen_face_pos_seq.append(curr_face_pos)
                gen_in_face_pos_seq.append(curr_in_face_pos)

            if (
                curr_face_pos >= self.max_face_num
                or curr_in_face_pos >= self.max_vtx_per_face
            ):
                # reached an invalid sequence
                return torch.LongTensor([])
            # new face sequence
            curr_seq = gen_seq + (out_seq_len - len(gen_seq)) * [0]
            curr_seq = torch.LongTensor(curr_seq).view(1, -1).to(device)
            # new face_pos sequence
            face_pos_seq = gen_face_pos_seq + (out_seq_len - len(gen_seq)) * [0]
            face_pos_seq = torch.LongTensor(face_pos_seq).view(1, -1).to(device)
            # new in_face_pos sequence
            in_face_pos_seq = gen_in_face_pos_seq + (out_seq_len - len(gen_seq)) * [0]
            in_face_pos_seq = torch.LongTensor(in_face_pos_seq).view(1, -1).to(device)
            # embed again
            joint_emb, _ = self.get_embedding(
                vertices_raw_batch, face_pos_seq, in_face_pos_seq, curr_seq
            )
            # update the input
            src = joint_emb.transpose(1, 0)  # L,N,E,

        return torch.LongTensor(gen_seq)
