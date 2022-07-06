"""
Custom Transformer layers and classes
Adapted from the official implementation

Eg: use a condition vector in the transformer
"""

from typing import Optional

from torch import Tensor
import torch.nn as nn


class CustomTransformerEncoder(nn.TransformerEncoder):
    r"""Accept an additional condition vector
    that is added to each token after each self attention layer
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        cond_vec: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            cond_vec: conditional vector (optional)
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                cond_vec=cond_vec,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        cond_vec: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            cond_vec: conditional vector (optional)

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        # broadcast and add cond_vec
        # dimensions should match!
        # src2: (seq_len, batchsize, fwd_dim)
        # cond_vec: (batchsize, fwd_dim)

        if cond_vec is not None:
            src2 += cond_vec

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
