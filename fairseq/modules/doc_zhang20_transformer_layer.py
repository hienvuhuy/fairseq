# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Hien-v: Reuse and develope from transforer_layer.py
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils, doc_utils
from fairseq.modules import LayerNorm, MultiheadAttention#DocMultiheadAttention, DocMultiFeaturesMultiheadAttention #,MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
import json
from fairseq.models.doc_zhang_20 import (
    DocZhang20TransformerConfig
)

class DocZhang20TransformerEncoderLayerBase(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        #hien-v
        # self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.local_attn = self.build_self_attention(self.embed_dim, cfg)
        self.global_attn = self.build_self_attention(self.embed_dim, cfg)
        self.attn_linear_proj_layer = self.build_attention_linear_layer(self.embed_dim, 
                            self.embed_dim, 
                            self.quant_noise,
                            self.quant_noise_block_size)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.encoder.normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.encoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )
    def build_attention_linear_layer(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim*2, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        local_attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            local_attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # attn_dict={}
        if local_attn_mask is not None:
            local_attn_mask = local_attn_mask.masked_fill(local_attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # x, _ = self.self_attn(
        #     query=x,
        #     key=x,
        #     value=x,
        #     key_padding_mask=encoder_padding_mask,
        #     need_weights=False,
        #     attn_mask=attn_mask,
        # )
        x_local, _ = self.local_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=local_attn_mask,
        )
        x_global, _ = self.global_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
        )
        x = torch.cat((x_local, x_global), dim=-1)
        x = self.attn_linear_proj_layer(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


# backward compatible with the legacy argparse format
class DocZhang20TransformerEncoderLayer(DocZhang20TransformerEncoderLayerBase):
    def __init__(self, args):
        super().__init__(DocZhang20TransformerConfig.from_namespace(args))
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, DocZhang20TransformerConfig.from_namespace(args)
        )


class DocZhang20TransformerDecoderLayerBase(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.cross_self_attention = cfg.cross_self_attention

        # Hien-v:
        # Change self-attn -> local-attn + global-attn
        # self.self_attn = self.build_self_attention(
        #     self.embed_dim,
        #     cfg,
        #     add_bias_kv=add_bias_kv,
        #     add_zero_attn=add_zero_attn,
        # )

        self.global_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.local_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.attn_linear_proj_layer = self.build_attention_linear_layer(self.embed_dim, 
                            self.embed_dim,
                            self.quant_noise,
                            self.quant_noise_block_size)
        # need to check dim

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.decoder.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        if no_encoder_attn:
            # self.encoder_attn = None
            # self.encoder_attn_layer_norm = None
            self.local_encoder_attn = None
            self.global_encoder_attn = None
        else:
            # hien-v: change encoder_attn -> local and global attn
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)


            # self.local_encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            # self.global_encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            # self.encoder_attn_linear_proj_layer = self.build_encoder_attention_linear_layer(
            #                 self.embed_dim, 
            #                 self.embed_dim, 
            #                 self.quant_noise,
            #                 self.quant_noise_block_size)
            # self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_attention_linear_layer(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim*2, output_dim), p=q_noise, block_size=qn_block_size
        )
    def build_encoder_attention_linear_layer(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim*2, output_dim), p=q_noise, block_size=qn_block_size
        )
    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None, # same
        encoder_padding_mask: Optional[torch.Tensor] = None, #same, in Gtrans-> padding_mask
        encoder_local_mask: Optional[torch.Tensor] = None, # new, this is encoder-decoder mask for local
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None, # same
        prev_self_attn_state: Optional[List[torch.Tensor]] = None, # same, but unused
        prev_attn_state: Optional[List[torch.Tensor]] = None, # same, but unused
        # self_attn_mask: Optional[torch.Tensor] = None, separate to local and global; local is self_attn_mask
        local_attn_mask: Optional[torch.Tensor] = None, # new, it is self_attn_mask for local. where is a mask for encoder-decoder attn?
        global_attn_mask: Optional[torch.Tensor] = None, # new, it is self_attn_mask for global
        self_attn_padding_mask: Optional[torch.Tensor] = None, # same
        need_attn: bool = False, # same
        need_head_weights: bool = False, # same
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # Can generate local_attn_mask, global_attn_mask and encoder_local_mask
        #
        if need_head_weights:
            need_attn = True
        # from pudb import set_trace; set_trace()
        attn_dict = {}
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # if prev_self_attn_state is not None:
        #     # this block is unused
        #     prev_key, prev_value = prev_self_attn_state[:2]
        #     saved_state: Dict[str, Optional[Tensor]] = {
        #         "prev_key": prev_key,
        #         "prev_value": prev_value,
        #     }
        #     if len(prev_self_attn_state) >= 3:
        #         saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
        #     assert incremental_state is not None
        #     self.self_attn._set_input_buffer(incremental_state, saved_state)
        # # _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        # _self_attn_input_buffer = self.global_attn._get_input_buffer(incremental_state)
        # # _self
        # if self.cross_self_attention and not (
        #     incremental_state is not None
        #     and _self_attn_input_buffer is not None
        #     and "prev_key" in _self_attn_input_buffer
        # ):
        #     if self_attn_mask is not None:
        #         assert encoder_out is not None
        #         self_attn_mask = torch.cat(
        #             (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
        #         )
        #     if self_attn_padding_mask is not None:
        #         if encoder_padding_mask is None:
        #             assert encoder_out is not None
        #             encoder_padding_mask = self_attn_padding_mask.new_zeros(
        #                 encoder_out.size(1), encoder_out.size(0)
        #             )
        #         self_attn_padding_mask = torch.cat(
        #             (encoder_padding_mask, self_attn_padding_mask), dim=1
        #         )
        #     assert encoder_out is not None
        #     y = torch.cat((encoder_out, x), dim=0)
        # else:
        #     y = x
        y = x

        

        x_local, attn_local = self.local_attn( # done
            query=x, key=y, value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            # attn_mask=self_attn_mask,
            attn_mask=local_attn_mask
        )
        # from pudb import set_trace; set_trace()
        if attn_local is not None:
            attn_dict['decoder_self_local'] = attn_local

        x_global, attn_global = self.global_attn(# done
            query=x, key=y, value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=global_attn_mask,
        )

        if attn_global is not None:
            attn_dict['decoder_self_global'] = attn_global
        
        # merge global and local attns
        x_combine = torch.cat((x_local, x_global), dim=-1)
        x = self.attn_linear_proj_layer(x_combine)

        # x, attn = self.self_attn(
        #     query=x,
        #     key=y,
        #     value=y,
        #     key_padding_mask=self_attn_padding_mask,
        #     incremental_state=incremental_state,
        #     need_weights=False,
        #     attn_mask=self_attn_mask,
        # )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
        # if self.local_encoder_attn is not None and self.global_encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            # if prev_attn_state is not None:
            #     prev_key, prev_value = prev_attn_state[:2]
            #     saved_state: Dict[str, Optional[Tensor]] = {
            #         "prev_key": prev_key,
            #         "prev_value": prev_value,
            #     }
            #     if len(prev_attn_state) >= 3:
            #         saved_state["prev_key_padding_mask"] = prev_attn_state[2]
            #     assert incremental_state is not None
            #     self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            # tam thoi comment
            # x_e_local, attn_e_local = self.local_encoder_attn(
            #     query=x, key=encoder_out, value=encoder_out,
            #     key_padding_mask=encoder_padding_mask,
            #     incremental_state=incremental_state,
            #     static_kv=True,
            #     attn_mask=encoder_local_mask, # need specify this value. is it correct?
            #     need_weights=need_attn or (not self.training and self.need_attn),
            #     need_head_weights=need_head_weights,
            # )
            # if attn_e_local is not None:
            #     attn_dict['decoder_cross_local'] = attn_e_local
            
            # x_e_global, attn_e_global = self.global_encoder_attn( #done, it is unchanged
            #     query=x, key=encoder_out, value=encoder_out,
            #     key_padding_mask=encoder_padding_mask,
            #     incremental_state=incremental_state,
            #     static_kv=True,
            #     need_weights=need_attn or (not self.training and self.need_attn),
            #     need_head_weights=need_head_weights,
            # )
            # x_combine = torch.cat((x_e_local, x_e_global), dim=-1)
            # x = self.encoder_attn_linear_proj_layer(x_combine)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            # I have not seen the main flow goes to this block
            saved_state_local = self.local_attn._get_input_buffer(incremental_state)
            assert saved_state_local is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state_local["prev_key"],
                    saved_state_local["prev_value"],
                    saved_state_local["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state_local["prev_key"], saved_state_local["prev_value"]]
            # return x, attn, self_attn_state
            return x, attn_dict, self_attn_state
        return x, attn_dict, None #attn_dict is used only for criterion with alignment
        # return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


# backward compatible with the legacy argparse format
class DocZhang20TransformerDecoderLayer(DocZhang20TransformerDecoderLayerBase):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            DocZhang20TransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            DocZhang20TransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            DocZhang20TransformerConfig.from_namespace(args),
        )
