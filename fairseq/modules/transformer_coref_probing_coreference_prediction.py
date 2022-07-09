# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple
from xml.sax.xmlreader import AttributesNSImpl

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module, LayerNorm, Dropout, Parameter

from fairseq import utils, probing_utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

def mask_tensor(t, mask):
    t = t + ((1.0 - mask.float()) * -10000.0)
    t = torch.clamp(t, min=-10000.0, max=10000.0)
    return t

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class FullyConnectedLayer(nn.Module):
    def __init__(self, cfg, input_dim, output_dim, dropout_prob):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense = Linear(self.input_dim, self.output_dim)

        self.layer_norm = LayerNorm(self.output_dim, eps=cfg.coref_module.layer_norm_eps)
        # self.activation_func = ACT2FN[config.hidden_act]
        self.activation_func = torch.nn.GELU()
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dropout(temp)
        return temp


class S2ECoref(nn.Module):
    # def __init__(self, config, args):
    def __init__(self, cfg, dtype=torch.float32):
        super().__init__()
        # from pudb import set_trace; set_trace()
        self.max_span_length = cfg.coref_module.max_span_length
        self.top_lambda = cfg.coref_module.top_lambda
        # self.ffnn_size = cfg.coref_module.ffnn_size
        self.ffnn_size = cfg.coref_module.ffn_embed_dim
        self.do_mlps = self.ffnn_size > 0
        self.ffnn_size = self.ffnn_size if self.do_mlps else cfg.coref_module.hidden_size
        self.normalise_loss = cfg.coref_module.normalise_loss

        # self.longformer = LongformerModel(config)
        self.start_mention_mlp = FullyConnectedLayer(cfg, cfg.coref_module.hidden_size, self.ffnn_size, cfg.coref_module.dropout_prob) if self.do_mlps else None
        self.end_mention_mlp = FullyConnectedLayer(cfg, cfg.coref_module.hidden_size, self.ffnn_size, cfg.coref_module.dropout_prob) if self.do_mlps else None
        self.start_coref_mlp = FullyConnectedLayer(cfg, cfg.coref_module.hidden_size, self.ffnn_size, cfg.coref_module.dropout_prob) if self.do_mlps else None
        self.end_coref_mlp = FullyConnectedLayer(cfg, cfg.coref_module.hidden_size, self.ffnn_size, cfg.coref_module.dropout_prob) if self.do_mlps else None

        self.start_coref_mlp = FullyConnectedLayer(cfg, cfg.coref_module.hidden_size, self.ffnn_size, cfg.coref_module.dropout_prob) if self.do_mlps else None
        self.end_coref_mlp = FullyConnectedLayer(cfg, cfg.coref_module.hidden_size, self.ffnn_size, cfg.coref_module.dropout_prob) if self.do_mlps else None

        self.mention_start_classifier = Linear(self.ffnn_size, 1)
        self.mention_end_classifier = Linear(self.ffnn_size, 1)
        self.mention_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.antecedent_s2s_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_e2e_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_e2s_classifier = Linear(self.ffnn_size, self.ffnn_size)

        # from pudb import set_trace; set_trace()
        # self.init_weights()
        self.dtype = dtype

        # self.tpu = None if not hasattr(cfg, 'common') else cfg.common.tpu
        # self.cuda = torch.cuda.is_available() and not cfg.common.cpu and not self.tpu
        # if self.cuda:
        #     self.device = torch.device("cuda")
        # elif self.tpu:
        #     self.device = utils.get_tpu_device()
        # else:
        #     self.device = torch.device("cpu")


    def init_weights(self):
        """
        If needed prunes and maybe initializes weights.
        """

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):

        """Initialize the weights"""
        # need to check xavier initialziation
        for m in self.modules():
            if isinstance(module, nn.Linear):
                # from pudb import set_trace; set_trace()
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
    def _calc_mention_logits(self, start_mention_reps, end_mention_reps):
        start_mention_logits = self.mention_start_classifier(start_mention_reps).squeeze(-1)  # [batch_size, seq_length]
        end_mention_logits = self.mention_end_classifier(end_mention_reps).squeeze(-1)  # [batch_size, seq_length]

        temp = self.mention_s2e_classifier(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch.matmul(temp,
                                            end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)
        mention_mask = self._get_mention_mask(mention_logits)  # [batch_size, seq_length, seq_length]
        mention_logits = mask_tensor(mention_logits, mention_mask)  # [batch_size, seq_length, seq_length]
        return mention_logits

    def _get_span_mask(self, batch_size, k, max_k):
        """
        :param batch_size: int
        :param k: tensor of size [batch_size], with the required k for each example
        :param max_k: int
        :return: [batch_size, max_k] of zero-ones, where 1 stands for a valid span and 0 for a padded span
        """
        size = (batch_size, max_k)
        # from pudb import set_trace; set_trace()
        # idx = torch.arange(max_k, device=self.device).unsqueeze(0).expand(size)
        idx = torch.arange(max_k, device=k.device).unsqueeze(0).expand(size)
        # idx = torch.arange(max_k).unsqueeze(0).expand(size)
        len_expanded = k.unsqueeze(1).expand(size)
        return (idx < len_expanded).int()

    def _prune_topk_mentions(self, mention_logits, attention_mask):
        """
        :param mention_logits: Shape [batch_size, seq_length, seq_length]
        :param attention_mask: [batch_size, seq_length]
        :param top_lambda:
        :return:
        """
        # from pudb import set_trace; set_trace()
        batch_size, seq_length, _ = mention_logits.size()
        actual_seq_lengths = torch.sum(attention_mask, dim=-1)  # [batch_size]

        k = (actual_seq_lengths * self.top_lambda).int()  # [batch_size]
        max_k = int(torch.max(k))  # This is the k for the largest input in the batch, we will need to pad

        _, topk_1d_indices = torch.topk(mention_logits.view(batch_size, -1), dim=-1, k=max_k)  # [batch_size, max_k]
        span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * ((seq_length ** 2) - 1)  # We take different k for each example
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        # topk_mention_start_ids = sorted_topk_1d_indices // seq_length  # [batch_size, max_k]
        topk_mention_start_ids = torch.div(sorted_topk_1d_indices, seq_length, rounding_mode='floor')  # [batch_size, max_k]
        topk_mention_end_ids = sorted_topk_1d_indices % seq_length  # [batch_size, max_k]

        topk_mention_logits = mention_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
                                             topk_mention_start_ids, topk_mention_end_ids]  # [batch_size, max_k]

        topk_mention_logits = topk_mention_logits.unsqueeze(-1) + topk_mention_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]

        return topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_mention_logits

    def _mask_antecedent_logits(self, antecedent_logits, span_mask):
        # We now build the matrix for each pair of spans (i,j) - whether j is a candidate for being antecedent of i?
        antecedents_mask = torch.ones_like(antecedent_logits, dtype=self.dtype).tril(diagonal=-1)  # [batch_size, k, k]
        antecedents_mask = antecedents_mask * span_mask.unsqueeze(-1)  # [batch_size, k, k]
        antecedent_logits = mask_tensor(antecedent_logits, antecedents_mask)
        return antecedent_logits

    def _get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k + 1), device='cpu')
        all_clusters_cpu = all_clusters.cpu().numpy()
        for b, (starts, ends, gold_clusters) in enumerate(zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)):
            gold_clusters = probing_utils.extract_clusters(gold_clusters)
            mention_to_gold_clusters = probing_utils.extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
        # from pudb import set_trace; set_trace()
        # new_cluster_labels = new_cluster_labels.to(self.device)
        new_cluster_labels = new_cluster_labels.to(span_starts.device)
        no_antecedents = 1 - torch.sum(new_cluster_labels, dim=-1).bool().float()
        new_cluster_labels[:, :, -1] = no_antecedents
        return new_cluster_labels

    def _get_marginal_log_likelihood_loss(self, coref_logits, cluster_labels_after_pruning, span_mask):
        """
        :param coref_logits: [batch_size, max_k, max_k]
        :param cluster_labels_after_pruning: [batch_size, max_k, max_k]
        :param span_mask: [batch_size, max_k]
        :return:
        """
        gold_coref_logits = mask_tensor(coref_logits, cluster_labels_after_pruning)

        gold_log_sum_exp = torch.logsumexp(gold_coref_logits, dim=-1)  # [batch_size, max_k]
        all_log_sum_exp = torch.logsumexp(coref_logits, dim=-1)  # [batch_size, max_k]

        gold_log_probs = gold_log_sum_exp - all_log_sum_exp
        losses = - gold_log_probs
        losses = losses * span_mask
        per_example_loss = torch.sum(losses, dim=-1)  # [batch_size]
        if self.normalise_loss:
            per_example_loss = per_example_loss / losses.size(-1)
        loss = per_example_loss.mean()
        return loss

    def _get_mention_mask(self, mention_logits_or_weights):
        """
        Returns a tensor of size [batch_size, seq_length, seq_length] where valid spans
        (start <= end < start + max_span_length) are 1 and the rest are 0
        :param mention_logits_or_weights: Either the span mention logits or weights, size [batch_size, seq_length, seq_length]
        """
        mention_mask = torch.ones_like(mention_logits_or_weights, dtype=self.dtype)
        mention_mask = mention_mask.triu(diagonal=0)
        mention_mask = mention_mask.tril(diagonal=self.max_span_length - 1)
        return mention_mask

    def _calc_coref_logits(self, top_k_start_coref_reps, top_k_end_coref_reps):
        # s2s
        temp = self.antecedent_s2s_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2s_coref_logits = torch.matmul(temp,
                                              top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2e
        temp = self.antecedent_e2e_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2e_coref_logits = torch.matmul(temp,
                                              top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # s2e
        temp = self.antecedent_s2e_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2e_coref_logits = torch.matmul(temp,
                                              top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2s
        temp = self.antecedent_e2s_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2s_coref_logits = torch.matmul(temp,
                                              top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # sum all terms
        coref_logits = top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits  # [batch_size, max_k, max_k]
        return coref_logits        

    # def forward(self, input_representation, input_ids=None, gold_clusters=None):
    def forward(self, input_representation, gold_clusters=None):
        
        """
            input_representation: tuple(dict, None, None)
                dict:
                    encoder_out: [x] # T x B x C
                    encoder_padding_mask: [encoder_padding_mask] # B x T 
                    encoder_embedding: [encoder_embedding] # B x T x C
                    encoder_state: encoder_states,  # List[T x B x C]
                    src_tokens: []
                    src_lengths: [src_lengths]

            Note: we need:
                input_ids: shape: B x C: Don't need it anymore. We had the vector representation of input_ids directly
                sequence_output (from longformer): shape: B x T x C
                attention_mask: B x T
        """
        # from pudb import set_trace; set_trace()
        # outputs = self.longformer(input_ids, attention_mask=attention_mask)
        # attention_mask = None # hold variable for attention_mask of longformer. We will remove it now

        attention_mask = torch.where(input_representation[0]['encoder_padding_mask'][0], 0, 1) # the `encoder_padding_mask` means True is masking; so we use 0 for preventing attention on padding
        outputs = input_representation[0]['bert_representations_out']
        sequence_output = outputs # [batch_size, seq_len, dim]

        # we change from T x B x C -> B x T x C
        sequence_output = sequence_output.transpose(0, 1)
        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output) if self.do_mlps else sequence_output
        end_mention_reps = self.end_mention_mlp(sequence_output) if self.do_mlps else sequence_output

        start_coref_reps = self.start_coref_mlp(sequence_output) if self.do_mlps else sequence_output
        end_coref_reps = self.end_coref_mlp(sequence_output) if self.do_mlps else sequence_output

        
        # mention scores
        mention_logits = self._calc_mention_logits(start_mention_reps, end_mention_reps)

        # from pudb import set_trace; set_trace()
        ## different in attention_mask
        # prune mentions
        mention_start_ids, mention_end_ids, span_mask, topk_mention_logits = self._prune_topk_mentions(mention_logits, attention_mask)

        batch_size, _, dim = start_coref_reps.size()
        max_k = mention_start_ids.size(-1)
        size = (batch_size, max_k, dim)

        # Antecedent scores
        # gather reps
        topk_start_coref_reps = torch.gather(start_coref_reps, dim=1, index=mention_start_ids.unsqueeze(-1).expand(size))
        topk_end_coref_reps = torch.gather(end_coref_reps, dim=1, index=mention_end_ids.unsqueeze(-1).expand(size))
        coref_logits = self._calc_coref_logits(topk_start_coref_reps, topk_end_coref_reps)

        final_logits = topk_mention_logits + coref_logits
        final_logits = self._mask_antecedent_logits(final_logits, span_mask)
        # from pudb import set_trace; set_trace()
        # adding zero logits for null span
        final_logits = torch.cat((final_logits, torch.zeros((batch_size, max_k, 1), device=final_logits.device)), dim=-1)  # [batch_size, max_k, max_k + 1]
        # final_logits = torch.cat((final_logits, torch.zeros((batch_size, max_k, 1), device=self.device)), dim=-1)  # [batch_size, max_k, max_k + 1]
        # final_logits = torch.cat((final_logits, torch.zeros((batch_size, max_k, 1))), dim=-1)  # [batch_size, max_k, max_k + 1]

        if gold_clusters is not None:
            labels_after_pruning = self._get_cluster_labels_after_pruning(mention_start_ids, mention_end_ids, gold_clusters)
            loss = self._get_marginal_log_likelihood_loss(final_logits, labels_after_pruning, span_mask)
        return {
                    "coref_probing_output":(labels_after_pruning, mention_start_ids, mention_end_ids, final_logits, mention_logits, span_mask),
                    "coref_loss": loss

                }
        # if return_all_outputs:
        #     outputs = (mention_start_ids, mention_end_ids, final_logits, mention_logits)
        # else:
        #     outputs = tuple()

        # if gold_clusters is not None:
        #     losses = {}
        #     labels_after_pruning = self._get_cluster_labels_after_pruning(mention_start_ids, mention_end_ids, gold_clusters)
        #     loss = self._get_marginal_log_likelihood_loss(final_logits, labels_after_pruning, span_mask)
        #     losses.update({"loss": loss})
        #     outputs = (loss,) + outputs + (losses,)

        # return outputs