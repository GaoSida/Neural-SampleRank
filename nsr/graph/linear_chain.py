"""Forward-backward algorithm loss inference, and Viterbi Decoding for linear-
chain CRF models.
The main implementation comes from Flair.
"""
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


class NegLogLikelihoodFwdLoss(nn.Module):
    """Neg log likelihood loss for forward algorithm.
    """
    def __init__(self, label_vocab_size, device):
        super().__init__()
        self.label_vocab_size = label_vocab_size
        self.start_tag_index = label_vocab_size
        self.stop_tag_index = label_vocab_size + 1
        self.device = device
    
    def _forward_algo(self, emissions, transitions, lengths):
        
        init_alphas = torch.FloatTensor(
            self.label_vocab_size + 2).fill_(-10000.0)
        init_alphas[self.start_tag_index] = 0.0
        
        forward_var = torch.zeros(
            emissions.shape[0], emissions.shape[1] + 1, emissions.shape[2],
            dtype=torch.float, device=self.device
        )
        
        forward_var[:, 0, :] = init_alphas[None, :].repeat(emissions.shape[0],
                                                           1)
        
        repeated_transitions = transitions.view(
            1, transitions.shape[0], transitions.shape[1]
        ).repeat(emissions.shape[0], 1, 1)
        
        for i in range(emissions.shape[1]):
            emit_score = emissions[:, i, :]
            
            tag_var = (
                emit_score[:, :, None].repeat(1, 1, transitions.shape[1])
                + repeated_transitions
                + forward_var[:, i, :][:, :, None]
                .repeat(1, 1, transitions.shape[1])
                .transpose(2, 1)
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[1]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned
        
        forward_var = forward_var[range(forward_var.shape[0]), lengths, :]

        terminal_var = forward_var + transitions[self.stop_tag_index][
            None, :].repeat(forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha

    def _score_label(self, emissions, transitions, lengths, labels):
        
        start = torch.tensor([self.start_tag_index], device=self.device)
        start = start[None, :].repeat(labels.shape[0], 1)
        
        stop = torch.tensor([self.stop_tag_index], device=self.device)
        stop = stop[None, :].repeat(labels.shape[0], 1)
        
        pad_start_tags = torch.cat([start, labels], 1)
        pad_stop_tags = torch.cat([labels, stop], 1)
        
        for i in range(len(lengths)):
            pad_stop_tags[i, lengths[i]:] = self.stop_tag_index
        
        score = torch.FloatTensor(emissions.shape[0]).to(self.device)
        
        for i in range(emissions.shape[0]):
            r = torch.LongTensor(range(lengths[i])).to(self.device)

            score[i] = torch.sum(
                transitions[pad_stop_tags[i, : lengths[i] + 1],
                            pad_start_tags[i, : lengths[i] + 1]]
            ) + torch.sum(emissions[i, r, labels[i, : lengths[i]]])
        
        return score
    
    def forward(self, emissions: Tensor, transitions: Tensor, lengths: Tensor,
                labels: Tensor) -> Tensor:
        """Compute negative log likelihood loss with forward algorithm.
        Args:
            emissions: [batch_size, max_num_tokens, label_vocab_size + 2]
            transitions: [label_vocab_size + 2, label_vocab_size + 2]
            lengths: [batch_size, ]
            labels: [batch_size, max_num_tokens]
        """
        lengths_list = lengths.tolist()
        
        forward_score = self._forward_algo(emissions, transitions,
                                           lengths_list)
        gold_score = self._score_label(emissions, transitions, lengths_list,
                                       labels)
        
        score = forward_score - gold_score
        
        return score.mean()

    def viterbi(self, emissions: Tensor, transitions: Tensor,
                lengths: Tensor) -> Tensor:
        """Viterbi decoding.
        Args:
            emissions: [batch_size, max_num_tokens, label_vocab_size + 2]
            transitions: [label_vocab_size + 2, label_vocab_size + 2]
            lengths: [batch_size, ]
        Returns: labels [batch_size, max_num_tokens]
        """
        emissions = emissions.detach().cpu().numpy()
        max_num_tokens = emissions.shape[1]
        transitions = transitions.detach().cpu().numpy()
        lengths_list = lengths.tolist()
        
        predictions = list()
        for emits, length in zip(emissions, lengths_list):
            emits = emits[:length]
            
            id_start = self.start_tag_index
            id_stop = self.stop_tag_index

            backpointers = np.empty(
                shape=(emits.shape[0], self.label_vocab_size + 2),
                dtype=np.int_
            )
            backscores = np.empty(
                shape=(emits.shape[0], self.label_vocab_size + 2),
                dtype=np.float32
            )

            init_vvars = np.expand_dims(
                np.repeat(-10000.0, self.label_vocab_size + 2), axis=0
            ).astype(np.float32)
            init_vvars[0][id_start] = 0

            forward_var = init_vvars
            for index, feat in enumerate(emits):
                # broadcasting will do the job of reshaping and is more 
                # efficient than calling repeat
                next_tag_var = forward_var + transitions
                bptrs_t = next_tag_var.argmax(axis=1)
                viterbivars_t = next_tag_var[np.arange(bptrs_t.shape[0]), 
                                             bptrs_t]
                forward_var = viterbivars_t + feat
                backscores[index] = forward_var
                forward_var = forward_var[np.newaxis, :]
                backpointers[index] = bptrs_t

            terminal_var = forward_var.squeeze() + transitions[id_stop]
            terminal_var[id_stop] = -10000.0
            terminal_var[id_start] = -10000.0
            best_tag_id = terminal_var.argmax()

            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)

            start = best_path.pop()
            assert start == id_start
            best_path.reverse()
            
            predictions.append(best_path + [1] * (max_num_tokens - length))
    
        return torch.tensor(predictions, dtype=torch.int)        
