# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class CRF2(nn.Module):
    def __init__(self, config, num_embeddings, embedding_dim, padding_idx, label_size, embeddings):
        super(CRF2, self).__init__()
        self.config = config
        self.label_size = label_size

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        if embeddings is not None:
            self.embedding.from_pretrained(torch.from_numpy(embeddings))
        self.dropout = nn.Dropout(config.dropout_embed)
        self.lstm = nn.LSTM(embedding_dim, config.hidden_size, num_layers=config.num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(config.hidden_size * 2, label_size)

        self.START_TAG, self.STOP_TAG = label_size - 2, label_size - 1
        self.transitions = nn.Parameter(torch.randn(label_size, label_size))
        self.transitions.data[:, self.START_TAG] = -10000
        self.transitions.data[self.STOP_TAG, :] = -10000

    def _get_lstm_features(self, feats):
        feats = self.embedding(feats)
        # h = pack_padded_sequence(x, length)
        h, _ = self.lstm(feats)
        # h, _ = pad_packed_sequence(h)
        h = self.dropout(h)
        # h = torch.transpose(h, 0, 1)
        lstm_feats = torch.squeeze(h, 1)
        lstm_feats = self.hidden2tag(lstm_feats)

        return lstm_feats

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.label_size), -10000.).cuda()
        init_alphas[0][self.START_TAG] = 0.
        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.label_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.label_size)
                trans_score = self.transitions[:, next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[:, self.STOP_TAG]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # my function
        score = torch.zeros(1).cuda()
        score = score + self.transitions[self.START_TAG, tags[0]]
        for i in range(len(feats) - 1):
            score = score + self.transitions[tags[i], tags[i + 1]] + feats[i][tags[i]]
        score = score + self.transitions[tags[-1], self.STOP_TAG] + feats[-1][tags[-1]]
        return score

        # function 2
        # score = torch.zeros(1).cuda()
        # tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long).cuda(), tags])
        # for i, feat in enumerate(feats):
        #     score = score + \
        #             self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # score = score + self.transitions[self.STOP_TAG, tags[-1]]
        # return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.label_size), -10000.).cuda()
        init_vvars[0][self.START_TAG] = 0

        # forward_var at step i holds the biterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the biterbi variables for this step

            for next_tag in range(self.label_size):
                next_tar_var = forward_var + self.transitions[:, next_tag]
                best_tag_id = argmax(next_tar_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tar_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[:, self.STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Fellow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.START_TAG
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, feats, tags):
        lstm_feats = self._get_lstm_features(feats)

        forward_score = self._forward_alg(lstm_feats)
        gold_score = self._score_sentence(lstm_feats, tags)

        return forward_score - gold_score

    def forward(self, feats):
        lstm_feats = self._get_lstm_features(feats)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
