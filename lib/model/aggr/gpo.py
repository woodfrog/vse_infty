# coding=utf-8
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden, pe_mode='sin', decode_mode='seq'):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden
        self.pe_mode = pe_mode
        self.decode_mode = decode_mode
        assert pe_mode in ['sin', 'learn', 'index']
        assert decode_mode in ['seq', 'interp']

        if pe_mode == 'learn':
            self.pe_embedding = nn.Embedding(1000, self.d_pe)
        else:
            self.pe_database = {}
        if decode_mode == 'seq':
            self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(self.d_hidden, 1, bias=False)
        else:
            self.n_pieces = 20
            self.weight = nn.Parameter(torch.zeros(1, self.n_pieces + 1) + 1.0 / self.n_pieces)

    def compute_pool_weights(self, lengths, features):
        if self.decode_mode == 'seq':
            max_len = int(lengths.max())
            pe_max_len = self.get_pe(max_len)
            pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
            mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
            mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
            pes = pes.masked_fill(mask == 0, 0)

            self.gru.flatten_parameters()
            packed = pack_padded_sequence(pes, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            padded = pad_packed_sequence(out, batch_first=True)
            out_emb, out_len = padded
            out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
            scores = self.linear(out_emb)
            scores[torch.where(mask == 0)] = -10000

            weights = torch.softmax(scores / 0.1, 1)
            return weights, mask
        else:
            sizes, mask = self.fill_sizes(lengths)

            # turn continuous into concrete weights
            weights = self.determine_weight(sizes)
            weights = weights.squeeze().unsqueeze(-1)
            weights[torch.where(mask == 0)] = -10000
            weights = torch.softmax(weights / 0.1, 1)

            mask = mask.bool()
            return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features, pool_weights

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if self.pe_mode == 'sin':
            if length in self.pe_database:
                return self.pe_database[length]
            else:
                pe = positional_encoding_1d(self.d_pe, length)
                self.pe_database[length] = pe
                return pe
        elif self.pe_mode == 'learn':
            device = list(self.pe_embedding.parameters())[0].device
            pos_idx = torch.arange(length, device=device)
            pe = self.pe_embedding(pos_idx)
            return pe
        else:
            pe = torch.zeros([length, self.d_pe])
            return pe


    @staticmethod
    def fill_sizes(sizes):
        """
            sizes is a LongTensor of size [batch_size], containing the set sizes.
            Each set size n is turned into [0/(n-1), 1/(n-1), ..., (n-2)/(n-1), 1, 0, 0, ..., 0, 0].
            These are the ratios r at which f is evaluated at.
            The 0s at the end are there for padding to the largest n in the batch.
            If the input set x is passed in, it guarantees that the mask is the correct size even when sizes.max()
            is less than x.size(), which can be a case if there is at least one padding element in each set in the batch.
        """
        max_size = sizes.max()

        size_tensor = torch.arange(end=max_size, device=sizes.device, dtype=torch.float32)
        size_tensor = size_tensor.unsqueeze(0) / (sizes.float() - 1).clamp(min=1).unsqueeze(1)

        mask = size_tensor <= 1
        mask = mask.unsqueeze(-1)

        return size_tensor.clamp(max=1), mask.float()


    def determine_weight(self, sizes):
        """
            Piecewise linear function. Evaluates f at the ratios in sizes.
            This should be a faster implementation than doing the sum over max terms, since we know that most terms in it are 0.
        """
        # share same sequence length within each sample, so copy weighht across batch dim
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(sizes.size(0), 1, weight.size(2))

        # linspace [0, 1] -> linspace [0, n_pieces]
        index = self.n_pieces * sizes
        index = index.unsqueeze(1)
        index = index.expand(index.size(0), weight.size(1), index.size(2))

        # points in the weight vector to the left and right
        idx = index.long()
        frac = index.frac()
        left = weight.gather(2, idx)
        right = weight.gather(2, (idx + 1).clamp(max=self.n_pieces))

        # interpolate between left and right point
        return (1 - frac) * left + frac * right
