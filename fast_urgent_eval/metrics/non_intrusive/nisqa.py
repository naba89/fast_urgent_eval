# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import copy
import math
import os
from functools import partial

import numpy as np
import pandas as pd
import torchaudio.transforms

pd.options.mode.chained_assignment = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class NISQA_DIM(nn.Module):
    '''
    NISQA_DIM: The main speech quality model with speech quality dimension
    estimation (MOS, Noisiness, Coloration, Discontinuity, and Loudness).
    The module loads the submodules for framewise modelling (e.g. CNN),
    time-dependency modelling (e.g. Self-Attention or LSTM), and pooling
    (e.g. max-pooling or attention-pooling)
    '''

    def __init__(self,
                 ms_seg_length=15,
                 ms_n_mels=48,

                 cnn_model='adapt',
                 cnn_c_out_1=16,
                 cnn_c_out_2=32,
                 cnn_c_out_3=64,
                 cnn_kernel_size=3,
                 cnn_dropout=0.2,
                 cnn_pool_1=[24, 7],
                 cnn_pool_2=[12, 5],
                 cnn_pool_3=[6, 3],
                 cnn_fc_out_h=None,

                 td='self_att',
                 td_sa_d_model=64,
                 td_sa_nhead=1,
                 td_sa_pos_enc=None,
                 td_sa_num_layers=2,
                 td_sa_h=64,
                 td_sa_dropout=0.1,
                 td_lstm_h=128,
                 td_lstm_num_layers=1,
                 td_lstm_dropout=0,
                 td_lstm_bidirectional=True,

                 td_2='skip',
                 td_2_sa_d_model=None,
                 td_2_sa_nhead=None,
                 td_2_sa_pos_enc=None,
                 td_2_sa_num_layers=None,
                 td_2_sa_h=None,
                 td_2_sa_dropout=None,
                 td_2_lstm_h=None,
                 td_2_lstm_num_layers=None,
                 td_2_lstm_dropout=None,
                 td_2_lstm_bidirectional=None,

                 pool='att',
                 pool_att_h=128,
                 pool_att_dropout=0.1,

                 ):
        super().__init__()

        self.name = 'NISQA_DIM'

        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1,
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,
        )

        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
        )

        self.time_dependency_2 = TimeDependency(
            input_size=self.time_dependency.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
        )

        pool = Pooling(
            self.time_dependency.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
        )

        self.pool_layers = self._get_clones(pool, 5)

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, x, n_wins, n_wins_cpu):
        x = self.cnn(x, n_wins_cpu)
        x, n_wins = self.time_dependency(x, n_wins)
        x, n_wins = self.time_dependency_2(x, n_wins)
        out = [mod(x, n_wins) for mod in self.pool_layers]
        out = torch.cat(out, dim=1)

        return out


# %% Framewise
class Framewise(nn.Module):
    '''
    Framewise: The main framewise module. It loads either a CNN or feed-forward
    network for framewise modelling of the Mel-spec segments. This module can
    also be skipped by loading the SkipCNN module. There are two CNN modules
    available. AdaptCNN with adaptive maxpooling and the StandardCNN module.
    However, they could also be replaced with new modules, such as PyTorch
    implementations of ResNet or Alexnet.
    '''

    def __init__(
            self,
            cnn_model,
            ms_seg_length=15,
            ms_n_mels=48,
            c_out_1=16,
            c_out_2=32,
            c_out_3=64,
            kernel_size=3,
            dropout=0.2,
            pool_1=[24, 7],
            pool_2=[12, 5],
            pool_3=[6, 3],
            fc_out_h=None,
    ):
        super().__init__()

        if cnn_model == 'adapt':
            self.model = AdaptCNN(
                input_channels=1,
                c_out_1=c_out_1,
                c_out_2=c_out_2,
                c_out_3=c_out_3,
                kernel_size=kernel_size,
                dropout=dropout,
                pool_1=pool_1,
                pool_2=pool_2,
                pool_3=pool_3,
                fc_out_h=fc_out_h,
            )
        elif cnn_model == 'standard':
            assert ms_n_mels == 48, "ms_n_mels is {} and should be 48, use adaptive model or change ms_n_mels".format(
                ms_n_mels)
            assert ms_seg_length == 15, "ms_seg_len is {} should be 15, use adaptive model or change ms_seg_len".format(
                ms_seg_length)
            assert ((kernel_size == 3) or (kernel_size == (
            3, 3))), "cnn_kernel_size is {} should be 3, use adaptive model or change cnn_kernel_size".format(
                kernel_size)
            self.model = StandardCNN(
                input_channels=1,
                c_out_1=c_out_1,
                c_out_2=c_out_2,
                c_out_3=c_out_3,
                kernel_size=kernel_size,
                dropout=dropout,
                fc_out_h=fc_out_h,
            )
        elif cnn_model == 'dff':
            self.model = DFF(ms_seg_length, ms_n_mels, dropout, fc_out_h)
        elif (cnn_model is None) or (cnn_model == 'skip'):
            self.model = SkipCNN(ms_seg_length, ms_n_mels, fc_out_h)
        else:
            raise NotImplementedError('Framwise model not available')

    def forward(self, x, n_wins):
        (bs, length, channels, height, width) = x.shape
        x_packed = pack_padded_sequence(
            x,
            n_wins,
            batch_first=True,
            enforce_sorted=False
        )
        x = self.model(x_packed.data)
        x = x_packed._replace(data=x)
        x, _ = pad_packed_sequence(
            x,
            batch_first=True,
            padding_value=0.0,
            total_length=n_wins.max())
        return x


class SkipCNN(nn.Module):
    '''
    SkipCNN: Can be used to skip the framewise modelling stage and directly
    apply an LSTM or Self-Attention network.
    '''

    def __init__(
            self,
            cnn_seg_length,
            ms_n_mels,
            fc_out_h
    ):
        super().__init__()

        self.name = 'SkipCNN'
        self.cnn_seg_length = cnn_seg_length
        self.ms_n_mels = ms_n_mels
        self.fan_in = cnn_seg_length * ms_n_mels
        self.bn = nn.BatchNorm2d(1)

        if fc_out_h is not None:
            self.linear = nn.Linear(self.fan_in, fc_out_h)
            self.fan_out = fc_out_h
        else:
            self.linear = nn.Identity()
            self.fan_out = self.fan_in

    def forward(self, x):
        x = self.bn(x)
        x = x.view(-1, self.fan_in)
        x = self.linear(x)
        return x


class DFF(nn.Module):
    '''
    DFF: Deep Feed-Forward network that was used as baseline framwise model as
    comparision to the CNN.
    '''

    def __init__(self,
                 cnn_seg_length,
                 ms_n_mels,
                 dropout,
                 fc_out_h=4096,
                 ):
        super().__init__()
        self.name = 'DFF'

        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h
        self.fan_out = fc_out_h

        self.cnn_seg_length = cnn_seg_length
        self.ms_n_mels = ms_n_mels
        self.fan_in = cnn_seg_length * ms_n_mels

        self.lin1 = nn.Linear(self.fan_in, self.fc_out_h)
        self.lin2 = nn.Linear(self.fc_out_h, self.fc_out_h)
        self.lin3 = nn.Linear(self.fc_out_h, self.fc_out_h)
        self.lin4 = nn.Linear(self.fc_out_h, self.fc_out_h)

        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d(self.fc_out_h)
        self.bn3 = nn.BatchNorm1d(self.fc_out_h)
        self.bn4 = nn.BatchNorm1d(self.fc_out_h)
        self.bn5 = nn.BatchNorm1d(self.fc_out_h)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.bn1(x)
        x = x.view(-1, self.fan_in)

        x = F.relu(self.bn2(self.lin1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.lin2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.lin3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.lin4(x)))

        return x


class AdaptCNN(nn.Module):
    '''
    AdaptCNN: CNN with adaptive maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module is more flexible
    than the StandardCNN that requires a fixed input dimension of 48x15.
    '''

    def __init__(self,
                 input_channels,
                 c_out_1,
                 c_out_2,
                 c_out_3,
                 kernel_size,
                 dropout,
                 pool_1,
                 pool_2,
                 pool_3,
                 fc_out_h=20,
                 ):
        super().__init__()
        self.name = 'CNN_adapt'

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.pool_3 = pool_3
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.dropout = nn.Dropout2d(p=self.dropout_rate)

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        # Set kernel width of last conv layer to last pool width to
        # downsample width to one.
        self.kernel_size_last = (self.kernel_size[0], self.pool_3[1])

        # kernel_size[1]=1 can be used for seg_length=1 -> corresponds to
        # 1D conv layer, no width padding needed.
        if self.kernel_size[1] == 1:
            self.cnn_pad = (1, 0)
        else:
            self.cnn_pad = (1, 1)

        self.conv1 = nn.Conv2d(
            self.input_channels,
            self.c_out_1,
            self.kernel_size,
            padding=self.cnn_pad)

        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(
            self.conv1.out_channels,
            self.c_out_2,
            self.kernel_size,
            padding=self.cnn_pad)

        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(
            self.conv2.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=self.cnn_pad)

        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)

        self.conv4 = nn.Conv2d(
            self.conv3.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=self.cnn_pad)

        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)

        self.conv5 = nn.Conv2d(
            self.conv4.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=self.cnn_pad)

        self.bn5 = nn.BatchNorm2d(self.conv5.out_channels)

        self.conv6 = nn.Conv2d(
            self.conv5.out_channels,
            self.c_out_3,
            self.kernel_size_last,
            padding=(1, 0))

        self.bn6 = nn.BatchNorm2d(self.conv6.out_channels)

        if self.fc_out_h:
            self.fc = nn.Linear(self.conv6.out_channels * self.pool_3[0], self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.pool_3[0])

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_1))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_2))

        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_3))

        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.view(-1, self.conv6.out_channels * self.pool_3[0])

        if self.fc_out_h:
            x = self.fc(x)
        return x


class StandardCNN(nn.Module):
    '''
    StandardCNN: CNN with fixed maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module requires a fixed
    input dimension of 48x15.
    '''

    def __init__(
            self,
            input_channels,
            c_out_1,
            c_out_2,
            c_out_3,
            kernel_size,
            dropout,
            fc_out_h=None
    ):
        super().__init__()

        self.name = 'CNN_standard'

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_size = 2
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.output_width = 2  # input width 15 pooled 3 times
        self.output_height = 6  # input height 48 pooled 3 times

        self.dropout = nn.Dropout2d(p=self.dropout_rate)

        self.pool_first = nn.MaxPool2d(
            self.pool_size,
            stride=self.pool_size,
            padding=(0, 1))

        self.pool = nn.MaxPool2d(
            self.pool_size,
            stride=self.pool_size,
            padding=0)

        self.conv1 = nn.Conv2d(
            self.input_channels,
            self.c_out_1,
            self.kernel_size,
            padding=1)

        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(
            self.conv1.out_channels,
            self.c_out_2,
            self.kernel_size,
            padding=1)

        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(
            self.conv2.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=1)

        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)

        self.conv4 = nn.Conv2d(
            self.conv3.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=1)

        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)

        self.conv5 = nn.Conv2d(
            self.conv4.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=1)

        self.bn5 = nn.BatchNorm2d(self.conv5.out_channels)

        self.conv6 = nn.Conv2d(
            self.conv5.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=1)

        self.bn6 = nn.BatchNorm2d(self.conv6.out_channels)

        if self.fc_out_h:
            self.fc_out = nn.Linear(self.conv6.out_channels * self.output_height * self.output_width, self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.output_height * self.output_width)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool_first(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)

        x = F.relu(self.bn6(self.conv6(x)))

        x = x.view(-1, self.conv6.out_channels * self.output_height * self.output_width)

        if self.fc_out_h:
            x = self.fc_out(x)

        return x


# %% Time Dependency
class TimeDependency(nn.Module):
    '''
    TimeDependency: The main time-dependency module. It loads either an LSTM
    or self-attention network for time-dependency modelling of the framewise
    features. This module can also be skipped.
    '''

    def __init__(self,
                 input_size,
                 td='self_att',
                 sa_d_model=512,
                 sa_nhead=8,
                 sa_pos_enc=None,
                 sa_num_layers=6,
                 sa_h=2048,
                 sa_dropout=0.1,
                 lstm_h=128,
                 lstm_num_layers=1,
                 lstm_dropout=0,
                 lstm_bidirectional=True,
                 ):
        super().__init__()

        if td == 'self_att':
            self.model = SelfAttention(
                input_size=input_size,
                d_model=sa_d_model,
                nhead=sa_nhead,
                pos_enc=sa_pos_enc,
                num_layers=sa_num_layers,
                sa_h=sa_h,
                dropout=sa_dropout,
                activation="relu"
            )
            self.fan_out = sa_d_model

        elif td == 'lstm':
            self.model = LSTM(
                input_size,
                lstm_h=lstm_h,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
                bidirectional=lstm_bidirectional,
            )
            self.fan_out = self.model.fan_out

        elif (td is None) or (td == 'skip'):
            self.model = self._skip
            self.fan_out = input_size
        else:
            raise NotImplementedError('Time dependency option not available')

    def _skip(self, x, n_wins):
        return x, n_wins

    def forward(self, x, n_wins):
        x, n_wins = self.model(x, n_wins)
        return x, n_wins


class LSTM(nn.Module):
    '''
    LSTM: The main LSTM module that can be used as a time-dependency model.
    '''

    def __init__(self,
                 input_size,
                 lstm_h=128,
                 num_layers=1,
                 dropout=0.1,
                 bidirectional=True
                 ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_h,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        self.fan_out = num_directions * lstm_h

    def forward(self, x, n_wins):

        x = pack_padded_sequence(
            x,
            n_wins,
            batch_first=True,
            enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]

        x, _ = pad_packed_sequence(
            x,
            batch_first=True,
            padding_value=0.0,
            total_length=n_wins.max())

        return x, n_wins


class SelfAttention(nn.Module):
    '''
    SelfAttention: The main SelfAttention module that can be used as a
    time-dependency model.
    '''

    def __init__(self,
                 input_size,
                 d_model=512,
                 nhead=8,
                 pool_size=3,
                 pos_enc=None,
                 num_layers=6,
                 sa_h=2048,
                 dropout=0.1,
                 activation="relu"
                 ):
        super().__init__()

        encoder_layer = SelfAttentionLayer(d_model, nhead, pool_size, sa_h, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear = nn.Linear(input_size, d_model)

        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        if pos_enc:
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        else:
            self.pos_encoder = nn.Identity()

        self._reset_parameters()

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, n_wins=None):
        src = self.linear(src)
        output = src.transpose(1, 0)
        output = self.norm1(output)
        output = self.pos_encoder(output)

        for mod in self.layers:
            output, n_wins = mod(output, n_wins=n_wins)
        return output.transpose(1, 0), n_wins


class SelfAttentionLayer(nn.Module):
    '''
    SelfAttentionLayer: The SelfAttentionLayer that is used by the
    SelfAttention module.
    '''

    def __init__(self, d_model, nhead, pool_size=1, sa_h=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, sa_h)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(sa_h, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

    def forward(self, src, n_wins=None):

        if n_wins is not None:
            mask = ~(torch.arange(src.shape[0], device=src.device)[None, :] < n_wins[:, None])
        else:
            mask = None

        src2 = self.self_attn(src, src, src, key_padding_mask=mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src, n_wins


class PositionalEncoding(nn.Module):
    '''
    PositionalEncoding: PositionalEncoding taken from the PyTorch Transformer
    tutorial. Can be applied to the SelfAttention module. However, it did not
    improve the results in previous experiments.
    '''

    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# %% Pooling
class Pooling(nn.Module):
    '''
    Pooling: Main Pooling module. It can load either attention-pooling, average
    pooling, maxpooling, or last-step pooling. In case of bidirectional LSTMs
    last-step-bi pooling should be used instead of last-step pooling.
    '''

    def __init__(self,
                 d_input,
                 output_size=1,
                 pool='att',
                 att_h=None,
                 att_dropout=0,
                 ):
        super().__init__()

        if pool == 'att':
            if att_h is None:
                self.model = PoolAtt(d_input, output_size)
            else:
                self.model = PoolAttFF(d_input, output_size, h=att_h, dropout=att_dropout)
        elif pool == 'last_step_bi':
            self.model = PoolLastStepBi(d_input, output_size)
        elif pool == 'last_step':
            self.model = PoolLastStep(d_input, output_size)
        elif pool == 'max':
            self.model = PoolMax(d_input, output_size)
        elif pool == 'avg':
            self.model = PoolAvg(d_input, output_size)
        else:
            raise NotImplementedError('Pool option not available')

    def forward(self, x, n_wins):
        return self.model(x, n_wins)


class PoolLastStepBi(nn.Module):
    '''
    PoolLastStepBi: last step pooling for the case of bidirectional LSTM
    '''

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, n_wins=None):
        x = x.view(x.shape[0], n_wins.max(), 2, x.shape[-1] // 2)
        x = torch.cat(
            (x[torch.arange(x.shape[0]), n_wins.type(torch.long) - 1, 0, :],
             x[:, 0, 1, :]),
            dim=1
        )
        x = self.linear(x)
        return x


class PoolLastStep(nn.Module):
    '''
    PoolLastStep: last step pooling can be applied to any one-directional
    sequence.
    '''

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, n_wins=None):
        x = x[torch.arange(x.shape[0]), n_wins.type(torch.long) - 1]
        x = self.linear(x)
        return x


class PoolAtt(torch.nn.Module):
    '''
    PoolAtt: Attention-Pooling module.
    '''

    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear1 = nn.Linear(d_input, 1)
        self.linear2 = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):
        att = self.linear1(x)

        att = att.transpose(2, 1)
        mask = torch.arange(att.shape[2], device=x.device)[None, :] < n_wins[:, None]
        att[~mask.unsqueeze(1)] = torch.inf
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)

        x = self.linear2(x)

        return x


class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''

    def __init__(self, d_input, output_size, h, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_input, h)
        self.linear2 = nn.Linear(h, 1)

        self.linear3 = nn.Linear(d_input, output_size)

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_wins):
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2, 1)
        mask = torch.arange(att.shape[2], device=x.device)[None, :] < n_wins[:, None]
        att[~mask.unsqueeze(1)] = torch.inf
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)

        x = self.linear3(x)

        return x


class PoolAvg(torch.nn.Module):
    '''
    PoolAvg: Average pooling that consideres masked time-steps.
    '''

    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < n_wins[:, None]
        mask = ~mask.unsqueeze(2)
        x.masked_fill_(mask, 0)

        x = torch.div(x.sum(1), n_wins.unsqueeze(1))

        x = self.linear(x)

        return x


class PoolMax(torch.nn.Module):
    '''
    PoolMax: Max-pooling that consideres masked time-steps.
    '''

    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < n_wins[:, None]
        mask = ~mask.unsqueeze(2)
        x.masked_fill_(mask, torch.inf)

        x = x.max(1)[0]

        x = self.linear(x)

        return x


def segment_specs3(x, seg_length, seg_hop=1, max_length=None):
    """
    args:
        x: (bs, n_mels, n_frames)
    """
    if seg_length % 2 == 0:
        raise ValueError('seg_length must be odd! (seg_lenth={})'.format(seg_length))

    n_wins = x.shape[2] - (seg_length - 1)
    if n_wins < 1:
        raise ValueError(
            f"Sample too short. Only {x.shape[2]} windows available but seg_length={seg_length}. "
            f"Consider zero padding the audio sample."
        )

    idx1 = torch.arange(seg_length, device=x.device)
    idx2 = torch.arange(n_wins, device=x.device)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    x = x.transpose(2, 1)[:, idx3].unsqueeze(2).transpose(4, 3)

    if seg_hop > 1:
        x = x[:, ::seg_hop]
        n_wins = int(np.ceil(n_wins / seg_hop))

    if max_length is not None:
        if max_length < n_wins:
            raise ValueError(
                'n_wins {} > max_length {}. Increase max window length ms_max_segments!'.format(n_wins,
                                                                                                 max_length))
        x_padded = torch.zeros((x.shape[0], x.shape[1], max_length, x.shape[3]))
        x_padded[:, :, :n_wins, :] = x
        x = x_padded

    n_wins_gpu = torch.tensor([n_wins] * x.shape[0], device=x.device)
    n_wins_cpu = torch.tensor([n_wins] * x.shape[0], device='cpu')

    return x, n_wins_gpu, n_wins_cpu


this_dir = os.path.dirname(os.path.abspath(__file__))
NISQA_MODEL_PATH = os.path.join(this_dir, 'nisqa_models', 'nisqa.tar')


class NISQA_DIM_MOS(nn.Module):
    def __init__(self):
        super().__init__()
        ckpt = torch.load(NISQA_MODEL_PATH, map_location='cpu')
        args = ckpt['args']
        args["dim"] = True
        args["csv_mos_train"] = None  # column names hardcoded for dim models
        args["csv_mos_val"] = None

        model_args = {
            "ms_seg_length": args["ms_seg_length"],
            "ms_n_mels": args["ms_n_mels"],
            "cnn_model": args["cnn_model"],
            "cnn_c_out_1": args["cnn_c_out_1"],
            "cnn_c_out_2": args["cnn_c_out_2"],
            "cnn_c_out_3": args["cnn_c_out_3"],
            "cnn_kernel_size": args["cnn_kernel_size"],
            "cnn_dropout": args["cnn_dropout"],
            "cnn_pool_1": args["cnn_pool_1"],
            "cnn_pool_2": args["cnn_pool_2"],
            "cnn_pool_3": args["cnn_pool_3"],
            "cnn_fc_out_h": args["cnn_fc_out_h"],
            "td": args["td"],
            "td_sa_d_model": args["td_sa_d_model"],
            "td_sa_nhead": args["td_sa_nhead"],
            "td_sa_pos_enc": args["td_sa_pos_enc"],
            "td_sa_num_layers": args["td_sa_num_layers"],
            "td_sa_h": args["td_sa_h"],
            "td_sa_dropout": args["td_sa_dropout"],
            "td_lstm_h": args["td_lstm_h"],
            "td_lstm_num_layers": args["td_lstm_num_layers"],
            "td_lstm_dropout": args["td_lstm_dropout"],
            "td_lstm_bidirectional": args["td_lstm_bidirectional"],
            "td_2": args["td_2"],
            "td_2_sa_d_model": args["td_2_sa_d_model"],
            "td_2_sa_nhead": args["td_2_sa_nhead"],
            "td_2_sa_pos_enc": args["td_2_sa_pos_enc"],
            "td_2_sa_num_layers": args["td_2_sa_num_layers"],
            "td_2_sa_h": args["td_2_sa_h"],
            "td_2_sa_dropout": args["td_2_sa_dropout"],
            "td_2_lstm_h": args["td_2_lstm_h"],
            "td_2_lstm_num_layers": args["td_2_lstm_num_layers"],
            "td_2_lstm_dropout": args["td_2_lstm_dropout"],
            "td_2_lstm_bidirectional": args["td_2_lstm_bidirectional"],
            "pool": args["pool"],
            "pool_att_h": args["pool_att_h"],
            "pool_att_dropout": args["pool_att_dropout"],
        }

        self.model = NISQA_DIM(**model_args)
        self.model.load_state_dict(
            ckpt["model_state_dict"], strict=True
        )
        self.model = self.model.eval()

        self.sample_rate = args["ms_sr"] if args["ms_sr"] else 48000

        n_fft = args["ms_n_fft"]
        hop_length = int(args["ms_hop_length"] * self.sample_rate)
        win_length = int(args["ms_win_length"] * self.sample_rate)
        n_mels = args["ms_n_mels"]
        fmax = args["ms_fmax"]
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_max=fmax,
            center=True,
            pad_mode="reflect",
            power=1.0,
            f_min=0.0,
            mel_scale="slaney",
            norm="slaney",
            normalized=True,
        )

        amin = 1e-4
        ref_value = 1.0
        db_multiplier = math.log10(max(amin, ref_value))
        self.amp_to_db = partial(torchaudio.functional.amplitude_to_DB, multiplier=20,
                                 amin=1e-4, db_multiplier=db_multiplier, top_db=80.0)

        self.seg_length = args["ms_seg_length"]
        self.seg_hop_length = args["ms_seg_hop_length"]
        self.max_segments = args["ms_max_segments"]

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, inf, fs, **kwargs):
        """
        x: (batch_size, n_samples)
        """
        x = torchaudio.functional.resample(inf, orig_freq=fs, new_freq=self.sample_rate)
        mel_spec = self.mel_spec(x)  # (batch_size, n_mels, n_frames)
        mel_spec = self.amp_to_db(mel_spec)
        spec_seg, wins, wins_cpu = segment_specs3(x=mel_spec, seg_length=self.seg_length, seg_hop=self.seg_hop_length)
        nisqa_mos = self.model(spec_seg, wins, wins_cpu)[:, 0]
        return nisqa_mos


if __name__ == '__main__':
    model = NISQA_DIM_MOS()
    x = torch.randn(2, 48000)
    out = model(x, 48000)
    print(out)
    print(out.shape)
