# # Monodepth2 extended to estimate depth and uncertainty
# #
# # This software is licensed under the terms of the Monodepth2 licence
# # which allows for non-commercial use only, the full terms of which are
# # available at https://github.com/nianticlabs/monodepth2/blob/master/LICENSE
# """
# Source: modified from https://github.com/mattpoggi/mono-uncertainty
# """
"""
Source: https://github.com/mattpoggi/mono-uncertainty/blob/master/networks/decoder.py
-- further modified for supervised training: last sigmoid is removed to directly estimate the depth instead of the scaled disparity
"""

import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from monodepth2.layers import *


class DepthUncertaintyDecoder_Supervised(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, dropout=False, uncert=False,
                 infer_dropout=False, infer_p=0.01):
        super(DepthUncertaintyDecoder_Supervised, self).__init__()
        self.dropout = dropout
        self.p = 0.2
        self.uncert = uncert
        self.infer_dropout = infer_dropout
        self.infer_p = infer_p

        self.output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales
        self.encoder_channels = num_ch_enc
        self.decoder_channels = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_in = self.encoder_channels[-1] if i == 4 else self.decoder_channels[i + 1]
            num_out = self.decoder_channels[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_in, num_out)

            # upconv_1
            num_in = self.decoder_channels[i]
            if self.use_skips and i > 0:
                num_in += self.encoder_channels[i - 1]
            num_out = self.decoder_channels[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_in, num_out)

        for s in self.scales:
            self.convs[("depthconv", s)] = Conv3x3(self.decoder_channels[s], self.output_channels)
            if self.uncert:
                self.convs[("uncertconv", s)] = Conv3x3(self.decoder_channels[s], self.output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)

            if self.dropout:
                x = F.dropout2d(x, p=self.p, training=True)

            if self.infer_dropout:
                x = F.dropout2d(x, p=self.infer_p, training=True)

            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if self.dropout:
                x = F.dropout2d(x, p=self.p, training=True)

            if self.infer_dropout:
                x = F.dropout2d(x, p=self.infer_p, training=True)

            if i in self.scales:
                self.outputs[("depth", i)] = self.convs[("depthconv", i)](x)
                if self.uncert:
                    uncerts = self.convs[("uncertconv", i)](x)
                    self.outputs[("uncert", i)] = uncerts

        return self.outputs