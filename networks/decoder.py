# # Monodepth2 extended to estimate depth and uncertainty
# #
# # This software is licensed under the terms of the Monodepth2 licence
# # which allows for non-commercial use only, the full terms of which are
# # available at https://github.com/nianticlabs/monodepth2/blob/master/LICENSE
# """
# Source: modified from https://github.com/mattpoggi/mono-uncertainty
# """

#
# MIT License
#
# Copyright (c) 2020 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Source: https://github.com/mattpoggi/mono-uncertainty/blob/master/networks/decoder.py
--> Further extended to use inference only dropout
"""


from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from monodepth2.layers import *


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)


class DepthUncertaintyDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, uncert=False, dropout=False,
                 infer_dropout=False, infer_p=0.2):
        super(DepthUncertaintyDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.p = 0.2
        self.uncert = uncert
        self.dropout = dropout

        self.infer_dropout = infer_dropout
        self.infer_p = infer_p

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            if self.uncert:
                self.convs[("uncertconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

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
                self.outputs[("dispconv", i)] = self.convs[("dispconv", i)]
                disps = self.convs[("dispconv", i)](x)
                self.outputs[("disp", i)] = self.sigmoid(disps)

                if self.uncert:
                    uncerts = self.convs[("uncertconv", i)](x)
                    self.outputs[("uncert", i)] = uncerts

        return self.outputs
