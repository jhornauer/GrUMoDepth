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
Source: modified from https://github.com/mattpoggi/mono-uncertainty
"""
from __future__ import absolute_import, division, print_function

import os
import argparse
from monodepth2.options import MonodepthOptions


# Extended set of options
class UncertaintyOptions(MonodepthOptions):

    def __init__(self):
        super(UncertaintyOptions, self).__init__()

        self.parser.add_argument("--custom_scale", type=float, default=100., help="custom scale factor for depth maps")

        self.parser.add_argument("--eval_uncert", help="if set enables uncertainty evaluation", action="store_true")
        self.parser.add_argument("--log",
                                 help="if set, adds the variance output to monodepth2 according to log-likelihood "
                                      "maximization technique",
                                 action="store_true")
        self.parser.add_argument("--repr", help="if set, adds the Repr output to monodepth2", action="store_true")

        # self.parser.add_argument("--dropout", help="if set enables dropout inference", action="store_true")

        self.parser.add_argument("--bootstraps", type=int, default=1,
                                 help="if > 1, loads multiple checkpoints from different trainings to build a "
                                      "bootstrapped ensamble")
        self.parser.add_argument("--snapshots", type=int, default=1,
                                 help="if > 1, loads the last N checkpoints to build a snapshots ensemble")

        self.parser.add_argument("--output_dir", type=str, default="output",
                                 help="output directory for predicted depth and uncertainty maps")
        self.parser.add_argument("--qual", help="if set save colored depth and uncertainty maps", action="store_true")

        ## additional gradient options 
        self.parser.add_argument("--grad", help="if set will calculate the gradients for uncertainty evaluation",
                                 action="store_true")
        self.parser.add_argument("--gloss", type=str, default="sq",
                                 help="squared difference [sq] "
                                      "or only uncertainty loss (only possible if w > 0.0) [none]"
                                      "or variance over augmentations [var]")
        self.parser.add_argument("--gref", type=str,
                                 help="reference image for gradient calculation: no grad reference [none]"
                                      "flipped image [flip], grayscale [gray], additive Gaussian noise [noise], "
                                      "ground truth (only for nyu) [gt], rotation [rot]")
        self.parser.add_argument("--angle", help="angle in degree if rotation is used as reference", type=int,
                                 default=10)
        self.parser.add_argument("--var_aug", help="use variance over multiple augmentations", action="store_true")
        self.parser.add_argument("--ext_layer", type=int, nargs="+", default=[6])
        self.parser.add_argument("--gred", type=str, default='max',
                                 help="method to reduce the gradients over the channels: [sum], [mean], [max], [norm]")
        self.parser.add_argument("--w", type=float, default=0.0, help="weighting of uncertainty loss term")

        self.parser.add_argument("--infer_dropout", help="if set will apply dropout only during inference",
                                 action="store_true")
        self.parser.add_argument("--infer_p", type=float, default=0.01, help="infer dropout probability")

        ##  additional options for supervised depth
        self.parser.add_argument("--save_depth_map", help="if set will save the detph maps", action="store_true")
        self.parser.add_argument("--save_uncert_map", help="if set will save the uncertainty maps", action="store_true")
        self.parser.add_argument("--save_error_map", help="if set will save the error maps", action="store_true")
        self.parser.add_argument("--save_rgb", help="if set will save the rgb images", action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
