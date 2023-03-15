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

import copy
import warnings

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch

import monodepth2
from monodepth2.options import MonodepthOptions
from monodepth2.layers import disp_to_depth
from monodepth2.utils import readlines
from extended_options import UncertaintyOptions
import progressbar
from eval_utils import compute_eigen_errors_visu, compute_eigen_errors, compute_aucs

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "monodepth2/splits")

# Real-world scale factor (see Monodepth2)
STEREO_SCALE_FACTOR = 5.4
uncertainty_metrics = ["abs_rel", "rmse", "a1"]


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3 
    MAX_DEPTH = opt.max_depth

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Loading 16 bit predictions from {}".format(opt.ext_disp_to_eval))
    pred_disps = []
    pred_uncerts = []
    for i in range(len(gt_depths)):
        img = cv2.imread(opt.ext_disp_to_eval+'/disp/%06d_10.png'%i,-1)
        src = img / 256. / (0.58*gt_depths[i].shape[1]) * 10
        pred_disps.append(src)
        if opt.eval_uncert:
            if opt.grad:
                folder_name = "uncert_" + opt.gref + "_" + opt.gloss
                if opt.w != 0.0:
                    folder_name = folder_name + "_weight" + str(opt.w)
                folder_name = folder_name + "_layer_" + "_".join(str(x) for x in opt.ext_layer)
            elif opt.infer_dropout:
                folder_name = "uncert_p_" + str(opt.infer_p)
            else: 
                folder_name = "uncert"
            uncert = cv2.imread(opt.ext_disp_to_eval+'/' + folder_name + '/%06d_10.png'%i,-1) / 256.
            pred_uncerts.append(uncert)

    pred_disps = np.array(pred_disps)

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    errors_abs_rel = []
    errors_rmse = []

    # dictionary with accumulators for each metric
    aucs = {"abs_rel":[], "rmse":[], "a1":[]}
    curves = {"abs_rel": [], "rmse": [], "a1":[]}

    pred_width, pred_height = pred_disps[0].shape[0], pred_disps[0].shape[1]
    bar = progressbar.ProgressBar(max_value=len(gt_depths))
    for i in range(len(gt_depths)):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        bar.update(i)

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        gt_depth_visu = copy.deepcopy(gt_depth)
        pred_depth_visu = copy.deepcopy(pred_depth)
        mask_visu = gt_depth > 0
        gt_depth_visu[~mask_visu] = MIN_DEPTH
        pred_depth_visu[~mask_visu] = MIN_DEPTH

        # get error maps
        tmp_abs_rel, tmp_rmse, tmp_a1 = compute_eigen_errors_visu(gt_depth_visu, pred_depth_visu, mask_visu)
        errors_abs_rel.append(tmp_abs_rel)
        errors_rmse.append(tmp_rmse)

        if opt.eval_uncert:
            pred_uncert = pred_uncerts[i]
            pred_uncert = cv2.resize(pred_uncert, (gt_width, gt_height))

        if opt.eval_split == "eigen":
        
            # traditional eigen crop
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
        
            # just mask out invalid depths
            mask = (gt_depth > 0)

        # apply masks
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        if opt.eval_uncert:
            pred_uncert = pred_uncert[mask]

        # apply scale factor and depth cap
        pred_depth *= opt.pred_depth_scale_factor
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        # get Eigen's metrics
        errors.append(compute_eigen_errors(gt_depth, pred_depth))
        if opt.eval_uncert:
        
            # get uncertainty metrics (AUSE and AURG)
            scores, spars_plots = compute_aucs(gt_depth, pred_depth, pred_uncert)

            # append AUSE and AURG to accumulators
            [aucs[m].append(scores[m]) for m in uncertainty_metrics]

            [curves[m].append(spars_plots[m]) for m in uncertainty_metrics]

    # compute mean depth metrics and print
    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    if opt.eval_uncert:
    
        # compute mean uncertainty metrics and print
        for m in uncertainty_metrics:
            aucs[m] = np.array(aucs[m]).mean(0)
            print("\n  " + ("{:>8} | " * 6).format("abs_rel", "", "rmse", "", "a1", ""))
        print("  " + ("{:>8} | " * 6).format("AUSE", "AURG", "AUSE", "AURG", "AUSE", "AURG"))
        print(("&{:8.3f}  " * 6).format(*aucs["abs_rel"].tolist()+aucs["rmse"].tolist()+aucs["a1"].tolist()) + "\\\\")

    errors_abs_rel = np.array(errors_abs_rel)
    errors_rmse = np.array(errors_rmse)

    # save sparsification plots
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    pickle.dump(curves, open(os.path.join(opt.output_dir, "spars_plots.pkl"), "wb"))

    if opt.save_error_map:
        if not os.path.exists(opt.output_dir):
            os.mkdir(opt.output_dir)
        if not os.path.exists(os.path.join(opt.output_dir, "abs_rel")):
            os.makedirs(os.path.join(opt.output_dir, "abs_rel"))
        if not os.path.exists(os.path.join(opt.output_dir, "rmse")):
            os.makedirs(os.path.join(opt.output_dir, "rmse"))

        print("--> Saving qualitative error maps: abs rel")
        bar = progressbar.ProgressBar(max_value=len(errors_abs_rel))
        for i in range(len(errors_abs_rel)):
            bar.update(i)
            # save colored depth maps
            plt.imsave(os.path.join(opt.output_dir, "abs_rel", '%06d_10.png' % i),
                       cv2.resize(errors_abs_rel[i], (pred_height, pred_width)), cmap='hot')

        print("--> Saving qualitative error maps: rmse")
        bar = progressbar.ProgressBar(max_value=len(errors_rmse))
        for i in range(len(errors_rmse)):
            bar.update(i)
            # save colored depth maps
            plt.imsave(os.path.join(opt.output_dir, "rmse", '%06d_10.png' % i),
                       cv2.resize(errors_rmse[i], (pred_height, pred_width)), cmap='hot')

    # see you next time!
    print("\n-> Done!")


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    options = UncertaintyOptions()
    evaluate(options.parse())
