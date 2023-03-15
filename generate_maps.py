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

-- added gradient-based uncertainty (grad)
-- added inference only dropout (infer_dropout)
-- added variance over different augmentations (var_aug)

"""

from __future__ import absolute_import, division, print_function

import time
import warnings

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

import monodepth2
import monodepth2.kitti_utils as kitti_utils
from monodepth2.layers import *
from monodepth2.utils import *
from extended_options import *
import monodepth2.datasets as datasets
import monodepth2.networks as legacy
import networks
import progressbar
import matplotlib.pyplot as plt

from gradients import *
from torchvision import transforms

import sys

splits_dir = os.path.join(os.path.dirname(__file__), "monodepth2/splits")


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def get_mono_ratio(disp, gt):
    """Returns the median scaling factor
    """
    mask = gt > 0
    return np.median(gt[mask]) / np.median(cv2.resize(1 / disp, (gt.shape[1], gt.shape[0]))[mask])


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    opt.batch_size = 1

    assert sum((opt.eval_mono, opt.eval_stereo, opt.no_eval)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono, --eval_stereo, --custom_run"
    assert sum((opt.log, opt.repr)) < 2, \
        "Please select only one between LR and LOG by setting --repr or --log"
    assert opt.bootstraps == 1 or opt.snapshots == 1, \
        "Please set only one of --bootstraps or --snapshots to be major than 1"

    # get the number of networks
    nets = max(opt.bootstraps, opt.snapshots)
    do_uncert = (opt.log or opt.repr or opt.dropout or opt.post_process or opt.bootstraps > 1 or opt.snapshots > 1
                 or opt.grad or opt.infer_dropout or opt.var_aug)

    print("-> Beginning inference...")

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    if opt.bootstraps > 1:
        # prepare multiple checkpoint paths from different trainings
        encoder_path = [os.path.join(opt.load_weights_folder, "boot_%d" % i, "weights_19", "encoder.pth") for i in
                        range(1, opt.bootstraps + 1)]
        decoder_path = [os.path.join(opt.load_weights_folder, "boot_%d" % i, "weights_19", "depth.pth") for i in
                        range(1, opt.bootstraps + 1)]
        encoder_dict = [torch.load(encoder_path[i]) for i in range(opt.bootstraps)]
        height = encoder_dict[0]['height']
        width = encoder_dict[0]['width']

    elif opt.snapshots > 1:
        # prepare multiple checkpoint paths from the same training
        encoder_path = [os.path.join(opt.load_weights_folder, "weights_%d" % i, "encoder.pth") for i in
                        range(opt.num_epochs - opt.snapshots, opt.num_epochs)]
        decoder_path = [os.path.join(opt.load_weights_folder, "weights_%d" % i, "depth.pth") for i in
                        range(opt.num_epochs - opt.snapshots, opt.num_epochs)]
        encoder_dict = [torch.load(encoder_path[i]) for i in range(opt.snapshots)]
        height = encoder_dict[0]['height']
        width = encoder_dict[0]['width']

    else:
        # prepare just a single path
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path)
        height = encoder_dict['height']
        width = encoder_dict['width']

    img_ext = '.png' if opt.png else '.jpg'
    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                       height, width,
                                       [0], 4, is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    if nets > 1:

        # load multiple encoders and decoders 
        encoder = [legacy.ResnetEncoder(opt.num_layers, False) for i in range(nets)]
        depth_decoder = [
            networks.DepthUncertaintyDecoder(encoder[i].num_ch_enc, num_output_channels=1,
                                             uncert=(opt.log or opt.repr),
                                             dropout=opt.dropout) for i in range(nets)]

        model_dict = [encoder[i].state_dict() for i in range(nets)]
        for i in range(nets):
            encoder[i].load_state_dict({k: v for k, v in encoder_dict[i].items() if k in model_dict[i]})
            depth_decoder[i].load_state_dict(torch.load(decoder_path[i]))
            encoder[i].cuda()
            encoder[i].eval()
            depth_decoder[i].cuda()
            depth_decoder[i].eval()

    else:

        # load a single encoder and decoder
        encoder = legacy.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthUncertaintyDecoder(encoder.num_ch_enc, num_output_channels=1,
                                                         uncert=(opt.log or opt.repr or opt.uncert),
                                                         dropout=opt.dropout)

        if opt.infer_dropout:
            # load separate depth deocder if dropout is onl applied during inference
            depth_decoder_drop = networks.DepthUncertaintyDecoder(encoder.num_ch_enc, num_output_channels=1,
                                                                  uncert=(opt.log or opt.repr or opt.uncert),
                                                                  dropout=opt.dropout, infer_dropout=opt.infer_dropout,
                                                                  infer_p=opt.infer_p)
            depth_decoder_drop.load_state_dict(torch.load(decoder_path))
            depth_decoder_drop.cuda()
            depth_decoder_drop.eval()
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

    # accumulators for depth and uncertainties
    pred_disps = []
    pred_uncerts = []

    if opt.grad:
        ext_layer = ['decoder.0.conv', 'decoder.1.conv', 'decoder.2.conv', 'decoder.3.conv', 'decoder.4.conv',
                     'decoder.5.conv', 'decoder.6.conv', 'decoder.7.conv', 'decoder.8.conv', 'decoder.9.conv',
                     'decoder.10.conv']
        layer_list = [ext_layer[layer_idx] for layer_idx in opt.ext_layer]
        gradient_extractor = Gradient_Analysis(depth_decoder, layer_list, height, width, opt.gred)
        print("-> Extract gradients from model for uncertainty estimation")

        bwd_time = 0
        n_samples = 0

        if opt.gloss not in ["sq", "none", "var"]:
            raise NotImplementedError

        for i, data in enumerate(dataloader):
            rgb_img = data[("color", 0, 0)].cuda()
            if opt.gref == "flip":
                # Post-processed results require each image to have two forward passes
                ref_img = torch.flip(rgb_img, [3])
                with torch.no_grad():
                    output = depth_decoder(encoder(ref_img))
                ref_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                ref_disp = ref_disp.squeeze(1)
                ref_depth = 1 / ref_disp
                ref_depth = ref_depth.cpu().numpy()[:, :, ::-1]
                ref_depth = torch.from_numpy(ref_depth.copy()).cuda()
            elif opt.gref == "var":
                ref_imgs = [torch.flip(rgb_img, [3]), transforms.Grayscale(num_output_channels=3)(rgb_img),
                            rgb_img + torch.normal(0.0, 0.01, rgb_img.size()).cuda(),
                            transforms.functional.rotate(rgb_img, 10)]
                ref_depths = []
                with torch.no_grad():
                    for j, input in enumerate(ref_imgs):
                        output = depth_decoder(encoder(input))
                        ref_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                        if j == 3:
                            ref_disp = transforms.functional.rotate(ref_disp, -10)
                        ref_disp = ref_disp.squeeze(1)
                        ref_depth = 1 / ref_disp
                        if j == 0:
                            ref_depth = ref_depth.cpu().numpy()[:, :, ::-1]
                            ref_depth = torch.from_numpy(ref_depth.copy()).cuda()
                        ref_depths.append(ref_depth)
            output = gradient_extractor(encoder(rgb_img))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.squeeze(1)
            pred_depth = 1 / pred_disp

            n_samples += rgb_img.shape[0]

            loss = 0
            if opt.gloss == "var":
                loss = torch.var(torch.cat([pred_depth, ref_depths[0], ref_depths[1], ref_depths[2], ref_depths[3]], 0), dim=0)
                loss = torch.mean(loss)
            else:
                if opt.gloss == "sq":
                    depth_diff = squared_difference(pred_depth, ref_depth)
                    loss += torch.mean(depth_diff)
                if opt.uncert and opt.w != 0.0:
                    pred_uncert = output[("uncert", 0)].squeeze(1)
                    uncert = torch.exp(pred_uncert) ** 2
                    loss += (opt.w * torch.mean(uncert))

            start_time = time.time()
            loss.backward()
            stop_time = time.time()

            bwd_time += (stop_time - start_time)
        pred_uncerts = gradient_extractor.get_gradients()
        bwd_time = bwd_time / len(dataloader)
        print('\nAverage backward time: {:.2f} ms'.format(bwd_time * 1000))

    print("-> Computing predictions with size {}x{}".format(width, height))

    fwd_time = 0
    with torch.no_grad():
        bar = progressbar.ProgressBar(max_value=len(dataloader))
        for i, data in enumerate(dataloader):
            input_color = data[("color", 0, 0)].cuda()

            # updating progress bar
            bar.update(i)
            if opt.post_process:
                # post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            if nets > 1:

                # infer multiple predictions from multiple networks
                disps_distribution = []
                uncerts_distribution = []
                for i in range(nets):
                    start_time = time.time()
                    output = depth_decoder[i](encoder[i](input_color))
                    stop_time = time.time()
                    disps_distribution.append(torch.unsqueeze(output[("disp", 0)], 0))
                    if opt.log:
                        uncerts_distribution.append(torch.unsqueeze(torch.exp(output[("uncert", 0)]) ** 2, 0))

                disps_distribution = torch.cat(disps_distribution, 0)
                if opt.log:

                    # bayesian uncertainty
                    pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False) + torch.sum(
                        torch.cat(uncerts_distribution, 0), dim=0, keepdim=False)
                else:

                    # uncertainty as variance of the predictions
                    pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False)
                pred_uncert = pred_uncert.cpu()[0].numpy()
                output = torch.mean(disps_distribution, dim=0, keepdim=False)
                pred_disp, _ = disp_to_depth(output, opt.min_depth, opt.max_depth)
            elif opt.dropout:

                # infer multiple predictions from multiple networks with dropout
                disps_distribution = []

                # we infer 8 predictions as the number of bootstraps and snaphots
                for j in range(8):
                    start_time = time.time()
                    output = depth_decoder(encoder(input_color))
                    stop_time = time.time()
                    disps_distribution.append(torch.unsqueeze(output[("disp", 0)], 0))
                disps_distribution = torch.cat(disps_distribution, 0)

                # uncertainty as variance of the predictions
                pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False).cpu()[0].numpy()

                # depth as mean of the predictions
                output = torch.mean(disps_distribution, dim=0, keepdim=False)
                pred_disp, _ = disp_to_depth(output, opt.min_depth, opt.max_depth)
            elif opt.infer_dropout:
                # get prediction with normal model
                start_time = time.time()
                output = depth_decoder(encoder(input_color))
                stop_time = time.time()
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)

                # infer multiple predictions from multiple networks with dropout
                disps_distribution = []

                # we infer 8 predictions as the number of bootstraps and snaphots
                for j in range(8):
                    output = depth_decoder_drop(encoder(input_color))
                    disps_distribution.append(torch.unsqueeze(output[("disp", 0)], 0))
                disps_distribution = torch.cat(disps_distribution, 0)

                # uncertainty as variance of the predictions
                pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False).cpu()[0].numpy()
            elif opt.var_aug:
                # variance over different augmentations
                start_time = time.time()
                disps_distribution = []
                # normal depth
                output = depth_decoder(encoder(input_color))
                disp_output = output[("disp", 0)]
                pred_disp, _ = disp_to_depth(disp_output, opt.min_depth, opt.max_depth)
                disps_distribution.append(torch.unsqueeze(disp_output, 0))
                # first augmentation: flipping
                rgb_input = torch.flip(input_color, [3])
                output = depth_decoder(encoder(rgb_input))
                disp_output = output[("disp", 0)]
                disps_distribution.append(torch.unsqueeze(torch.flip(disp_output, [3]), 0))
                # second augmentation: gray-scale
                rgb_input = transforms.Grayscale(num_output_channels=3)(input_color)
                output = depth_decoder(encoder(rgb_input))
                disp_output = output[("disp", 0)]
                disps_distribution.append(torch.unsqueeze(disp_output, 0))
                # third augmentation: additive noise
                rgb_input = input_color + torch.normal(0.0, 0.01, input_color.size()).cuda()
                output = depth_decoder(encoder(rgb_input))
                disp_output = output[("disp", 0)]
                disps_distribution.append(torch.unsqueeze(disp_output, 0))
                # last augmentation: rotation
                rgb_input = transforms.functional.rotate(input_color, 10)
                output = depth_decoder(encoder(rgb_input))
                disp_output = output[("disp", 0)]
                disps_distribution.append(torch.unsqueeze(transforms.functional.rotate(disp_output, -10), 0))
                disps_distribution = torch.cat(disps_distribution, 0)
                pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False).cpu()[:, 0].numpy()
                pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
                pred_uncerts.append(pred_uncert)
                stop_time = time.time()

            else:
                start_time = time.time()
                output = depth_decoder(encoder(input_color))
                stop_time = time.time()
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                if opt.log:

                    # log-likelihood maximization
                    pred_uncert = torch.exp(output[("uncert", 0)]).cpu()[:, 0].numpy()
                elif opt.repr:

                    # learned reprojection
                    pred_uncert = (output[("uncert", 0)]).cpu()[:, 0].numpy()

            fwd_time += (stop_time - start_time)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            if opt.post_process:
                # applying Monodepthv1 post-processing to improve depth and get uncertainty
                N = pred_disp.shape[0] // 2
                pred_uncert = np.abs(pred_disp[:N] - pred_disp[N:, :, ::-1])
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                pred_uncerts.append(pred_uncert)

            pred_disps.append(pred_disp)

            # uncertainty normalization
            if opt.log or opt.repr or opt.dropout or opt.infer_dropout or nets > 1:
                pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
                pred_uncerts.append(pred_uncert)
    pred_disps = np.concatenate(pred_disps)

    fwd_time = fwd_time / len(dataset)
    print('\nAverage inference: {:.2f} ms'.format(fwd_time * 1000))

    if do_uncert and not opt.grad:
        pred_uncerts = np.concatenate(pred_uncerts)

    # saving 16 bit depth and uncertainties
    print("-> Saving 16 bit maps")
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    if not os.path.exists(os.path.join(opt.output_dir, "raw", "disp")):
        os.makedirs(os.path.join(opt.output_dir, "raw", "disp"))

    if opt.grad:
        folder_name = "uncert_" + opt.gref + "_" + opt.gloss
        if opt.w != 0.0:
            folder_name = folder_name + "_weight" + str(opt.w)
        folder_name = folder_name + "_layer_" + "_".join(str(x) for x in opt.ext_layer)
    elif opt.infer_dropout:
        folder_name = "uncert_p_" + str(opt.infer_p)
    else:
        folder_name = "uncert"
    if not os.path.exists(os.path.join(opt.output_dir, "raw", folder_name)):
        os.makedirs(os.path.join(opt.output_dir, "raw", folder_name))

    if opt.qual:
        if not os.path.exists(os.path.join(opt.output_dir, "qual", "disp")):
            os.makedirs(os.path.join(opt.output_dir, "qual", "disp"))
        if do_uncert:
            if opt.grad:
                folder_name = "uncert_" + opt.gref + "_" + opt.gloss
                if opt.w != 0.0:
                    folder_name = folder_name + "_weight" + str(opt.w)
                folder_name = folder_name + "_layer_" + "_".join(str(x) for x in opt.ext_layer)
            elif opt.infer_dropout:
                folder_name = "uncert_p_" + str(opt.infer_p)
            else:
                folder_name = "uncert"
            if not os.path.exists(os.path.join(opt.output_dir, "qual", folder_name)):
                os.makedirs(os.path.join(opt.output_dir, "qual", folder_name))

    bar = progressbar.ProgressBar(max_value=len(pred_disps))
    for i in range(len(pred_disps)):
        bar.update(i)
        if opt.eval_stereo:

            # save images scaling with KITTI baseline
            cv2.imwrite(os.path.join(opt.output_dir, "raw", "disp", '%06d_10.png' % i),
                        (pred_disps[i] * (dataset.K[0][0] * gt_depths[i].shape[1]) * 256. / 10).astype(np.uint16))

        elif opt.eval_mono:

            # save images scaling with ground truth median
            ratio = get_mono_ratio(pred_disps[i], gt_depths[i])
            cv2.imwrite(os.path.join(opt.output_dir, "raw", "disp", '%06d_10.png' % i),
                        (pred_disps[i] * (dataset.K[0][0] * gt_depths[i].shape[1]) * 256. / ratio / 10.).astype(
                            np.uint16))
        else:

            # save images scaling with custom factor
            cv2.imwrite(os.path.join(opt.output_dir, "raw", "disp", '%06d_10.png' % i),
                        (pred_disps[i] * (opt.custom_scale) * 256. / 10).astype(np.uint16))

        if do_uncert:
            # save uncertainties
            if opt.grad or opt.infer_dropout:
                cv2.imwrite(os.path.join(opt.output_dir, "raw", folder_name, '%06d_10.png' % i),
                            (pred_uncerts[i] * (256 * 256 - 1)).astype(np.uint16))
            else:
                cv2.imwrite(os.path.join(opt.output_dir, "raw", folder_name, '%06d_10.png' % i),
                            (pred_uncerts[i] * (256 * 256 - 1)).astype(np.uint16))

        if opt.qual:

            # save colored depth maps
            plt.imsave(os.path.join(opt.output_dir, "qual", "disp", '%06d_10.png' % i), pred_disps[i], cmap='magma')
            if do_uncert:
                # save colored uncertainty maps
                plt.imsave(os.path.join(opt.output_dir, "qual", folder_name, '%06d_10.png' % i), pred_uncerts[i],
                           cmap='hot')

    # see you next time! 
    print("\n-> Done!")


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    options = UncertaintyOptions()
    evaluate(options.parse())
