# Monodepth2 to train in a supervised manner on NYU Depth V2
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
"""
Source: modified from https://github.com/nianticlabs/monodepth2
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from monodepth2.layers import compute_depth_errors

from monodepth2.trainer import Trainer as base_trainer
import monodepth2.networks as legacy
import networks as networks
from datasets.nyu_dataset import NYUDataset


def uncertainty_loss(pred, gt, uct):
    abs_diff = torch.abs(pred - gt)
    loss = (abs_diff / torch.exp(uct)) + uct
    loss = torch.mean(loss, dim=[1, 2, 3])
    loss = torch.mean(loss)
    return loss


class Trainer(base_trainer):
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)

        self.models["encoder"] = legacy.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthUncertaintyDecoder_Supervised(self.models["encoder"].num_ch_enc,
                                                                           self.opt.scales, dropout=self.opt.dropout,
                                                                           uncert=self.opt.uncert)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        train_dataset = NYUDataset(self.opt.data_path + '/train/', split='train', height=self.opt.height,
                                   width=self.opt.width)
        num_train_samples = len(train_dataset)
        val_dataset = NYUDataset(self.opt.data_path + '/train/', split='holdout', height=self.opt.height,
                                 width=self.opt.width)

        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def train(self):
        """Run the training pipeline in a supervised manner
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        if self.opt.uncert:
            criterion = uncertainty_loss
        else:
            criterion = nn.L1Loss()
        for self.epoch in range(self.opt.num_epochs):
            self.model_lr_scheduler.step()

            print("Training")
            self.set_train()

            for batch_idx, inputs in enumerate(self.train_loader):
                rgb_img = inputs[("color", 0, 0)].cuda()
                gt_depth = inputs["depth_gt"].cuda()
                before_op_time = time.time()

                # Otherwise, we only feed the image with frame_id 0 through the depth encoder
                features = self.models["encoder"](rgb_img)
                outputs = self.models["depth"](features)

                total_loss = 0
                losses = {}

                for scale in self.opt.scales:

                    depth = outputs[("depth", scale)]
                    depth = F.interpolate(depth, [self.opt.height, self.opt.width], mode="bilinear",
                                          align_corners=False)
                    if self.opt.uncert:
                        uncert = outputs[("uncert", scale)]
                        uncert = F.interpolate(uncert, [self.opt.height, self.opt.width], mode="bilinear",
                                               align_corners=False)
                        loss = criterion(depth, gt_depth, uncert)
                    else:
                        loss = criterion(depth, gt_depth)

                    total_loss += loss
                    losses["loss/{}".format(scale)] = loss

                total_loss /= self.num_scales
                losses["loss"] = total_loss

                self.model_optimizer.zero_grad()
                total_loss.backward()
                self.model_optimizer.step()

                duration = time.time() - before_op_time

                # log less frequently after the first 2000 steps to save time & disk space
                early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
                late_phase = self.step % 2000 == 0

                if early_phase or late_phase:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                    if "depth_gt" in inputs:
                        self.compute_depth_losses(inputs, outputs, losses)

                    self.log("train", inputs, outputs, losses)
                    self.val()

                self.step += 1
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def val(self):
        """Validate the model on a single minibatch
        """
        if self.opt.uncert:
            criterion = uncertainty_loss
        else:
            criterion = nn.L1Loss()

        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
                rgb_img = inputs[("color", 0, 0)].to(self.device)
                gt_depth = inputs["depth_gt"].to(self.device)

                # Otherwise, we only feed the image with frame_id 0 through the depth encoder
                features = self.models["encoder"](rgb_img)
                outputs = self.models["depth"](features)

                total_loss = 0
                losses = {}

                for scale in self.opt.scales:
                    depth = outputs[("depth", scale)]
                    depth = F.interpolate(depth, [self.opt.height, self.opt.width], mode="bilinear",
                                          align_corners=False)
                    if self.opt.uncert:
                        uncert = outputs[("uncert", scale)]
                        uncert = F.interpolate(uncert, [self.opt.height, self.opt.width], mode="bilinear",
                                               align_corners=False)
                        loss = criterion(depth, gt_depth, uncert)
                    else:
                        loss = criterion(depth, gt_depth)

                    total_loss += loss
                    losses["loss/{}".format(scale)] = loss

                total_loss /= self.num_scales
                losses["loss"] = total_loss

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("val", inputs, outputs, losses)
                del inputs, outputs, losses
        self.set_train()

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0)].detach()

        depth_gt = inputs["depth_gt"].to(self.device)
        mask = depth_gt > 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=10)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            writer.add_image("color/{}".format(j), inputs[("color", 0, 0)][j].data, self.step)
            for s in self.opt.scales:
                writer.add_image("depth_{}/{}".format(s, j), outputs[("depth", s)][j].data, self.step)
