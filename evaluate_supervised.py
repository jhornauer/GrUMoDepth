from __future__ import absolute_import, division, print_function
import warnings
import pickle
from torch.utils.data import DataLoader

from extended_options import *
import datasets as datasets
import monodepth2.networks as legacy
import networks as networks
import progressbar
import matplotlib.pyplot as plt

from gradients import *
from torchvision import transforms
from eval_utils import compute_eigen_errors, compute_eigen_errors_visu, compute_aucs

import sys

uncertainty_metrics = ["abs_rel", "rmse", "a1"]


splits_dir = os.path.join(os.path.dirname(__file__), "monodepth2/splits")


def batch_post_process_depth(l_depth, r_depth):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_depth.shape
    m_depth = 0.5 * (l_depth + r_depth)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_depth + l_mask * r_depth + (1.0 - l_mask - r_mask) * m_depth


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = opt.max_depth
    opt.batch_size = 1

    print("-> Beginning inference...")

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    # prepare just a single path
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path)
    height = encoder_dict['height']
    width = encoder_dict['width']

    dataset = datasets.NYUDataset(opt.data_path + '/val/', split='val', height=height, width=width)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True,
                            drop_last=False)

    # load a single encoder and decoder
    encoder = legacy.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthUncertaintyDecoder_Supervised(encoder.num_ch_enc, scales=opt.scales,
                                                                dropout=opt.dropout, uncert=opt.uncert)

    if opt.infer_dropout:
        depth_decoder_drop = networks.DepthUncertaintyDecoder_Supervised(encoder.num_ch_enc, scales=opt.scales,
                                                                         dropout=opt.dropout, uncert=opt.uncert,
                                                                         infer_dropout=opt.infer_dropout,
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
    pred_depths = []
    pred_uncerts = []

    if opt.grad:
        ext_layer = ['decoder.0.conv', 'decoder.1.conv', 'decoder.2.conv', 'decoder.3.conv', 'decoder.4.conv',
                     'decoder.5.conv', 'decoder.6.conv', 'decoder.7.conv', 'decoder.8.conv', 'decoder.9.conv',
                     'decoder.10.conv']
        layer_list = [ext_layer[layer_idx] for layer_idx in opt.ext_layer]
        gradient_extractor = Gradient_Analysis(depth_decoder, layer_list, encoder_dict['height'],
                                               encoder_dict['width'], opt.gred)

        print("-> Extract gradients from model for uncertainty estimation")

        bwd_time = 0
        n_samples = 0

        # check loss function type
        if opt.gloss not in ["sq", "none", "var"]:
            raise NotImplementedError

        for i, inputs in enumerate(dataloader):
            rgb_img = inputs[("color", 0, 0)].cuda()
            gt_depth = inputs["depth_gt"]

            if opt.gref == "flip":
                ref_img = torch.flip(rgb_img, [3])
            elif opt.gref == "gray":
                ref_img = transforms.Grayscale(num_output_channels=3)(rgb_img)
            elif opt.gref == "noise":
                ref_img = rgb_img + torch.normal(0.0, 0.01, rgb_img.size()).cuda()
            elif opt.gref == "rot":
                ref_img = transforms.functional.rotate(rgb_img, angle=opt.angle)
            elif opt.gref == "var":
                ref_imgs = [torch.flip(rgb_img, [3]), transforms.Grayscale(num_output_channels=3)(rgb_img),
                            rgb_img + torch.normal(0.0, 0.01, rgb_img.size()).cuda(),
                            transforms.functional.rotate(rgb_img, 10)]
            elif opt.gref in ["none", "gt"]:
                pass
            else:
                raise NotImplementedError

            if opt.gref in ["flip", "gray", "noise", "rot"]:
                with torch.no_grad():
                    output = depth_decoder(encoder(ref_img))
                    ref_depth = output[("depth", 0)]
                    if opt.uncert:
                        ref_uncert = output[("uncert", 0)]
                if opt.gref == "flip":
                    ref_depth = torch.from_numpy(ref_depth.cpu().numpy()[:, :, :, ::-1].copy()).cuda()
                    if opt.uncert:
                        ref_uncert = torch.from_numpy(ref_uncert.cpu().numpy()[:, :, :, ::-1].copy()).cuda()
                elif opt.gref == "rot":
                    ref_depth = transforms.functional.rotate(ref_depth, -opt.angle)
                    if opt.uncert:
                        ref_uncert = transforms.functional.rotate(ref_uncert, -opt.angle)
            elif opt.gref == "var":
                ref_depths = []
                with torch.no_grad():
                    for i, input in enumerate(ref_imgs):
                        output = depth_decoder(encoder(input))
                        if i == 0:
                            ref_depths.append(torch.flip(output[("depth", 0)], [3]))
                        elif i == 3:
                            ref_depths.append(transforms.functional.rotate(output[("depth", 0)], -10))
                        else:
                            ref_depths.append(output[("depth", 0)])
            elif opt.gref == "gt":
                ref_depth = gt_depth.cuda()
            elif opt.gref == "none":
                if opt.gloss != "none":
                    print("Gradient reference required for loss calculation.")
                    raise NotImplementedError
            else:
                raise NotImplementedError

            output = gradient_extractor(encoder(rgb_img))
            pred_depth = output[("depth", 0)]

            n_samples += rgb_img.shape[0]

            loss = 0
            if opt.gloss == "var":
                loss = torch.var(torch.cat([pred_depth, ref_depths[0], ref_depths[1], ref_depths[2], ref_depths[3]], 0),
                                 dim=0)
                loss = torch.mean(loss)
            else:
                if opt.gloss == "sq":
                    depth_diff = squared_difference(pred_depth, ref_depth)
                    loss += torch.mean(depth_diff)
                if opt.uncert and opt.w != 0.0:
                    pred_uncert = output[("uncert", 0)]
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
    errors = []
    errors_abs_rel = []
    errors_rmse = []

    # dictionary with accumulators for each metric
    aucs = {"abs_rel": [], "rmse": [], "a1": []}
    curves = {"abs_rel": [], "rmse": [], "a1":[]}

    with torch.no_grad():
        bar = progressbar.ProgressBar(max_value=len(dataloader))
        for i, inputs in enumerate(dataloader):
            rgb_img = inputs[("color", 0, 0)]
            gt_depth = inputs["depth_gt"]
            gt_depth = gt_depth[:, 0].cpu().numpy()
            rgb_img = rgb_img.cuda()
            # updating progress bar
            bar.update(i)
            if opt.post_process:
                # post-processed results require each image to have two forward passes
                rgb_img = torch.cat((rgb_img, torch.flip(rgb_img, [3])), 0)

            if opt.dropout:
                # infer multiple predictions from multiple networks with dropout
                depth_distribution = []
                # we infer 8 predictions as the number of bootstraps and snaphots
                for j in range(8):
                    start_time = time.time()
                    output = depth_decoder(encoder(rgb_img))
                    stop_time = time.time()
                    depth_distribution.append(torch.unsqueeze(output[("depth", 0)], 0))
                depth_distribution = torch.cat(depth_distribution, 0)

                # uncertainty as variance of the predictions
                pred_uncert = torch.var(depth_distribution, dim=0, keepdim=False).cpu()[:, 0].numpy()
                pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
                pred_uncerts.append(pred_uncert)

                # depth as mean of the predictions
                pred_depth = torch.mean(depth_distribution, dim=0, keepdim=False).cpu()[:, 0].numpy()
            elif opt.infer_dropout:
                start_time = time.time()
                output = depth_decoder(encoder(rgb_img))
                stop_time = time.time()
                pred_depth = output[("depth", 0)][:, 0].cpu().numpy()

                # infer multiple predictions from multiple networks with dropout
                depth_distribution = []
                # we infer 8 predictions as the number of bootstraps and snaphots
                for j in range(8):
                    start_time = time.time()
                    output = depth_decoder_drop(encoder(rgb_img))
                    stop_time = time.time()
                    depth_distribution.append(torch.unsqueeze(output[("depth", 0)], 0))
                depth_distribution = torch.cat(depth_distribution, 0)

                # uncertainty as variance of the predictions
                pred_uncert = torch.var(depth_distribution, dim=0, keepdim=False).cpu()[:, 0].numpy()
                pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
                pred_uncerts.append(pred_uncert)

            elif opt.var_aug:
                start_time = time.time()
                depth_distribution = []
                # normal depth
                output = depth_decoder(encoder(rgb_img))
                depth_output = output[("depth", 0)]
                pred_depth = depth_output[:, 0].cpu().numpy()
                depth_distribution.append(torch.unsqueeze(depth_output, 0))
                # first augmentation: flipping
                rgb_input = torch.flip(rgb_img, [3])
                output = depth_decoder(encoder(rgb_input))
                depth_output = output[("depth", 0)]
                depth_distribution.append(torch.unsqueeze(torch.flip(depth_output, [3]), 0))
                # second augmentation: gray-scale
                rgb_input = transforms.Grayscale(num_output_channels=3)(rgb_img)
                output = depth_decoder(encoder(rgb_input))
                depth_output = output[("depth", 0)]
                depth_distribution.append(torch.unsqueeze(depth_output, 0))
                # third augmentation: additive noise
                rgb_input = rgb_img + torch.normal(0.0, 0.01, rgb_img.size()).cuda()
                output = depth_decoder(encoder(rgb_input))
                depth_output = output[("depth", 0)]
                depth_distribution.append(torch.unsqueeze(depth_output, 0))
                # last augmentation: rotation
                rgb_input = transforms.functional.rotate(rgb_img, 10)
                output = depth_decoder(encoder(rgb_input))
                depth_output = output[("depth", 0)]
                depth_distribution.append(torch.unsqueeze(transforms.functional.rotate(depth_output, -10), 0))
                depth_distribution = torch.cat(depth_distribution, 0)
                pred_uncert = torch.var(depth_distribution, dim=0, keepdim=False).cpu()[:, 0].numpy()
                pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
                pred_uncerts.append(pred_uncert)
                stop_time = time.time()
            else:
                start_time = time.time()
                output = depth_decoder(encoder(rgb_img))
                stop_time = time.time()

                pred_depth = output[("depth", 0)][:, 0]
                pred_depth = pred_depth.cpu().numpy()

            fwd_time += (stop_time - start_time)

            if opt.post_process:
                # applying Monodepthv1 post-processing to improve depth and get uncertainty
                N = pred_depth.shape[0] // 2
                pred_uncert = np.abs(pred_depth[:N] - pred_depth[N:, :, ::-1])
                pred_depth = batch_post_process_depth(pred_depth[:N], pred_depth[N:, :, ::-1])
                pred_uncerts.append(pred_uncert)

            # only needed is maps are saved
            pred_depths.append(pred_depth)

            if opt.log:
                pred_uncert = torch.exp(output[("uncert", 0)])[:,0].cpu().numpy()
                pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
                pred_uncerts.append(pred_uncert)

            if opt.grad:
                pred_uncert = pred_uncerts[i].reshape(1, pred_uncerts.shape[1], pred_uncerts.shape[2])

            # traditional eigen crop
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            # get error maps
            tmp_abs_rel, tmp_rmse, tmp_a1 = compute_eigen_errors_visu(gt_depth, pred_depth)
            errors_abs_rel.append(tmp_abs_rel)
            errors_rmse.append(tmp_rmse)

            # apply masks
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            if opt.eval_uncert:
                pred_uncert = pred_uncert[mask]

            # apply depth cap
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

    fwd_time = fwd_time / len(dataset)
    print('\nAverage inference: {:.2f} ms'.format(fwd_time * 1000))

    if type(pred_uncerts) == list:
        pred_uncerts = np.concatenate(pred_uncerts)
    pred_depths = np.concatenate(pred_depths)

    if opt.save_error_map:
        errors_abs_rel = np.concatenate(errors_abs_rel)
        errors_rmse = np.concatenate(errors_rmse)

    # compute mean depth metrics and print
    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    # pred_depths = np.concatenate(pred_depths)

    if opt.eval_uncert:
        # compute mean uncertainty metrics and print
        for m in uncertainty_metrics:
            aucs[m] = np.array(aucs[m]).mean(0)
            print("\n  " + ("{:>8} | " * 6).format("abs_rel", "", "rmse", "", "a1", ""))
        print("  " + ("{:>8} | " * 6).format("AUSE", "AURG", "AUSE", "AURG", "AUSE", "AURG"))
        print(
            ("&{:8.3f}  " * 6).format(*aucs["abs_rel"].tolist() + aucs["rmse"].tolist() + aucs["a1"].tolist()) + "\\\\")

        # save sparsification plots
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    pickle.dump(curves, open(os.path.join(opt.output_dir, "spars_plots.pkl"), "wb"))

    if opt.save_depth_map:
        # check if output directory exists
        if not os.path.exists(opt.output_dir):
            os.mkdir(opt.output_dir)
        # only save qualitative results
        if not os.path.exists(os.path.join(opt.output_dir, "depth")):
            os.makedirs(os.path.join(opt.output_dir, "depth"))

        print("--> Saving qualitative depth maps")
        bar = progressbar.ProgressBar(max_value=len(pred_depths))
        for i in range(len(pred_depths)):
            bar.update(i)
            # save colored depth maps
            plt.imsave(os.path.join(opt.output_dir, "depth", '%06d_10.png' % i), pred_depths[i],
                       cmap='magma_r')

    if opt.save_error_map:
        if not os.path.exists(opt.output_dir):
            os.mkdir(opt.output_dir)

        if not os.path.exists(os.path.join(opt.output_dir, "abs_rel")):
            os.makedirs(os.path.join(opt.output_dir, "abs_rel"))
        if not os.path.exists(os.path.join(opt.output_dir, "rmse")):
            os.makedirs(os.path.join(opt.output_dir, "rmse"))
        if not os.path.exists(os.path.join(opt.output_dir, "a1")):
            os.makedirs(os.path.join(opt.output_dir, "a1"))

        print("--> Saving qualitative error maps: abs rel")
        bar = progressbar.ProgressBar(max_value=len(errors_abs_rel))
        for i in range(len(errors_abs_rel)):
            bar.update(i)
            # save colored depth maps
            plt.imsave(os.path.join(opt.output_dir, "abs_rel", '%06d_10.png' % i), errors_abs_rel[i], cmap='hot')

        print("--> Saving qualitative error maps: rmse")
        bar = progressbar.ProgressBar(max_value=len(errors_rmse))
        for i in range(len(errors_rmse)):
            bar.update(i)
            # save colored depth maps
            plt.imsave(os.path.join(opt.output_dir, "rmse", '%06d_10.png' % i), errors_rmse[i], cmap='hot')

    if opt.save_uncert_map:
        # check if output directory exists
        if not os.path.exists(opt.output_dir):
            os.mkdir(opt.output_dir)
        if opt.grad:
            folder_name = "uncert_" + opt.gref + "_" + opt.gloss + "_" + opt.gred
            if opt.w != 0.0:
                folder_name = folder_name + "_weight" + str(opt.w)
            folder_name = folder_name + "_layer_" + "_".join(str(x) for x in opt.ext_layer)
        elif opt.infer_dropout:
            folder_name = "uncert_p_" + str(opt.infer_p)
        else:
            folder_name = "uncert"
        if not os.path.exists(os.path.join(opt.output_dir, folder_name)):
            os.makedirs(os.path.join(opt.output_dir, folder_name))

        print("--> Saving qualitative uncertainty maps")
        bar = progressbar.ProgressBar(max_value=len(pred_uncerts))
        for i in range(len(pred_uncerts)):
            bar.update(i)
            # save colored uncertainty maps
            plt.imsave(os.path.join(opt.output_dir, folder_name, '%06d_10.png' % i), pred_uncerts[i], cmap='hot')

    # see you next time!
    print("\n-> Done!")


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    options = UncertaintyOptions()
    evaluate(options.parse())
