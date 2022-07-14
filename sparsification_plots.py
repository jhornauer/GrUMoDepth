import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=str, help="[kitti-m], [kitti-s], [nyu]")
    parser.add_argument("--metric", type=str, default="rmse", help="[rmse], [abs_rel], [a1]")
    args = parser.parse_args()
    intervals = 50
    plotx = [1./intervals*t for t in range(0, intervals+1)]

    plt.rcParams.update({"font.size": 12})

    if args.setup == "kitti-m":
        root_dir = 'experiments/M/'
        models = ['post_model/Post', 'post_model/Infer-Drop', 'post_model/Grad', 'log_model/Log',
                  'log_model/Infer-Drop', 'log_model/Grad', 'self_model/Self', 'self_model/Infer-Drop',
                  'self_model/Grad', 'drop_model/Drop', 'boot_model/Boot']
        legend = ['post-Base', 'post-In-Drop', 'post-Grad', 'log-Base', 'log-In-Drop', 'log-Grad', 'self-Base',
                  'self-In-Drop', 'self-Grad', 'Drop', 'Boot']
        colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:green',
                  'tab:green', 'tab:green', 'tab:red', 'tab:purple']
        marksers = ['--', ':', '-', '--', ':', '-', '--', ':', '-', '--', '--']
    elif args.setup == "kitti-s":
        root_dir = 'experiments/S/'
        models = ['post_model/Post', 'post_model/Infer-Drop', 'post_model/Grad', 'log_model/Log',
                  'log_model/Infer-Drop', 'log_model/Grad', 'self_model/Self', 'self_model/Infer-Drop',
                  'self_model/Grad', 'drop_model/Drop', 'boot_model/Boot']
        legend = ['post-Base', 'post-In-Drop', 'post-Grad', 'log-Base', 'log-In-Drop', 'log-Grad', 'self-Base',
                  'self-In-Drop', 'self-Grad', 'Drop', 'Boot']
        colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:green',
                  'tab:green', 'tab:green', 'tab:red', 'tab:purple']
        marksers = ['--', ':', '-', '--', ':', '-', '--', ':', '-', '--', '--']
    elif args.setup == 'nyu':
        root_dir = 'experiments/NYU'
        models = ['post_model/Post', 'post_model/Infer-Drop', 'post_model/Grad', 'log_model/Log',
                  'log_model/Infer-Drop', 'log_model/Grad', 'drop_model/Drop']
        legend = ['post-Base', 'post-In-Drop', 'post-Grad', 'log-Base', 'log-In-Drop', 'log-Grad', 'Drop']
        colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:red']
        marksers = ['--', ':', '-', '--', ':', '-', '--']
    else:
        raise NotImplementedError

    error_curves = []
    for i, m in enumerate(models):
        curves = pickle.load(open(os.path.join(root_dir, m, 'spars_plots.pkl'), 'rb'))
        rmse_curves = np.array(curves[args.metric])
        mean_rmse_curves = rmse_curves.mean(0)
        opt = mean_rmse_curves[0]
        spc = mean_rmse_curves[2]
        plt.plot(plotx[:-1], (spc[:-1] - opt[:-1]), color=colors[i], linestyle=marksers[i])

    plt.grid()
    plt.legend(legend)
    plt.xlabel("Fraction of Removed Pixels")
    plt.ylabel("Sparsification Error")
    plt.show()

