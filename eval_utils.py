"""
Source: https://github.com/mattpoggi/mono-uncertainty

-- moved eval functions to separate file
"""

import numpy as np

uncertainty_metrics = ["abs_rel", "rmse", "a1"]


def compute_eigen_errors_visu(gt, pred, valid_mask=None):
    """Computation of error metrics between predicted and ground truth depths
    """
    # for visualization
    if valid_mask:
        gt[~valid_mask] = -1.
        pred[~valid_mask] = -1.
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25)
    rmse = (gt - pred) ** 2
    abs_rel = np.abs(gt - pred) / gt

    return abs_rel, rmse, a1


def compute_eigen_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_eigen_errors_v2(gt, pred, metrics=uncertainty_metrics, mask=None, reduce_mean=False):
    """Revised compute_eigen_errors function used for uncertainty metrics, with optional reduce_mean argument and (1-a1) computation
    """
    results = []

    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]

    if "abs_rel" in metrics:
        abs_rel = (np.abs(gt - pred) / gt)
        if reduce_mean:
            abs_rel = abs_rel.mean()
        results.append(abs_rel)

    if "rmse" in metrics:
        rmse = (gt - pred) ** 2
        if reduce_mean:
            rmse = np.sqrt(rmse.mean())
        results.append(rmse)

    if "a1" in metrics:
        a1 = np.maximum((gt / pred), (pred / gt))
        if reduce_mean:

            # invert to get outliers
            a1 = (a1 >= 1.25).mean()
        results.append(a1)

    return results


def compute_aucs(gt, pred, uncert, intervals=50):
    """Computation of auc metrics
    """

    # results dictionaries
    AUSE = {"abs_rel" :0, "rmse" :0, "a1" :0}
    AURG = {"abs_rel" :0, "rmse" :0, "a1" :0}

    # revert order (high uncertainty first)
    uncert = -uncert
    true_uncert = compute_eigen_errors_v2(gt ,pred)
    true_uncert = {"abs_rel" :-true_uncert[0] ,"rmse" :-true_uncert[1] ,"a1" :-true_uncert[2]}

    # prepare subsets for sampling and for area computation
    quants = [100. / intervals * t for t in range(0, intervals)]
    plotx = [1. / intervals * t for t in range(0, intervals+1)]

    # get percentiles for sampling and corresponding subsets
    thresholds = [np.percentile(uncert, q) for q in quants]
    subs = [(uncert >= t) for t in thresholds]

    # compute sparsification curves for each metric (add 0 for final sampling)
    sparse_curve = \
        {m: [compute_eigen_errors_v2(gt, pred, metrics=[m], mask=sub, reduce_mean=True)[0] for sub in subs] + [0] for m
        in uncertainty_metrics}

    # human-readable call
    '''
    sparse_curve =  {"rmse":[compute_eigen_errors_v2(gt,pred,metrics=["rmse"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0], 
                     "a1":[compute_eigen_errors_v2(gt,pred,metrics=["a1"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0],
                     "abs_rel":[compute_eigen_errors_v2(gt,pred,metrics=["abs_rel"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0]}
    '''

    # get percentiles for optimal sampling and corresponding subsets
    opt_thresholds = {m: [np.percentile(true_uncert[m], q) for q in quants] for m in uncertainty_metrics}
    opt_subs = {m: [(true_uncert[m] >= o) for o in opt_thresholds[m]] for m in uncertainty_metrics}

    # compute sparsification curves for optimal sampling (add 0 for final sampling)
    opt_curve = {m: [compute_eigen_errors_v2(gt, pred, metrics=[m], mask=opt_sub, reduce_mean=True)[0] for opt_sub in
                     opt_subs[m]] + [0] for m in uncertainty_metrics}

    # compute metrics for random sampling (equal for each sampling)
    rnd_curve = {m: [compute_eigen_errors_v2(gt, pred, metrics=[m], mask=None, reduce_mean=True)[0] for t in
                     range(intervals + 1)] for m in uncertainty_metrics}

    # compute error and gain metrics
    for m in uncertainty_metrics:
        # error: subtract from method sparsification (first term) the oracle sparsification (second term)
        AUSE[m] = np.trapz(sparse_curve[m], x=plotx) - np.trapz(opt_curve[m], x=plotx)

        # gain: subtract from random sparsification (first term) the method sparsification (second term)
        AURG[m] = rnd_curve[m][0] - np.trapz(sparse_curve[m], x=plotx)

    # returns a dictionary with AUSE and AURG for each metric
    return {m: [AUSE[m], AURG[m]] for m in uncertainty_metrics}, \
        {m: [opt_curve[m], rnd_curve[m], sparse_curve[m]] for m in uncertainty_metrics}