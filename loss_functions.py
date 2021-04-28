import torch


# Negative loglikelihood loss
def loss_loglikelihood(probas):
    return - torch.log(probas.sum(1)).mean()
    

# Absolut loss for two stratum values    
def loss_abs_gl_ml(pred_pl, gt, gl_mv_loss=False):
    if gl_mv_loss:  #if we want to get separate losses for ground level and medium level
        return ((pred_pl[:, [0, 2]] - gt[:, [0, 2]]).pow(2) + 0.0001).pow(0.5).mean(0)
    return((pred_pl[:, [0, 2]] - gt[:, [0, 2]]).pow(2) + 0.0001).pow(0.5).mean()


# Absolut loss for three stratum values
def loss_abs_gl_ml_hl(pred_pl, gt):
    gt_has_values = ~torch.isnan(gt)
    gt_has_values = gt_has_values[:, [0, 2, 3]]
    return ((pred_pl[:, [0, 2, 3]][gt_has_values] - gt[:, [0, 2, 3]][gt_has_values]).pow(2) + 0.0001).pow(0.5).mean()
