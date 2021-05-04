import torch
import numpy as np
from scipy.stats import gamma



# Negative loglikelihood loss
def loss_loglikelihood(pred_pointwise, cloud, params, PCC, args):
    fit_alpha_g, fit_loc_g, fit_beta_g = params["a_g"], params["loc_g"], params["scale_g"]
    fit_alpha_v, fit_loc_v, fit_beta_v = params["a_v"], params["loc_v"], params["scale_v"]

    # We extract heights of every point
    z_all = np.empty((0))
    for current_cloud in cloud:
        z = current_cloud[2] * args.z_max  # we go back from normalized data
        z_all = np.append(z_all, z)

    z_all = np.asarray(z_all).reshape(-1)


    # we compute the z-likelihood for each point of the cloud that the point belongs to strate1 (ground) or strate2 (medium and high vegetation)
    if fit_loc_g == 0:
        pdf_ground = gamma.pdf(z_all + 1e-2, a=fit_alpha_g, loc=fit_loc_g,
                            scale=fit_beta_g)  # probability for ground points
        pdf_nonground = gamma.pdf(z_all + 1e-2, a=fit_alpha_v, loc=fit_loc_v,
                            scale=fit_beta_v)  # probability for medium and high vegetation points
    else:
        pdf_ground = gamma.pdf(z_all, a=fit_alpha_g, loc=fit_loc_g,
                            scale=fit_beta_g)  # probability for ground points
        pdf_nonground = gamma.pdf(z_all, a=fit_alpha_v, loc=fit_loc_v,
                            scale=fit_beta_v)  # probability for medium and high vegetation points

    p_all_pdf = np.concatenate((pdf_ground.reshape(-1, 1), pdf_nonground.reshape(-1, 1)), 1)
    p_all_pdf = torch.tensor(p_all_pdf)
    p_ground, p_nonground = pred_pointwise[:, :2].sum(1), pred_pointwise[:, 2:].sum(1)

    if PCC.is_cuda:
        p_all_pdf = p_all_pdf.cuda()
        p_ground = p_ground.cuda()
        p_nonground = p_nonground.cuda()

    p_ground_nonground = torch.cat((p_ground.view(-1, 1), p_nonground.view(-1, 1)), 1)
    likelihood = torch.mul(p_ground_nonground, p_all_pdf)
    return - torch.log(likelihood.sum(1)).mean(), likelihood
    
#
# # Absolut loss for two stratum values
# def loss_abs_gl_ml(pred_pl, gt, gl_mv_loss=False):
#     if gl_mv_loss:  #if we want to get separate losses for ground level and medium level
#         return ((pred_pl[:, [0, 2]] - gt[:, [0, 2]]).pow(2) + 0.0001).pow(0.5).mean(0)
#     return((pred_pl[:, [0, 2]] - gt[:, [0, 2]]).pow(2) + 0.0001).pow(0.5).mean()
#
#
# # Absolut loss for three stratum values
# def loss_abs_gl_ml_hl(pred_pl, gt):
#     gt_has_values = ~torch.isnan(gt)
#     gt_has_values = gt_has_values[:, [0, 2, 3]]
#     return ((pred_pl[:, [0, 2, 3]][gt_has_values] - gt[:, [0, 2, 3]][gt_has_values]).pow(2) + 0.0001).pow(0.5).mean()


# Admissibility loss
def loss_abs_adm(pred_adm, gt_adm):
    return ((pred_adm - gt_adm[:, -1]).pow(2) + 0.0001).pow(0.5).mean()


def loss_absolute(pred_pl, gt, args, level_loss=False):
    """
    level_loss: wheather we want to obtain losses for different vegetation levels separately
    """
    if args.nb_stratum==2:
        if level_loss:  # if we want to get separate losses for ground level and medium level
            return ((pred_pl[:, [0, 2]] - gt[:, [0, 2]]).pow(2) + 0.0001).pow(0.5).mean(0)
        return ((pred_pl[:, [0, 2]] - gt[:, [0, 2]]).pow(2) + 0.0001).pow(0.5).mean()
    if args.nb_stratum==3:
        gt_has_values = ~torch.isnan(gt)
        gt_has_values = gt_has_values[:, [0, 2, 3]]
        if level_loss:   # if we want to get separate losses for ground level and medium level
            return ((pred_pl[:, [0, 2, 3]] - gt[:, [0, 2, 3]]).pow(2) + 0.0001).pow(
                0.5).mean(0)
        return ((pred_pl[:, [0, 2, 3]][gt_has_values] - gt[:, [0, 2, 3]][gt_has_values]).pow(2) + 0.0001).pow(
            0.5).mean()