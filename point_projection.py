import numpy as np
import torch
from torch_scatter import scatter_max, scatter_mean
from scipy.stats import gamma



def project_to_2d(pred_pointwise, cloud, pred_pointwise_b, PCC, args, params):
    """
    We do all the computation to obtain
    pred_pl - [Bx4] prediction vector for the plot
    scores -  [(BxN)x2] probas that a point belongs to stratum 1 or stratum 2
    scores_list - [BxNx2] same as scores, but separated by batch
    """



    fit_alpha_g, fit_loc_g, fit_beta_g = params["a_g"], params["loc_g"], params["scale_g"]
    fit_alpha_v, fit_loc_v, fit_beta_v = params["a_v"], params["loc_v"], params["scale_v"]


    index_batches = []
    index_group = []
    batches_len = []
    probas_list = []
    z_all = np.empty((0))

    # we project 3D points to 2D plane
    for b in range(len(pred_pointwise_b)):
        current_cloud = cloud[b]
        xy = current_cloud[:2]
        xy = torch.floor((xy - torch.min(xy, dim=1).values.view(2, 1).expand_as(xy)) / (
                torch.max(xy, dim=1).values - torch.min(xy, dim=1).values + 0.0001).view(2, 1).expand_as(
            xy) * args.diam_pix).int()
        z = current_cloud[2] * args.z_max  # we go back from normalized data
        z_all = np.append(z_all, z)
        unique, index = torch.unique(xy.T, dim=0, return_inverse=True)
        index_b = torch.full(torch.unique(index).size(), b)
        if PCC.is_cuda:
            index = index.cuda()
            index_b = index_b.cuda()
        index = index + np.asarray(batches_len).sum()
        index_batches.append(index.type(torch.LongTensor))
        index_group.append(index_b.type(torch.LongTensor))
        batches_len.append(torch.unique(index).size(0))
    index_batches = torch.cat(index_batches)
    index_group = torch.cat(index_group)
    if PCC.is_cuda:
        index_batches = index_batches.cuda()
        index_group = index_group.cuda()
    pixel_max = scatter_max(pred_pointwise.T, index_batches)[0]

    # we compute thr probabilities that a point belongs to a stratum given the observed data
    z_all = np.asarray(z_all).reshape(-1)
    if fit_loc_g == 0:
        p_g_all = gamma.pdf(z_all + 1e-2, a=fit_alpha_g, loc=fit_loc_g,
                            scale=fit_beta_g)  # probability for ground points
        p_v_all = gamma.pdf(z_all + 1e-2, a=fit_alpha_v, loc=fit_loc_v,
                            scale=fit_beta_v)  # probability for medium and high vegetation points
    else:
        p_g_all = gamma.pdf(z_all, a=fit_alpha_g, loc=fit_loc_g,
                            scale=fit_beta_g)  # probability for ground points
        p_v_all = gamma.pdf(z_all, a=fit_alpha_v, loc=fit_loc_v,
                            scale=fit_beta_v)  # probability for medium and high vegetation points

    p_all_pdf = np.concatenate((p_g_all.reshape(-1, 1), p_v_all.reshape(-1, 1)), 1)
    p_all_pdf = torch.tensor(p_all_pdf)
    p_strate1, p_strate2 = pred_pointwise[:, :2].sum(1), pred_pointwise[:, 2:].sum(1)
    if PCC.is_cuda:
        p_all_pdf = p_all_pdf.cuda()
        p_strate1 = p_strate1.cuda()
        p_strate2 = p_strate2.cuda()
    # p_strate1[p_strate1>-100000] =0.54582767
    # p_strate2[p_strate2 > -100000] = 0.45417233

    p_strates_1_2 = torch.cat((p_strate1.view(-1, 1), p_strate2.view(-1, 1)), 1)
    probas = torch.mul(p_strates_1_2, p_all_pdf)

    # probas_list.append(probas)

    # We compute prediction values per pixel
    c_low_veg_pix = pixel_max[0, :] / (pixel_max[:2, :].sum(0))
    c_bare_soil_pix = pixel_max[1, :] / (pixel_max[:2, :].sum(0))
    c_med_veg_pix = pixel_max[2, :]
    # c_other_pix = 1 - c_med_veg_pix

    # We compute prediction values per plot
    c_low_veg = scatter_mean(c_low_veg_pix, index_group)
    c_bare_soil = scatter_mean(c_bare_soil_pix, index_group)
    c_med_veg = scatter_mean(c_med_veg_pix, index_group)
    # c_other = scatter_mean(c_other_pix, index_group)
    pred_pl = torch.stack([c_low_veg, c_bare_soil, c_med_veg]).T

    return pred_pl, probas


def project_to_2d_3_stratum(pred_pointwise, cloud, pred_pointwise_b, PCC, args, params):
    fit_alpha_g, fit_loc_g, fit_beta_g = params["a_g"], params["loc_g"], params["scale_g"]
    fit_alpha_v, fit_loc_v, fit_beta_v = params["a_v"], params["loc_v"], params["scale_v"]


    index_batches = []
    index_group = []
    batches_len = []
    probas_list = []
    z_all = np.empty((0))

    # we project 3D points to 2D plane
    for b in range(len(pred_pointwise_b)):
        current_cloud = cloud[b]
        xy = current_cloud[:2]
        xy = torch.floor((xy - torch.min(xy, dim=1).values.view(2, 1).expand_as(xy)) / (
                torch.max(xy, dim=1).values - torch.min(xy, dim=1).values + 0.0001).view(2, 1).expand_as(
            xy) * args.diam_pix).int()
        z = current_cloud[2] * args.z_max  # we go back from normalized data
        z_all = np.append(z_all, z)
        unique, index = torch.unique(xy.T, dim=0, return_inverse=True)
        index_b = torch.full(torch.unique(index).size(), b)
        if PCC.is_cuda:
            index = index.cuda()
            index_b = index_b.cuda()
        index = index + np.asarray(batches_len).sum()
        index_batches.append(index.type(torch.LongTensor))
        index_group.append(index_b.type(torch.LongTensor))
        batches_len.append(torch.unique(index).size(0))
    index_batches = torch.cat(index_batches)
    index_group = torch.cat(index_group)
    if PCC.is_cuda:
        index_batches = index_batches.cuda()
        index_group = index_group.cuda()
    pixel_max = scatter_max(pred_pointwise.T, index_batches)[0]

    # we compute the z-probas for each point of the cloud that the point belongs to strate1 (ground) or strate2 (medium and high vegetation)
    z_all = np.asarray(z_all).reshape(-1)
    if fit_loc_g == 0:
        p_g_all = gamma.pdf(z_all + 1e-2, a=fit_alpha_g, loc=fit_loc_g,
                            scale=fit_beta_g)  # probability for ground points
        p_v_all = gamma.pdf(z_all + 1e-2, a=fit_alpha_v, loc=fit_loc_v,
                            scale=fit_beta_v)  # probability for medium and high vegetation points
    else:
        p_g_all = gamma.pdf(z_all, a=fit_alpha_g, loc=fit_loc_g,
                            scale=fit_beta_g)  # probability for ground points
        p_v_all = gamma.pdf(z_all, a=fit_alpha_v, loc=fit_loc_v,
                            scale=fit_beta_v)  # probability for medium and high vegetation points

    p_all_pdf = np.concatenate((p_g_all.reshape(-1, 1), p_v_all.reshape(-1, 1)), 1)
    p_all_pdf = torch.tensor(p_all_pdf)
    p_strate1, p_strate2 = pred_pointwise[:, :2].sum(1), pred_pointwise[:, 2:].sum(1)
    if PCC.is_cuda:
        p_all_pdf = p_all_pdf.cuda()
        p_strate1 = p_strate1.cuda()
        p_strate2 = p_strate2.cuda()

    p_strates_1_2 = torch.cat((p_strate1.view(-1, 1), p_strate2.view(-1, 1)), 1)
    probas = torch.mul(p_strates_1_2, p_all_pdf)


    # We compute prediction values par pixel
    c_low_veg_pix = pixel_max[0, :]
    c_bare_soil_pix = 1 - c_low_veg_pix
    # c_bare_soil_pix = pixel_max[1, :]
    c_med_veg_pix = pixel_max[2, :]
    c_high_veg_pix = pixel_max[3, :]

    # We compute prediction values by each zone
    c_low_veg = scatter_mean(c_low_veg_pix, index_group)
    c_bare_soil = scatter_mean(c_bare_soil_pix, index_group)
    c_med_veg = scatter_mean(c_med_veg_pix, index_group)
    c_high_veg = scatter_mean(c_high_veg_pix, index_group)
    pred_pl = torch.stack([c_low_veg, c_bare_soil, c_med_veg, c_high_veg]).T


    return pred_pl, probas