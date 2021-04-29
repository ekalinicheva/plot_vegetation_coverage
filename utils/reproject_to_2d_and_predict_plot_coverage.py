import numpy as np
import torch
from torch_scatter import scatter_max, scatter_mean


def project_to_2d(pred_pointwise, cloud, pred_pointwise_b, PCC, args, nb_stratum=2):
    """
    We do all the computation to obtain
    pred_pl - [Bx4] prediction vector for the plot
    scores -  [(BxN)x2] probas_ground_nonground that a point belongs to stratum 1 or stratum 2
    """
    index_batches = []
    index_group = []
    batches_len = []
    z_all = np.empty((0))

    # we project 3D points to 2D plane
    # We use torch scatter to process
    for b in range(len(pred_pointwise_b)):
        current_cloud = cloud[b]
        xy = current_cloud[:2]
        xy = torch.floor((xy - torch.min(xy, dim=1).values.view(2, 1).expand_as(xy)) / (
                torch.max(xy, dim=1).values - torch.min(xy, dim=1).values + 0.0001).view(2, 1).expand_as(
            xy) * args.diam_pix).int()

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

    if nb_stratum==2:
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

    if nb_stratum == 3:
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

    if args.adm:
        c_adm_pix = torch.max(pixel_max[[0,2], :], dim=0)[0]
        c_adm = scatter_mean(c_adm_pix, index_group)
    else:
        c_adm = None

    return pred_pl, c_adm

