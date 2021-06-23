import matplotlib
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.gridspec as gridspec

import numpy as np
import torch
from osgeo import gdal, osr
from utils.useful_functions import print_stats
import torch.nn as nn

plt.rcParams["font.size"] = 25


def visualize_article(
    image_soil, image_med_veg, image_high_veg, cloud, pl_id, stats_path, args, txt=None
):

    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3)

    # Original data
    ax1 = fig.add_subplot(gs[:, 0:2], projection="3d")

    # Fake color to see vegetation more clearly
    nir_r_g_indexes = [6, 3, 4]
    c = cloud[nir_r_g_indexes].numpy().transpose()

    # # NDVI calculation
    # r_infra = cloud[[3, 6]].numpy().transpose()
    # r = r_infra[:, 0]
    # infra = r_infra[:, 1]
    # ndvi = (infra - r) / (infra + r)
    # top = cm.get_cmap("Blues_r", 128)
    # bottom = cm.get_cmap("Greens", 128)
    # cmap = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    # cmap = colors.ListedColormap(cmap, name="GreensBlues")

    ax1.scatter(
        cloud[0],
        cloud[1],
        cloud[2] * args.z_max,
        c=c,
        vmin=0,
        vmax=1,
        s=10,
        alpha=1,
    )
    ax1.auto_scale_xyz
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_title(pl_id)
    for line in ax1.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax1.yaxis.get_ticklines():
        line.set_visible(False)

    # LV stratum raster
    ax2 = fig.add_subplot(gs[0, 2])
    color_grad = [(0.8, 0.4, 0.1), (0, 1, 0)]  # first color is white, last is green
    cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
    ax2.imshow(image_soil, cmap=cmap, vmin=0, vmax=1)
    ax2.set_title("Ground level")
    ax2.tick_params(
        axis="both",  # changes apply to both axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])

    # MV stratum raster
    ax3 = fig.add_subplot(gs[1, 2])
    color_grad = [(1, 1, 1), (0, 1, 0)]  # first color is white, last is green
    cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
    ax3.imshow(image_med_veg, cmap=cmap, vmin=0, vmax=1)
    ax3.set_title("Medium level")
    ax3.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])

    # Plot high vegetation stratum
    ax4 = fig.add_subplot(gs[2, 2])
    color_grad = [(1, 1, 1), (0, 1, 0)]  # first color is white, last is red
    cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
    ax4.imshow(image_high_veg, cmap=cmap, vmin=0, vmax=1)
    ax4.set_title("High level")
    ax4.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])

    if txt is not None:
        fig.text(0.5, 0.05, txt, ha="center")
    plt.savefig(
        stats_path + pl_id + "_article.svg", format="svg", bbox_inches="tight", dpi=300
    )
    plt.clf()
    plt.close("all")


def visualize(
    image_low_veg,
    image_med_veg,
    cloud,
    prediction,
    pl_id,
    stats_path,
    args,
    text_pred_vs_gt=None,
    scores=None,
    image_high_veg=None,
    predictions=["Vb", "soil", "Vm", "Vh", "Adm"],
    gt=["Vb", "soil", "Vm", "Vh", "Adm"],
):

    if image_low_veg.ndim == 3:
        image_low_veg = image_low_veg[:, :, 0]
        image_med_veg = image_med_veg[:, :, 0]

    # We set figure size depending on the number of subplots
    if scores is None and image_high_veg is None:
        row, col = 2, 2
        fig = plt.figure(figsize=(20, 15))
    else:
        row, col = 3, 2
        fig = plt.figure(figsize=(20, 25))

    # Original point data
    ax1 = fig.add_subplot(row, col, 1, projection="3d")

    # Fake color to see vegetation more clearly
    nir_r_g_indexes = [6, 3, 4]
    c = cloud[nir_r_g_indexes].numpy().transpose()

    # NDVI calculation
    # r_infra = cloud[[3, 6]].numpy().transpose()
    # r = r_infra[:, 0]
    # infra = r_infra[:, 1]
    # ndvi = (infra - r) / (infra + r)
    # top = cm.get_cmap("Blues_r", 128)
    # bottom = cm.get_cmap("Greens", 128)
    # cmap = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    # cmap = colors.ListedColormap(cmap, name="GreensBlues")
    ax1.scatter(
        cloud[0],
        cloud[1],
        cloud[2] * args.z_max,
        c=c,
        vmin=0,
        vmax=1,
        s=10,
        alpha=1,
    )
    ax1.auto_scale_xyz
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_title(f"{pl_id}")
    # sm = ScalarMappable(cmap=cmap)  # bad norm 0-1 right now
    # sm.set_array([])
    # plt.colorbar(sm, ax=ax1)

    # LV stratum raster
    ax2 = fig.add_subplot(row, col, 2)
    color_grad = [
        (0.8, 0.4, 0.1),
        (0.91, 0.91, 0.91),
        (0, 1, 0),
    ]  # first color is brown, second is grey, last is green
    cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
    ax2.imshow(image_low_veg, cmap=cmap, vmin=0, vmax=1)
    ax2.set_title(f"Low veg. = {predictions[0]:.0%} (gt={gt[0]:.0%})")
    ax2.tick_params(
        axis="both",  # changes apply to both axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    PCM = ax2.get_children()[9]
    plt.colorbar(PCM, ax=ax2)

    # Pointwise prediction
    ax3 = fig.add_subplot(row, col, 3, projection="3d")
    ax3.auto_scale_xyz
    colors_pred = prediction.cpu().detach().numpy().transpose()
    color_matrix = [[0, 1, 0], [0.8, 0.4, 0.1], [0, 0, 1], [1, 0, 0]]
    colors_pred = np.matmul(colors_pred, color_matrix)
    ax3.scatter(
        cloud[0],
        cloud[1],
        cloud[2] * args.z_max,
        c=colors_pred,
        s=10,
        vmin=0,
        vmax=1,
        alpha=1,
    )
    ax3.set_title("Pointwise prediction")
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])

    # MV stratum raster
    ax4 = fig.add_subplot(row, col, 4)
    color_grad = [(1, 1, 1), (0, 0, 1)]  # first color is white, last is blue
    cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
    ax4.imshow(image_med_veg, cmap=cmap, vmin=0, vmax=1)
    ax4.set_title(f"Medium veg. = {predictions[2]:.0%} (gt={gt[2]:.0%})")
    ax4.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    PCM = ax4.get_children()[9]
    plt.colorbar(PCM, ax=ax4)

    # Plot stratum scores
    if scores is not None:
        ax5 = fig.add_subplot(row, col, 5, projection="3d")
        ax5.auto_scale_xyz
        sc_sum = scores.sum(1)
        scores[:, 0] = scores[:, 0] / sc_sum
        scores[:, 1] = scores[:, 1] / sc_sum
        scores = scores / (scores.max())
        colors_pred = (
            scores.transpose(1, 0)[[0, 0, 1], :].cpu().detach().numpy().transpose()
        )
        ax5.scatter(
            cloud[0],
            cloud[1],
            cloud[2] * args.z_max,
            c=colors_pred,
            s=10,
            vmin=0,
            vmax=1,
        )
        ax5.set_title("Ground vs. non-ground")
        ax5.set_yticklabels([])
        ax5.set_xticklabels([])

    # Plot high vegetation stratum
    if image_high_veg is not None:
        ax6 = fig.add_subplot(row, col, 6)
        color_grad = [(1, 1, 1), (1, 0, 0)]  # first color is white, last is red
        cmap = colors.LinearSegmentedColormap.from_list("Custom", color_grad, N=100)
        ax6.imshow(image_high_veg, cmap=cmap, vmin=0, vmax=1)
        ax6.set_title(f"High veg. = {predictions[3]:.0%} (gt={gt[3]:.0%})")
        ax6.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
        )  # labels along the bottom edge are off
        ax6.set_yticklabels([])
        ax6.set_xticklabels([])
        PCM = ax6.get_children()[9]
        plt.colorbar(PCM, ax=ax6)

    if text_pred_vs_gt is not None:
        fig.text(0.5, 0.05, text_pred_vs_gt, ha="center")
    plt.savefig(stats_path + pl_id + ".png", format="png", bbox_inches="tight", dpi=300)
    plt.clf()
    plt.close("all")


def create_final_images(
    pred_pl,
    gt,
    pred_pointwise_b,
    cloud,
    likelihood,
    plot_name,
    xy_centers_dict,
    stats_path,
    stats_file,
    args,
    create_and_save_raster_as_TIFF_file=True,
    adm=None,
):
    """
    We do final data reprojection to the 2D space (2 stratum - ground vegetation level and medium level, optionally high level)
    by associating the points to the pixels.
    Then we create the images with those stratum
    """
    for b in range(len(pred_pointwise_b)):
        # we get prediction stats string
        pred_pointwise = pred_pointwise_b[b]
        current_cloud = cloud[b]  # (9, N) tensor
        plot_center = xy_centers_dict[plot_name]  # tuple (x,y)

        # we do raster reprojection, but we do not use torch scatter as we have to associate each value to a pixel
        image_low_veg, image_med_veg, image_high_veg = infer_and_project_on_rasters(
            current_cloud, args, pred_pointwise
        )
        # We normalize back x,y values to create a tiff file with 2 rasters
        if create_and_save_raster_as_TIFF_file:
            xy = (
                current_cloud[:2, :].detach().cpu().numpy()
            )  # (2, N) tensor -> (2, N) nparray
            img_to_write, geo = rescale_xy_and_get_geotransformation_(
                xy,
                plot_center,
                args,
                image_low_veg,
                image_med_veg,
                image_high_veg,
            )
            create_tiff(
                nb_channels=args.nb_stratum,
                new_tiff_name=stats_path + plot_name + ".tif",
                width=args.diam_pix,
                height=args.diam_pix,
                datatype=gdal.GDT_Float32,
                data_array=img_to_write,
                geotransformation=geo,
            )

        if args.adm:
            preds_nparray = np.round(
                np.asarray(pred_pl[b].cpu().detach().numpy().reshape(-1)), 2
            )
            adm_ = adm[b].cpu().detach().numpy().round(2)
            gt_nparray = gt.cpu().numpy()[0]
            text_pred_vs_gt = (
                f"Coverage: Pred {preds_nparray[:4]} GT   {gt_nparray[:-1]}\n "
                + f"Admissibility: Pred {adm_:.2f}  GT  {gt_nparray[-1]:.2f}"
            )
        else:
            preds_nparray = np.round(
                np.asarray(pred_pl[b].cpu().detach().numpy().reshape(-1)), 2
            )
            gt_nparray = gt.cpu().numpy()[0]
            text_pred_vs_gt = (
                f"Coverage: Pred {preds_nparray[:4]} GT   {gt_nparray[:-1]}\n "
                + f"Admissibility (GT only) {gt_nparray[-1]:.2f}"
            )

        text_pred_vs_gt = "LOW, soil, MID, HIGH \n" + text_pred_vs_gt
        print_stats(
            stats_file, plot_name + " " + text_pred_vs_gt, print_to_console=True
        )
        # We create an image with 5 or 6 subplots:
        # 1. original point cloud, 2. LV image, 3. pointwise prediction point cloud, 4. MV image, 5.Stratum probabilities point cloud, 6.(optional) HV image
        visualize(
            image_low_veg,
            image_med_veg,
            current_cloud,
            pred_pointwise,
            plot_name,
            stats_path,
            args,
            text_pred_vs_gt=text_pred_vs_gt,
            scores=likelihood,
            image_high_veg=image_high_veg,
            predictions=preds_nparray,
            gt=gt_nparray,
        )
        if args.nb_stratum == 3:
            visualize_article(
                image_low_veg,
                image_med_veg,
                image_high_veg,
                current_cloud,
                plot_name,
                stats_path,
                args,
                txt=text_pred_vs_gt,
            )


# TODO: correct rescaling to avoid artefact at borders
def rescale_xy_and_get_geotransformation_(
    xy_arr, plot_center_xy, args, image_low_veg, image_med_veg, image_high_veg
):
    # points_zone_center = (plot_center_xy.max(axis=1) + plot_center_xy.min(axis=1)) / 2
    # xy_arr = xy_arr * 10 + points_zone_center.reshape(
    #     -1, 1
    # )  # 10 is hardcoded normalization factor

    # geotransform reference : https://gdal.org/user/raster_data_model.html
    # top_left_x, pix_width_in_meters, _, top_left_y, pix_heighgt_in_meters (neg for north up picture)
    DIAM_METERS = 20
    geo = [
        plot_center_xy[0] - DIAM_METERS // 2,  # xmin
        DIAM_METERS / args.diam_pix,
        0,
        plot_center_xy[1] + DIAM_METERS // 2,  # ymax
        0,
        -DIAM_METERS / args.diam_pix,
        # negative b/c in geographic raster coordinates (0,0) is at top left
    ]

    if args.nb_stratum == 2:
        img_to_write = np.concatenate(([image_low_veg], [image_med_veg]), 0)
    else:
        img_to_write = np.concatenate(
            ([image_low_veg], [image_med_veg], [image_high_veg]), 0
        )
    return img_to_write, geo


def infer_and_project_on_rasters(current_cloud, args, pred_cloud):
    """
    We do raster reprojection, but we do not use torch scatter as we have to associate each value to a pixel
    current_cloud: (2, N) 2D tensor
     image_low_veg, image_med_veg, image_high_veg
    """

    # we get unique pixel coordinate to serve as group for raster prediction
    # TODO : extract somewhere else
    DIAM_METERS = 20

    scaling_factor = 10 * (
        args.diam_pix / DIAM_METERS
    )  # * pix/normalized_unit, using hardcoded factor of 10

    xy = current_cloud[:2, :]

    # width_height = (
    #     (xy.max(axis=1).values - xy.min(axis=1).values).view(2, 1).expand_as(xy)
    # )

    # xy = (
    #     torch.floor((xy + width_height / 2 + 0.0001) * scaling_factor)
    # ).int()  # values are between 0 and args.dim_pix-1, as expected

    xy = (
        torch.floor(
            (xy + 0.0001) * scaling_factor + torch.Tensor([[10], [10]]).expand_as(xy)
        )
    ).int()  # values are between 0 and args.dim_pix-1, as expected

    xy = xy.cpu().numpy()
    _, _, inverse = np.unique(xy.T, axis=0, return_index=True, return_inverse=True)

    # we get the values for each unique pixel and write them to rasters
    image_low_veg = np.full((args.diam_pix, args.diam_pix), np.nan)
    image_med_veg = np.full((args.diam_pix, args.diam_pix), np.nan)
    if args.nb_stratum == 3:
        image_high_veg = np.full((args.diam_pix, args.diam_pix), np.nan)
    else:
        image_high_veg = None
    for i in np.unique(inverse):
        where = np.where(inverse == i)[0]
        k, m = xy.T[where][0]
        maxpool = nn.MaxPool1d(len(where))
        max_pool_val = (
            maxpool(pred_cloud[:, where].unsqueeze(0)).cpu().detach().numpy().flatten()
        )

        if args.norm_ground:  # we normalize ground level coverage values
            proba_low_veg = max_pool_val[0] / (max_pool_val[:2].sum())
        else:  # we do not normalize anything, as bare soil coverage does not participate in absolute loss
            proba_low_veg = max_pool_val[0]
        proba_med_veg = max_pool_val[2]
        image_low_veg[m, k] = proba_low_veg
        image_med_veg[m, k] = proba_med_veg

        if args.nb_stratum == 3:
            proba_high_veg = max_pool_val[3]
            image_high_veg[m, k] = proba_high_veg
    # We flip along y axis as the 1st raster row starts with 0
    image_low_veg = np.flip(image_low_veg, axis=0)
    image_med_veg = np.flip(image_med_veg, axis=0)
    if args.nb_stratum == 3:
        image_high_veg = np.flip(image_high_veg, axis=0)
    return image_low_veg, image_med_veg, image_high_veg


# We create a tiff file with 2 or 3 stratum
def create_tiff(
    nb_channels, new_tiff_name, width, height, datatype, data_array, geotransformation
):
    # We set Lambert 93 projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    proj = srs.ExportToWkt()
    # We create a datasource
    driver_tiff = gdal.GetDriverByName("GTiff")
    dst_ds = driver_tiff.Create(new_tiff_name, width, height, nb_channels, datatype)
    dst_ds.SetGeoTransform(geotransformation)
    dst_ds.SetProjection(proj)
    if nb_channels == 1:
        outband = dst_ds.GetRasterBand(1)
        outband.WriteArray(data_array)
        outband.SetNoDataValue(np.nan)
        outband = None
    else:
        for ch in range(nb_channels):
            outband = dst_ds.GetRasterBand(ch + 1)
            outband.WriteArray(data_array[ch])
            # nodataval is needed for the first band only
            if ch == 0:
                outband.SetNoDataValue(np.nan)
            outband = None
    # write to file
    dst_ds.FlushCache()
    dst_ds = None
