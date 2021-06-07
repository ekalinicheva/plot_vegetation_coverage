# Dependency imports
import os
from math import cos, pi, ceil
from utils.useful_functions import create_dir
import pandas as pd
import json
import seaborn as sns
from sys import getsizeof

sns.set()
import matplotlib.pyplot as plt
import matplotlib

from shapely.prepared import prep
from shapely.geometry import Point, asMultiPoint


# We import from other files
from utils.create_final_images import *
from data_loader.loader import *
from utils.reproject_to_2d_and_predict_plot_coverage import *
from model.loss_functions import *
from utils.load_las_data import load_single_las

np.random.seed(42)


def divide_parcel_las_and_get_disk_centers(
    args, las_folder, las_filename, save_fig_of_division=True
):
    """
    Identify centers of plots whose squares cover at least partially every pixel of the parcel
    We consider the square included in a plot with r=10m. Formula for width of
    the square is  W = 2 * (cos(45°) * r) since max radius in square equals r as well.
    We add an overlap of s*0.625 i.e. a pixel in currently produced plots of size 32 pix = 10
    :param las_folder: path
    :param las_filenae: "004000715-5-18.las" like string
    :returns:
        centers_nparray: a nparray of centers coordinates
        points_nparray: a nparray of full cloud coordinates
    Note: outputs are preprcessed by load_single_las but are not normalized
    """
    points_nparray, _ = load_single_las(las_folder, las_filename)
    # size_MB = getsizeof(round(getsizeof(points_nparray) / 1024 / 1024, 2))

    las_id = las_filename.split(".")[0]
    x_las, y_las = points_nparray[:, 0], points_nparray[:, 1]
    # subsample = False
    # if subsample:
    #     subsampling = 500
    #     subset = np.random.choice(points_nparray.shape[0],size=subsampling, replace=False)
    #     x_las = x_las[subset]
    #     y_las = y_las[subset]
    x_min = x_las.min()
    y_min = y_las.min()
    x_max = x_las.max()
    y_max = y_las.max()

    # Get or calculate dimensions of disk and max square in said disk
    plot_radius_meters = 10  # This is hardcoded, but should not change at any time.
    cos_of_45_degrees = cos(pi / 4)
    within_circle_square_width_meters = 2 * cos_of_45_degrees * plot_radius_meters
    plot_diameter_in_pixels = args.diam_pix  # 32 by default
    plot_diameter_in_meters = 2 * plot_radius_meters
    s = 1  # size of overlap in pixels
    square_xy_overlap = (
        s * plot_diameter_in_meters / plot_diameter_in_pixels
    )  # 0.625 by default
    movement_in_meters = within_circle_square_width_meters - square_xy_overlap

    print(
        f"Square dimensions are {within_circle_square_width_meters:.2f}m*{within_circle_square_width_meters:.2f}m"
        + f"but we move {movement_in_meters:.2f}m at a time to have {square_xy_overlap:.2f}m of overlap"
    )

    x_range_of_parcel_in_movements = ceil((x_max - x_min) / (movement_in_meters)) + 1
    y_range_of_parcel_in_movements = ceil((y_max - y_min) / (movement_in_meters)) + 1

    start_x = x_min + movement_in_meters / 4
    start_y = y_min + movement_in_meters / 4
    grid_pixel_xy_centers = [[start_x, start_y]]

    for i_dx in range(x_range_of_parcel_in_movements):
        current_x = start_x + i_dx * movement_in_meters  # move along x axis
        for i_dy in range(y_range_of_parcel_in_movements):
            current_y = start_y + i_dy * movement_in_meters  # move along y axis
            new_plot_center = [current_x, current_y]
            grid_pixel_xy_centers.append(new_plot_center)

    # visualization
    if save_fig_of_division:
        # we need to normalize coordinates points for easier visualization
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_min_c = x_min - x_center
        x_max_c = x_max - x_center
        y_min_c = y_min - y_center
        y_max_c = y_max - y_center

        # xy to dataframe for visualization
        coordinates = np.array(np.stack([x_las - x_center, y_las - y_center], axis=1))
        coordinates = pd.DataFrame(data=coordinates)
        coordinates.columns = ["x_n", "y_n"]

        sampling_size_for_kde = (
            10000  # fixed size which could lead to poor kde in large parcels.
        )
        if len(coordinates) > sampling_size_for_kde:
            coordinates = coordinates.sample(n=sampling_size_for_kde, replace=False)

        # centers to dataframe for visualization
        centers = np.array(grid_pixel_xy_centers - np.array([x_center, y_center]))
        centers = pd.DataFrame(data=centers)
        centers.columns = ["x_n", "y_n"]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"aspect": "equal"})
        ax.grid(False)
        ax.set_aspect("equal")  # Not working right now
        plt.xlim(x_min_c - 5, x_max_c + 5)
        plt.ylim(y_min_c - 5, y_max_c + 5)
        plt.ylabel("y_n", rotation=0)
        plt.title(
            f'Cutting in r=10m plots for parcel "{las_id}"'
            + f"\n Contained squares: W={within_circle_square_width_meters:.2f}m with overlap={square_xy_overlap:.2f}m (i.e. {s}pix)"
        )
        # plot kde of parcel
        fig.tight_layout()
        sns.kdeplot(
            data=coordinates,
            x="x_n",
            y="y_n",
            fill=True,
            alpha=0.5,
            color="g",
            clip=[[x_min_c, x_max_c], [y_min_c, y_max_c]],
        )  # thresh=0.2

        # plot disks and squares
        for _, (x, y) in centers.iterrows():
            a_circle = plt.Circle(
                (x, y), 10, fill=True, alpha=0.1, edgecolor="white", linewidth=1
            )
            ax.add_patch(a_circle)
            a_circle = plt.Circle(
                (x, y), 10, fill=False, edgecolor="white", linewidth=0.3
            )
            ax.add_patch(a_circle)
        #     a_square = matplotlib.patches.Rectangle((x-within_circle_square_width_meters/2,
        #                                              y-within_circle_square_width_meters/2),
        #                                             within_circle_square_width_meters,
        #                                             within_circle_square_width_meters,
        #                                             fill=True, alpha =0.1)
        #     ax.add_patch(a_square)
        sns.scatterplot(data=centers, x="x_n", y="y_n", s=5)

        # plot boundaries of parcel
        plt.axhline(
            y=y_min_c,
            xmin=x_min_c,
            xmax=x_max_c,
            color="black",
            alpha=0.6,
            linestyle="-",
        )
        plt.axhline(
            y=y_max_c,
            xmin=x_min_c,
            xmax=x_max_c,
            color="black",
            alpha=0.6,
            linestyle="-",
        )
        plt.axvline(
            x=x_min_c,
            ymin=y_min_c,
            ymax=y_max_c,
            color="black",
            alpha=0.6,
            linestyle="-",
        )
        plt.axvline(
            x=x_max_c,
            ymin=y_min_c,
            ymax=y_max_c,
            color="black",
            alpha=0.6,
            linestyle="-",
        )
        # fig.show()

        # saving
        parcel_id = las_filename.split(".")[0]

        cutting_plot_save_folder_path = os.path.join(args.stats_path, f"img/cuttings/")
        create_dir(cutting_plot_save_folder_path)
        cutting_plot_save_path = os.path.join(
            cutting_plot_save_folder_path, f"cut_{parcel_id}.png"
        )

        plt.savefig(cutting_plot_save_path, dpi=200)
        plt.clf()
        plt.close("all")

    return grid_pixel_xy_centers, points_nparray


def extract_points_within_disk(points_nparray, center, radius=10):
    """From a (2, N) np.array with x, y as first features, extract points within radius
    from the center = (x_center, y_center)"""
    xy = points_nparray[:, :2]  # (N, 2)
    contained_points = points_nparray[
        ((xy - center) ** 2).sum(axis=1) <= (radius * radius)
    ]  # (N, f)

    return contained_points


def save_centers_dict(centers_dict, centers_dict_path):
    with open(centers_dict_path, "w") as outfile:
        json.dump(centers_dict, outfile)


# def infer_from_single_cloud(model, PCC, cloud, args):
#     """
#     Returns: prediction at the pixel level
#     Cloud is a single cloud tensor with batch dimension = 1.
#     """
#     model.eval()
#     # loader = torch.utils.data.DataLoader(
#     #     test_set, collate_fn=cloud_collate, batch_size=1, shuffle=False
#     # )
#     # if PCC.is_cuda:
#     #     gt = gt.cuda()

#     pred_pointwise, pred_pointwise_b = PCC.run(model, cloud)  # compute the prediction

#     _, _, pred_pixels = project_to_2d(
#         pred_pointwise, cloud, pred_pointwise_b, PCC, args
#     )  # compute plot prediction

#     return pred_pointwise, pred_pixels


def create_geotiff_raster(
    args,
    xy_nparray,  # (2,N) tensor
    pred_pointwise,
    plot_points_tensor,  # (1,f,N) tensor
    plot_center,
    plot_name,
    add_weights_band=False,
):
    """ """
    # we do raster reprojection, but we do not use torch scatter as we have to associate each value to a pixel
    image_low_veg, image_med_veg, image_high_veg = infer_and_project_on_rasters(
        plot_points_tensor, args, pred_pointwise
    )

    # We normalize back x,y values to create a tiff file with
    img_to_write, geo = rescale_xy_and_get_geotransformation_(
        xy_nparray,
        plot_center,
        args,
        image_low_veg,
        image_med_veg,
        image_high_veg,
    )

    # we get hard rasters for medium veg, creating a fourth canal
    image_med_veg_hard = image_med_veg > 0.5  # TODO: define a bette strategy
    img_to_write = np.concatenate(
        [img_to_write, [image_med_veg_hard]], 0
    )  # (nb_canals, 32, 32)

    if add_weights_band:
        # Currently: linear variation with distance to center

        x = (np.arange(-16, 16, 1) + 0.5) / 32
        y = (np.arange(-16, 16, 1) + 0.5) / 32
        xx, yy = np.meshgrid(x, y, sparse=True)
        image_weights = 1 - np.sqrt(xx ** 2 + yy ** 2)

        img_to_write = np.concatenate(
            [img_to_write, [image_weights]], 0
        )  # (nb_canals, 32, 32)

    # define save paths
    tiff_folder_path = os.path.join(
        args.stats_path,
        f"img/rasters/{plot_name}/",
    )
    create_dir(tiff_folder_path)
    tiff_file_path = os.path.join(
        tiff_folder_path,
        f"predictions_{plot_name}_X{plot_center[0]:.0f}_Y{plot_center[1]:.0f}.tif",
    )

    create_tiff(
        nb_channels=args.nb_stratum
        + 1
        + add_weights_band,  # stratum + hard raster + weights
        new_tiff_name=tiff_file_path,
        width=args.diam_pix,
        height=args.diam_pix,
        datatype=gdal.GDT_Float32,
        data_array=img_to_write,
        geotransformation=geo,
    )
