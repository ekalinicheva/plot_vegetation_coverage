import os
import numpy as np
import pandas as pd
from laspy.file import File
from sklearn.neighbors import NearestNeighbors
import warnings
import random

random.seed(0)
from random import random, shuffle


warnings.simplefilter(action="ignore")


def load_all_las_from_folder(args):
    las_folder = args.las_placettes_folder_path

    # We open las files and create a training dataset
    nparray_clouds_dict = {}  # dict to store numpy array with each plot separately
    xy_averages_dict = (
        {}
    )  # we keep track of plots means to reverse the normalisation in the future

    # We iterate through las files and transform them to np array
    las_files = os.listdir(las_folder)
    las_files = [l for l in las_files if l.lower().endswith(".las")]

    if args.mode == "DEV":
        shuffle(las_files)
        las_files = las_files[: (5 * 5)]  # nb plot by fold

        # #LOOK AT SPECIFIC FILES
        # las_files = las_files[:10] + [
        #     l
        #     for l in las_files
        #     if any(n in l for n in ["OBS15", "F68", "2021_POINT_OBS2"])
        # ]

        print(las_files)
    all_points_nparray = np.empty((0, args.nb_feats_for_train))
    for las_file in las_files:

        # Parse LAS files
        points_nparray, xy_averages = load_single_las(las_folder, las_file)

        all_points_nparray = np.append(all_points_nparray, points_nparray, axis=0)
        nparray_clouds_dict[os.path.splitext(las_file)[0]] = points_nparray
        xy_averages_dict[os.path.splitext(las_file)[0]] = xy_averages

    return all_points_nparray, nparray_clouds_dict, xy_averages_dict


def load_single_las(las_folder, las_file):
    # Parse LAS files
    las = File(os.path.join(las_folder, las_file), mode="r")
    x_las = las.X
    y_las = las.Y
    z_las = las.Z
    r = las.Red
    g = las.Green
    b = las.Blue
    nir = las.nir
    intensity = las.intensity
    return_nb = las.return_num
    points_placette = np.asarray(
        [x_las / 100, y_las / 100, z_las / 100, r, g, b, nir, intensity, return_nb]
    ).T  # we divide by 100 as all the values in las are in cm

    # There is a file with 2 points 60m above others (maybe birds), we delete these points
    if las_file == "Releve_Lidar_F70.las":
        points_placette = points_placette[points_placette[:, 2] < 640]
    # We do the same for the intensity
    if las_file == "POINT_OBS8.las":
        points_placette = points_placette[points_placette[:, -2] < 32768]
    if las_file == "Releve_Lidar_F39.las":
        points_placette = points_placette[points_placette[:, -2] < 20000]

    # Add a feature:min-normalized using min-z of the plot
    zmin_plot = np.min(points_placette[:, 2])
    points_placette = np.append(
        points_placette, points_placette[:, 2:3] - zmin_plot, axis=1
    )

    # # We directly substract z_min at local level
    points_placette = normalize_z_with_minz_in_50cm_radius(points_placette)
    # points_placette = normalize_z_with_approximate_DTM(points_placette)

    # get the average
    xy_averages = [
        np.mean(x_las) / 100,
        np.mean(y_las) / 100,
    ]
    return points_placette, xy_averages


def normalize_z_with_minz_in_50cm_radius(points_placette):
    # # We directly substract z_min at local level
    xyz = points_placette[:, :3]
    knn = NearestNeighbors(500, algorithm="kd_tree").fit(xyz[:, :2])
    _, neigh = knn.radius_neighbors(xyz[:, :2], 0.5)
    z = xyz[:, 2]
    zmin_neigh = []
    for n in range(
        len(z)
    ):  # challenging to make it work without a loop as neigh has different length for each point
        zmin_neigh.append(np.min(z[neigh[n]]))
    points_placette[:, 2] = points_placette[:, 2] - zmin_neigh
    return points_placette


def normalize_z_with_approximate_DTM(points_placette, args):
    points_placette


def open_metadata_dataframe(args, pl_id_to_keep):
    """This opens the ground truth file. It completes if necessary admissibility value using ASP method."""

    df_gt = pd.read_csv(
        args.gt_file_path,
        sep=",",
        header=0,
    )  # we open GT file
    # Here, adapt columns names
    df_gt = df_gt.rename(args.coln_mapper_dict, axis=1)

    # Keep metadata for placettes we are considering
    df_gt = df_gt[df_gt["Name"].isin(pl_id_to_keep)]

    # Correct Soil value to have
    df_gt["COUV_SOL"] = 100 - df_gt["COUV_BASSE"]

    # his is ADM based on ASP definition - NOT USED at the moment
    if "ADM" not in df_gt:
        df_gt["ADM_BASSE"] = df_gt["COUV_BASSE"] - df_gt["NON_ACC_1"]
        df_gt["ADM_INTER"] = df_gt["COUV_HAUTE"] - df_gt["NON_ACC_2"]
        df_gt["ADM"] = df_gt[["ADM_BASSE", "ADM_INTER"]].max(axis=1)

        del df_gt["ADM_BASSE"]
        del df_gt["ADM_INTER"]

    # check that we have all columns we need
    assert all(
        coln in df_gt
        for coln in [
            "Name",
            "COUV_BASSE",
            "COUV_SOL",
            "COUV_INTER",
            "COUV_HAUTE",
            "ADM",
        ]
    )

    placettes_names = df_gt[
        "Name"
    ].to_numpy()  # We extract the names of the plots to create train and test list

    return df_gt, placettes_names
