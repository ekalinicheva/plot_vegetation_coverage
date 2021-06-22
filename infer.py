import warnings

warnings.simplefilter(action="ignore")

import numpy as np
import os
import torch
import matplotlib
from tqdm import tqdm

# Weird behavior: loading twice in cell appears to remove an elsewise occuring error.
for i in range(2):
    try:
        matplotlib.use("TkAgg")  # rerun this cell if an error occurs.
    except:
        print("!")

print(torch.cuda.is_available())
np.random.seed(42)
torch.cuda.empty_cache()

# We import from other files
from config import args
from utils.useful_functions import create_new_experiment_folder, print_stats
from data_loader.loader import rescale_cloud_data
from utils.point_cloud_classifier import PointCloudClassifier
from model.infer_utils import (
    divide_parcel_las_and_get_disk_centers,
    extract_points_within_disk,
    create_geotiff_raster,
    merge_geotiff_rasters,
)
from utils.load_las_data import transform_features_of_plot_cloud


args.z_max = 24.14  # the TRAINING args should be loaded from stats.csv/txt...


print("Everything is imported")

print(torch.cuda.is_available())
np.random.seed(42)
torch.cuda.empty_cache()

# Create the result folder
create_new_experiment_folder(args, infer_mode=True)  # new paths are added to args

# find parcels .LAS files
las_folder = args.las_parcelles_folder_path
las_filenames = os.listdir(las_folder)
las_filenames = [l for l in las_filenames if l.lower().endswith(".las")]

# define the classifier
model = torch.load(args.trained_model_path)
print_stats(
    args.stats_file,
    f"trained model was loaded from {args.trained_model_path}",
    print_to_console=True,
)
model.eval()
PCC = PointCloudClassifier(args)

for las_filename in las_filenames:
    # TODO : remove this debug condition
    # if args.mode == "DEV":
    #     if las_filename != "004000715-5-18.las":  # small
    #         continue
    # if args.mode == "DEV":
    #     if las_filename == "004009611-11-13.las":  # too big
    #         continue

    print_stats(
        args.stats_file,
        f"Inference on parcel file {las_filename}",
        print_to_console=True,
    )

    # Divide parcel into plots
    (
        grid_pixel_xy_centers,
        parcel_points_nparray,
    ) = divide_parcel_las_and_get_disk_centers(
        args, las_folder, las_filename, save_fig_of_division=True
    )
    print(f"File {las_filename} with shape {parcel_points_nparray.shape}")
    # TODO: replace this loop by a cleaner ad-hoc DataLoader

    idx_for_break = 0  # TODO: remove
    idx_for_break_max = 40  # np.inf
    centers = tqdm(
        grid_pixel_xy_centers, desc="Centers for parcel in {las_filename}", leave=True
    )
    for plot_center in centers:

        plots_point_nparray = extract_points_within_disk(
            parcel_points_nparray, plot_center
        )

        # infer if non-empty selection
        # TODO: remove print
        # print(plots_point_nparray.shape)

        if plots_point_nparray.shape[0] > 0:

            # TODO: Clarityt: make operations on the same axes instead of transposing inbetween
            plots_point_nparray = transform_features_of_plot_cloud(
                plots_point_nparray, args.znorm_radius_in_meters
            )
            plots_point_nparray = plots_point_nparray.transpose()
            plots_point_nparray = rescale_cloud_data(plots_point_nparray, args)

            # add a batch dim before trying out dataloader
            plots_point_nparray = np.expand_dims(plots_point_nparray, axis=0)
            plot_points_tensor = torch.from_numpy(plots_point_nparray)

            # compute pointwise prediction
            # TODO: upsample points before predictions
            pred_pointwise, _ = PCC.run(model, plot_points_tensor)

            # pred_pointwise was permuted from (N_scores, N_points) to (N_points, N_scores) for some reasons at the end of PCC.run
            pred_pointwise = pred_pointwise.permute(1, 0)

            plot_name = las_filename.split(".")[0]
            create_geotiff_raster(
                args,
                plots_point_nparray[0, :2, :],  # (2, N_points) xy nparray
                pred_pointwise,
                plot_points_tensor[0, :, :],  # cloud 2D tensor (N_feats, N_points)
                plot_center,
                plot_name,
                add_weights_band=True,
            )
            idx_for_break += 1
            if idx_for_break >= idx_for_break_max:
                break

    merge_geotiff_rasters(args, plot_name)
