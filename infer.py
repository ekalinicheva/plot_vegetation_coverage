import warnings

warnings.simplefilter(action="ignore")

import numpy as np
import os
import torch
import torchnet as tnt
import torch.nn as nn
import matplotlib

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
from utils.useful_functions import *
from data_loader.loader import *
from utils.load_las_data import load_all_las_from_folder, open_metadata_dataframe
from model.loss_functions import *
from model.accuracy import *
from em_gamma.get_gamma_parameters_em import *
from model.model import PointNet
from utils.point_cloud_classifier import PointCloudClassifier
from model.infer_utils import (
    divide_parcel_las_and_get_disk_centers,
    extract_points_within_disk,
    create_geotiff_raster,
)
from utils.reproject_to_2d_and_predict_plot_coverage import project_to_2d


from config import args

args.z_max = 24.14  # the TRAINING args should be loaded !


print("Everything is imported")

print(torch.cuda.is_available())
np.random.seed(42)
torch.cuda.empty_cache()

# Create the result folder
create_new_experiment_folder(args, infer_mode=True)  # new paths are added to args
las_folder = args.las_parcelles_folder_path
las_filenames = os.listdir(las_folder)


# define the classifier
model = torch.load(args.trained_model_path)
print(f"trained model was loaded from {args.trained_model_path}")
model.eval()
# load the model
# print(
#     "Total number of parameters: {}".format(
#         sum([p.numel() for p in model.parameters()])
#     )
# )
# print(model)
PCC = PointCloudClassifier(args)

for las_filename in las_filenames:
    if args.mode == "DEV":
        if las_filename != "004000715-5-18.las":
            continue
    # her we divide all parcels into plots
    grid_pixel_xy_centers, points_nparray = divide_parcel_las_and_get_disk_centers(
        args, las_folder, las_filename, save_fig_of_division=True
    )

    # TODO: replace this loop by a cleaner ad-hoc DataLoader

    idx_for_break = 0  # TODO: remove
    for plot_center in grid_pixel_xy_centers[40:]:
        contained_points_nparray = extract_points_within_disk(
            points_nparray, plot_center
        )
        # infer if non-empty selection
        if contained_points_nparray.shape[0] > 0:
            las_id = las_filename.split(".")[0]

            # TODO: remove print
            print(contained_points_nparray.shape)

            contained_points_nparray = contained_points_nparray.transpose()
            contained_points_nparray = normalize_cloud_data(
                contained_points_nparray, args
            )

            # add a batch dim before trying out dataloader
            contained_points_nparray = np.expand_dims(contained_points_nparray, axis=0)
            contained_points_tensor = torch.from_numpy(contained_points_nparray)

            # infer
            model.eval()
            # compute pointwise prediction
            pred_pointwise, _ = PCC.run(model, contained_points_tensor)

            # pred_pointwise was permuted from (scores_nb, pts_nb) to (pts_nb, scores_nb) for some reasons at the end of PCC.run
            pred_pointwise = pred_pointwise.permute(1, 0)
            create_geotiff_raster(
                args,
                contained_points_nparray[0, :2, :],  # xy nparray
                pred_pointwise,
                contained_points_tensor,
                plot_center,
                las_id,
                add_weights_band=False,
            )
            idx_for_break += 1
            if idx_for_break > 10:
                break
