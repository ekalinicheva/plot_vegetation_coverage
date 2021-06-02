import warnings

warnings.simplefilter(action="ignore")

import functools
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import gamma
import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchnet as tnt
from sklearn.neighbors import NearestNeighbors

import torch.nn as nn
from scipy.special import digamma, polygamma

import matplotlib

# Weird behavior: loading twice in cell appears to remove an elsewise occuring error.
for i in range(2):
    try:
        matplotlib.use("TkAgg")  # rerun this cell if an error occurs.
    except:
        print("!")
import matplotlib.pyplot as plt


from torch_scatter import scatter_max, scatter_mean

print(torch.cuda.is_available())
np.random.seed(42)
torch.cuda.empty_cache()

# We import from other files
from config import args
from model.model import PointNet
from utils.useful_functions import *
from data_loader.loader import *
from utils.load_las_data import load_all_las_from_folder, open_metadata_dataframe
from model.loss_functions import *
from model.accuracy import *
from em_gamma.get_gamma_parameters_em import *
from train import train_full

print("Everything is imported")


print(torch.cuda.is_available())
np.random.seed(42)
torch.cuda.empty_cache()


def main():

    # Create the result folder
    create_new_experiment_folder(args)  # new paths are added to args

    # Load Las files for placettes
    (
        all_points_nparray,
        nparray_clouds_dict,
        xy_averages_dict,
    ) = load_all_las_from_folder(args.las_placettes_folder_path)
    print("Our dataset contains " + str(len(nparray_clouds_dict)) + " plots.")

    # Load ground truth csv file
    # Name, 'COUV_BASSE', 'COUV_SOL', 'COUV_INTER', 'COUV_HAUTE', 'ADM'
    df_gt, placettes_names = open_metadata_dataframe(
        args, pl_id_to_keep=nparray_clouds_dict.keys()
    )
    print(df_gt.head())

    # Fit a mixture of 2 gamma distribution if not already done
    z_all = all_points_nparray[:, 2]
    args.z_max = np.max(
        z_all
    )  # maximum z value for data normalization, obtained from the normalized dataset analysis
    args.n_input_feats = len(args.input_feats)  # number of input features
    print_stats(
        args.stats_file, str(args), print_to_console=True
    )  # save all the args parameters
    params = run_or_load_em_analysis(z_all, args)
    print_stats(args.stats_file, str(params), print_to_console=True)

    # We use several folds for cross validation (set the number in args)
    kf = KFold(n_splits=args.folds, random_state=42, shuffle=True)

    # None lists that will stock stats for each fold, so we can compute the mean at the end
    all_folds_loss_train_lists = None
    all_folds_loss_test_lists = None

    # cross-validation
    start_time = time.time()
    fold_id = 1
    cloud_info_list_by_fold = {}
    print("Starting cross-validation")
    for train_ind, test_ind in kf.split(placettes_names):
        print("Cross-validation FOLD = %d" % (fold_id))
        train_list = placettes_names[train_ind]
        test_list = placettes_names[test_ind]

        # generate the train and test dataset
        test_set = tnt.dataset.ListDataset(
            test_list,
            functools.partial(
                cloud_loader,
                dataset=nparray_clouds_dict,
                df_gt=df_gt,
                train=False,
                args=args,
            ),
        )
        train_set = tnt.dataset.ListDataset(
            train_list,
            functools.partial(
                cloud_loader,
                dataset=nparray_clouds_dict,
                df_gt=df_gt,
                train=True,
                args=args,
            ),
        )

        # TRAINING on fold
        (
            trained_model,
            final_train_losses_list,
            final_test_losses_list,
            cloud_info_list,
        ) = train_full(
            args, fold_id, train_set, test_set, test_list, xy_averages_dict, params
        )

        cloud_info_list_by_fold[fold_id] = cloud_info_list

        # save the trained model
        PATH = os.path.join(
            args.stats_path,
            "model_ss_"
            + str(args.subsample_size)
            + "_dp_"
            + str(args.diam_pix)
            + "_fold_"
            + str(fold_id)
            + ".pt",
        )
        torch.save(trained_model, PATH)

        # We compute stats per fold
        all_folds_loss_train_lists, all_folds_loss_test_lists = stats_per_fold(
            all_folds_loss_train_lists,
            all_folds_loss_test_lists,
            final_train_losses_list,
            final_test_losses_list,
            args.stats_file,
            fold_id,
            args,
        )

        print_stats(
            args.stats_file,
            "training time "
            + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))),
            print_to_console=True,
        )
        fold_id += 1
        if args.mode == "DEV" and fold_id >= 2:
            break

    stats_for_all_folds(
        all_folds_loss_train_lists, all_folds_loss_test_lists, args.stats_file, args
    )

    # cloud_info_list_by_fold
    cloud_info_list_all_folds = [
        dict(p, **{"fold_id": fold_id})
        for fold_id, infos in cloud_info_list_by_fold.items()
        for p in infos
    ]
    df_inference = pd.DataFrame(cloud_info_list_all_folds)
    inference_path = os.path.join(args.stats_path, "PCC_inference_all_placettes.csv")
    df_inference["error_veg_b"] = (
        df_inference["pred_veg_b"] - df_inference["vt_veg_b"]
    ).abs()
    df_inference["error_veg_moy"] = (
        df_inference["pred_veg_moy"] - df_inference["vt_veg_moy"]
    ).abs()
    df_inference.to_csv(inference_path, index=False)  # TODO: remove just in case
    df_inference["error_veg_b_and_moy"] = (
        df_inference["error_veg_b"] + df_inference["error_veg_moy"]
    ) / 2
    df_inference.to_csv(inference_path, index=False)


if __name__ == "__main__":
    main()
