import warnings
warnings.simplefilter(action='ignore')

import functools
import argparse
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold


# We import from other files
from train import *
from test import *
from model.model import PointNet
from utils.point_cloud_classifier import PointCloudClassifier
from utils.useful_functions import *
from data_loader.loader import *
from utils.open_las import open_las
from model.loss_functions import *
from model.accuracy import *


print(torch.cuda.is_available())
np.random.seed(42)


torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='model')

# Parameters you should not modify
parser.add_argument('--n_class', default=4, type=int, help="Size of the model output vector. In our case 4 - different vegetation coverage types")
parser.add_argument('--input_feats', default='xyzrgbnir', type=str, help="Point features that we keep. in this code, we keep them all. permuting those letters will break everything. To be modified")
parser.add_argument('--MLP_1', default=[32, 32], type=list, help="Parameters of the 1st MLP block (output size of each layer). See PointNet article")
parser.add_argument('--MLP_2', default=[64, 128], type=list, help="Parameters of the 2nd MLP block (output size of each layer). See PointNet article")
parser.add_argument('--MLP_3', default=[64, 32], type=list, help="Parameters of the 3rd MLP block (output size of each layer). See PointNet article")
parser.add_argument('--subsample_size', default=4096, type=int, help="Subsample cloud size")
parser.add_argument('--cuda', default=1, type=int, help="Whether we use cuda (1) or not (0)")
parser.add_argument('--wd', default=0.001, type=float, help="Weight decay for the optimizer")
parser.add_argument('--diam_pix', default=32, type=int, help="Size of the output stratum raster (its diameter in pixels)")
parser.add_argument('--drop', default=0.4, type=float, help="Probability value of the DropOut layer of the model")
parser.add_argument('--soft', default=True, type=bool, help="Whether we use sortmax layer for the model output (True) of sigmoid (False)")
parser.add_argument('--m', default=1, type=float, help="Loss regularization. The weight of the negative loglikelihood loss in the total loss")


# args.n_class = 4                # size of the output vector
# args.input_feats = 'xyzrgbnir'  # point features that we keep. in this code, we keep them all. permuting those letters will break everything
# args.n_input_feats = len(args.input_feats)
# args.MLP_1 = [32, 32]           # parameters of mlp blocks (output size of each layer)
# args.MLP_2 = [64, 128]
# args.MLP_3 = [64, 32]
# args.subsample_size = 2048 * 2  # subsample cloud size
# args.cuda = 1                   # we use cuda
# args.lr = 5e-4                  # learning rate
# args.wd = 0.001                 # weight decay for the optimizer
# args.diam_pix = 32              # size of the output stratum raster
# args.drop = 0.4                 # dropout layer probability
# args.soft = True                # Wheather we use sortmax layer of sigmoid
# args.m = 1                      # loss regularization


#
# # Parameters you can modify
# args.lr = 5e-4                  # learning rate
# args.n_epoch = 100             # number of training epochs
# args.n_epoch_test = 5           # we evaluate every -th epoch
# args.batch_size = 20
# args.z_max = 24                 # maximum z value for data normalization, obtained from the normalized dataset analysis
# args.adm = True                # wheather we compute admissibility or not
# args.nb_stratum = 3             # [2, 3] Number of vegetation stratum that we compute 2 - ground level + medium level; 3 - ground level + medium level + high level
# args.folds = 5                  # number of folds to train the model
# args.path = "/home/ign.fr/ekalinicheva/DATASET_regression/"  # folder directory
# # args.path = "/home/ekaterina/DATASET_regression/"
# args.gt_file = "resultats_placettes_combo.csv"          # GT csv file
# args.plot_folder_name = "placettes_combo/"

# Parameters you can modify
# Parameters you can modify
parser.add_argument('--lr', default=5e-4, type=float, help="Learning rate")
parser.add_argument('--n_epoch', default=100, type=int, help="Number of training epochs")
parser.add_argument('--n_epoch_test', default=5, type=int, help="We evaluate every -th epoch")
parser.add_argument('--batch_size', default=20, type=int, help="Size of the training batch")
parser.add_argument('--adm', default=True, type=bool, help="Whether we compute admissibility or not")
parser.add_argument('--nb_stratum', default=3, type=int, help="[2, 3] Number of vegetation stratum that we compute 2 - ground level + medium level; 3 - ground level + medium level + high level")
parser.add_argument('--folds', default=5, type=int, help="Number of folds for cross validation model training")
parser.add_argument('--path', default="/home/ign.fr/ekalinicheva/DATASET_regression/", type=str, help="Main folder directory")
parser.add_argument('--gt_file', default="resultats_placettes_combo.csv", type=str, help="Name of GT *.cvs file")
parser.add_argument('--plot_folder_name', default="placettes_combo", type=str, help="Name of GT *.cvs file")

args = parser.parse_args()


args.n_input_feats = len(args.input_feats)  # number input features
args.z_max = 24                 # maximum z value for data normalization, obtained from the normalized dataset analysis


assert(args.nb_stratum in [2, 3]), "Number of stratum should be 2 or 3!"
assert(args.n_epoch % args.n_epoch_test == 0), "Number of train epoch should be dividable by number of test epoch"

las_folder = args.path + args.plot_folder_name + "/"            # folder with las files

# We keep track of everything (time ans stats)
start_time = time.time()
print(time.strftime("%H:%M:%S", time.gmtime(start_time)))

run_name = str(time.strftime("%Y-%m-%d_%H%M%S"))

# We write results to different folders depending on the chosen parameters
if args.nb_stratum == 2:
    results_path = stats_path = args.path + "RESULTS/"
else:
    results_path = stats_path = args.path + "RESULTS_3_stratum/"

if args.adm:
    results_path = results_path + "admissibility/"
else:
    results_path = results_path + "only_stratum/"


stats_path = results_path + run_name + "/"
print(stats_path)

stats_file = stats_path + "stats.txt"
create_dir(stats_path)
print_stats(stats_file, str(args), print_to_console=True) # save all the args parameters


def main():
    # We open las files and create a dataset
    all_points, dataset, mean_dataset = open_las(las_folder)

    # #   Parameters of gamma distributions for two stratum
    # params = get_gamma_params(z_all)

    params = {'phi': 0.47535938, 'a_g': 0.1787963, 'a_v': 3.5681678,
              'loc_g': 0, 'loc_v': 0, 'scale_g': 0.48213167,
              'scale_v': 1.46897231}


    def train_full(args, fold_id):
        """The full training loop"""
        # initialize the model
        model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args)
        writer = SummaryWriter(results_path + "runs/"+run_name + "fold_" + str(fold_id) +"/")

        print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
        print(model)
        print_stats(stats_file, str(model), print_to_console=False)

        # define the classifier
        PCC = PointCloudClassifier(args)

        # define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

        for i_epoch in range(args.n_epoch):
            scheduler.step()

            # train one epoch
            train_losses = train(model, PCC, train_set, params, optimizer, args)
            writer = write_to_writer(writer, args, i_epoch, train_losses, train=True)

            if (i_epoch + 1) % args.n_epoch_test == 0:
                if (i_epoch + 1) == args.n_epoch:   # if last epoch, we creare 2D images with points projections
                    test_losses = eval(model, PCC, test_set, params, args, test_list, mean_dataset, stats_path, stats_file, last_epoch=True)
                else:
                    test_losses = eval(model, PCC, test_set, params, args, test_list, mean_dataset, stats_path, stats_file)
                gc.collect()
                writer = write_to_writer(writer, args, i_epoch, test_losses, train=False)
        writer.flush()

        final_train_losses_list = train_losses
        final_test_losses_list = test_losses
        return model, final_train_losses_list, final_test_losses_list



    df_gt = pd.read_csv(args.path + args.gt_file, sep=',', header=0)  # we open GT file
    placettes = df_gt['Name'].to_numpy()    # We extract the names of the plots to create train and test list


    # We use several folds for cross validation (set the number in args)
    kf = KFold(n_splits=args.folds, random_state=42, shuffle=True)

    # None lists that will stock stats for each fold, so we can compute the mean at the end
    all_folds_loss_train_lists = None
    all_folds_loss_test_lists = None


    fold_id = 1
    for train_ind, test_ind in kf.split(placettes):
        train_list = placettes[train_ind]
        test_list = placettes[test_ind]

        # generate the train and test dataset
        test_set = tnt.dataset.ListDataset(test_list,
                                           functools.partial(cloud_loader, dataset=dataset, df_gt=df_gt, train=False, args=args))
        train_set = tnt.dataset.ListDataset(train_list,
                                            functools.partial(cloud_loader, dataset=dataset, df_gt=df_gt, train=True, args=args))

        trained_model, final_train_losses_list, final_test_losses_list = train_full(args, fold_id)

        # save the trained model
        PATH = stats_path + "model_ss_" + str(args.subsample_size) + "_dp_" + str(args.diam_pix) + "_fold_" + str(fold_id) + ".pt"
        torch.save(trained_model, PATH)

        # We compute stats per fold
        all_folds_loss_train_lists, all_folds_loss_test_lists = stats_per_fold(all_folds_loss_train_lists, all_folds_loss_test_lists, final_train_losses_list, final_test_losses_list, stats_file, fold_id, args)

        print_stats(stats_file,
                    "training time " + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))),
                    print_to_console=True)
        fold_id += 1

    # compute mean stats for 5 folds
    stats_for_all_folds(all_folds_loss_train_lists, all_folds_loss_test_lists, stats_file, args)


if __name__ == "__main__":
    main()







