import warnings
warnings.simplefilter(action='ignore')

import functools
import argparse
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
import pickle

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
from em_gamma.get_gamma_parameters_em import *

print(torch.cuda.is_available())
np.random.seed(42)
torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='model')

    # System Parameters
    parser.add_argument('--path', default="/home/ign.fr/ekalinicheva/DATASET_regression/", type=str,
                        help="Main folder directory")
    parser.add_argument('--gt_file', default="resultats_placettes_combo.csv", type=str, help="Name of GT *.cvs file")
    parser.add_argument('--plot_folder_name', default="placettes_combo", type=str, help="Name of GT *.csv file")
    parser.add_argument('--cuda', default=1, type=int, help="Whether we use cuda (1) or not (0)")
    parser.add_argument('--folds', default=5, type=int, help="Number of folds for cross validation model training")

    # Model Parameters
    parser.add_argument('--n_class', default=4, type=int,
                        help="Size of the model output vector. In our case 4 - different vegetation coverage types")
    parser.add_argument('--input_feats', default='xyzrgbnir', type=str,
                        help="Point features that we keep. in this code, we keep them all. permuting those letters will break everything. To be modified")
    parser.add_argument('--subsample_size', default=4096, type=int, help="Subsample cloud size")
    parser.add_argument('--diam_pix', default=32, type=int,
                        help="Size of the output stratum raster (its diameter in pixels)")
    parser.add_argument('--m', default=1, type=float,
                        help="Loss regularization. The weight of the negative loglikelihood loss in the total loss")
    parser.add_argument('--norm_ground', default=False, type=bool,
                        help="Whether we normalize low vegetation and bare soil values, so LV+BS=1 (True) or we keep unmodified LV value (False) (recommended)")
    parser.add_argument('--adm', default=True, type=bool, help="Whether we compute admissibility or not")
    parser.add_argument('--nb_stratum', default=3, type=int,
                        help="[2, 3] Number of vegetation stratum that we compute 2 - ground level + medium level; 3 - ground level + medium level + high level")
    parser.add_argument('--ECM_ite_max', default=5, type=int, help='Max number of EVM iteration')
    parser.add_argument('--NR_ite_max', default=10, type=int, help='Max number of Netwon-Rachson iteration')

    # Network Parameters
    parser.add_argument('--MLP_1', default=[32, 32], type=list,
                        help="Parameters of the 1st MLP block (output size of each layer). See PointNet article")
    parser.add_argument('--MLP_2', default=[64, 128], type=list,
                        help="Parameters of the 2nd MLP block (output size of each layer). See PointNet article")
    parser.add_argument('--MLP_3', default=[64, 32], type=list,
                        help="Parameters of the 3rd MLP block (output size of each layer). See PointNet article")
    parser.add_argument('--drop', default=0.4, type=float, help="Probability value of the DropOut layer of the model")
    parser.add_argument('--soft', default=True, type=bool,
                        help="Whether we use softmax layer for the model output (True) of sigmoid (False)")

    # Optimization Parameters
    parser.add_argument('--wd', default=0.001, type=float, help="Weight decay for the optimizer")
    parser.add_argument('--lr', default=5e-4, type=float, help="Learning rate")
    parser.add_argument('--step_size', default=50, type=int,
                        help="After this number of steps we decrease learning rate. (Period of learning rate decay)")
    parser.add_argument('--lr_decay', default=0.1, type=float,
                        help="We multiply learning rate by this value after certain number of steps (see --step_size). (Multiplicative factor of learning rate decay)")
    parser.add_argument('--n_epoch', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--n_epoch_test', default=5, type=int, help="We evaluate every -th epoch")
    parser.add_argument('--batch_size', default=20, type=int, help="Size of the training batch")

    args = parser.parse_args()

    assert (args.nb_stratum in [2, 3]), "Number of stratum should be 2 or 3!"
    assert (args.lr_decay < 1), "Learning rate decrease should be smaller than 1, as learning rate should decrease"

    las_folder = os.path.join(args.path, args.plot_folder_name) # folder with las files

    # We keep track of time and stats
    start_time = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(start_time)))
    run_name = str(time.strftime("%Y-%m-%d_%H%M%S"))

    # We write results to different folders depending on the chosen parameters
    if args.nb_stratum == 2:
        results_path = stats_path = os.path.join(args.path, "RESULTS/")
    else:
        results_path = stats_path = os.path.join(args.path,"RESULTS_3_stratum/")

    if args.adm:
        results_path = os.path.join(results_path, "admissibility/")
    else:
        results_path = os.path.join(results_path, "only_stratum/")

    stats_path = os.path.join(results_path, run_name) + "/"
    print("Results folder: ", stats_path)
    stats_file = os.path.join(stats_path, "stats.txt")
    create_dir(stats_path)
    # We open las files and create a dataset
    print("Loading data in memory")
    all_points, dataset, mean_dataset = open_las(las_folder)
    z_all = all_points[:, 2]
    args.z_max = np.max(z_all)   # maximum z value for data normalization, obtained from the normalized dataset analysis
    args.n_input_feats = len(args.input_feats)  # number of input features

    print_stats(stats_file, str(args), print_to_console=True)  # save all the args parameters

    # #   Parameters of gamma distributions for two stratum
    gamma_file = os.path.join(stats_path, "gamma.pkl")
    if not os.path.isfile(gamma_file):
        print("Computing gamma mixture (should only happen once)")
        params = get_gamma_parameters(z_all, args)
        with open(gamma_file, 'wb') as f:
            pickle.dump(params, f)
    else:
        print("Found precomputed Gamma parameters")
        with open(gamma_file, 'rb') as f:
            params = pickle.load(f)
    print_stats(stats_file, str(params), print_to_console=True)

    def train_full(args, fold_id):
        """The full training loop"""
        # initialize the model
        model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args)
        writer = SummaryWriter(results_path + "runs/"+run_name + "fold_" + str(fold_id) +"/")

        print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
        print(model)

        # define the classifier
        PCC = PointCloudClassifier(args)

        # define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)

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

    df_gt = pd.read_csv(os.path.join(args.path, args.gt_file), sep=',', header=0)  # we open GT file
    placettes = df_gt['Name'].to_numpy()    # We extract the names of the plots to create train and test list

    # We use several folds for cross validation (set the number in args)
    kf = KFold(n_splits=args.folds, random_state=42, shuffle=True)

    # None lists that will stock stats for each fold, so we can compute the mean at the end
    all_folds_loss_train_lists = None
    all_folds_loss_test_lists = None

    #cross-validation
    fold_id = 1
    print("Starting cross-validation")
    for train_ind, test_ind in kf.split(placettes):
        print("Cross-validation FOLD = %d" % (fold_id))
        train_list = placettes[train_ind]
        test_list = placettes[test_ind]

        # generate the train and test dataset
        test_set = tnt.dataset.ListDataset(test_list,
                                           functools.partial(cloud_loader, dataset=dataset, df_gt=df_gt, train=False, args=args))
        train_set = tnt.dataset.ListDataset(train_list,
                                            functools.partial(cloud_loader, dataset=dataset, df_gt=df_gt, train=True, args=args))

        trained_model, final_train_losses_list, final_test_losses_list = train_full(args, fold_id)

        # save the trained model
        PATH = os.path.join(stats_path, "model_ss_" + str(args.subsample_size) + "_dp_" + str(args.diam_pix) + "_fold_" + str(fold_id) + ".pt")
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







