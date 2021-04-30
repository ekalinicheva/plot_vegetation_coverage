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
from utils.create_final_images import *
from data_loader.loader import *
from utils.open_las import open_las
from model.loss_functions import *



print(torch.cuda.is_available())
np.random.seed(42)


torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='model')
args = parser.parse_args()
args.n_epoch = 200             # number of training epochs
args.n_epoch_test = 5           # we evaluate every -th epoch
args.batch_size = 20
args.n_class = 4                # size of the output vector
args.input_feats = 'xyzrgbnir'  # point features that we keep. in this code, we keep them all. permuting those letters will break everything
args.n_input_feats = len(args.input_feats)
args.MLP_1 = [32, 32]           # parameters of mlp blocks (output size of each layer)
args.MLP_2 = [64, 128]
args.MLP_3 = [64, 32]
args.subsample_size = 2048 * 2  # subsample cloud size
args.cuda = 1                   # we use cuda
args.lr = 1e-3                  # learning rate
args.wd = 0.001                 # weight decay for the optimizer
args.diam_pix = 32              # size of the output stratum raster
args.drop = 0.4                 # dropout layer probability
args.soft = True                # Wheather we use sortmax layer of sigmoid
args.m = 1                      # loss regularization
args.z_max = 24                 # maximum z value for data normalization
args.adm = False                # wheather we compute admissibility or not
args.nb_stratum = 3             # [2, 3] Number of vegetation stratum that we compute 2 - ground level + medium level; 3 - ground level + medium level + high level



path = "/home/ign.fr/ekalinicheva/DATASET_regression/"  # folder directory
# path = "/home/ekaterina/DATASET_regression/"

# gt_file = "resultats_placettes_recherche1.csv"          # GT csv file
# las_folder = path + "placettes/"                        # folder with las files


gt_file = "resultats_placettes_combo.csv"          # GT csv file
las_folder = path + "placettes_combo/"             # folder with las files


# We keep track of everything (time ans stats)
start_time = time.time()
print(time.strftime("%H:%M:%S", time.gmtime(start_time)))

run_name = str(time.strftime("%Y-%m-%d_%H%M%S"))
if args.nb_stratum == 2:
    results_path = stats_path = path + "RESULTS/"
else:
    results_path = stats_path = path + "RESULTS_3_stratum/"

if args.adm:
    results_path = results_path + "admissibility/"
else:
    results_path = results_path + "only_stratum/"


stats_path = results_path + run_name + "/"
print(stats_path)

stats_file = stats_path + "stats.txt"
create_dir(stats_path)
print_stats(stats_file, str(args), print_to_console=True) # save all the args parameters


#   Parameters of gamma distributions
# params = {'phi': 0.6261676951828907, 'a_g': 2.7055041947617378, 'a_v': 2.455886147981429, 'loc_g': -0.01741796054681828,
#           'loc_v': 0.06129981952874307, 'scale_g': 0.06420226677528383, 'scale_v': 2.2278027968946112}




def main():
    # We open las files and create a dataset
    all_points, dataset, mean_dataset = open_las(las_folder)

    # #   Parameters of gamma distributions for two stratum
    # params = get_gamma_params(z_all)

    params = {'phi': 0.47535938, 'a_g': 0.1787963, 'a_v': 3.5681678,
              'loc_g': 0, 'loc_v': 0, 'scale_g': 0.48213167,
              'scale_v': 1.46897231}




    def train_full(args):
        """The full training loop"""
        # initialize the model
        model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args)
        writer = SummaryWriter(results_path + "runs/")

        print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
        print(model)
        print_stats(stats_file, str(model), print_to_console=False)

        # define the classifier
        PCC = PointCloudClassifier(args)

        # define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

        TESTCOLOR = '\033[104m'
        TRAINCOLOR = '\033[100m'
        NORMALCOLOR = '\033[0m'

        for i_epoch in range(args.n_epoch):
            scheduler.step()

            # train one epoch
            loss_train, loss_train_abs, loss_train_log, loss_train_adm = train(model, PCC, train_set, params, optimizer, args)
            if args.adm:
                print(TRAINCOLOR + 'Epoch %3d -> Train Loss: %1.4f Train Loss Abs: %1.4f Train Loss Log: %1.4f Train Loss Adm: %1.4f' % (i_epoch, loss_train, loss_train_abs, loss_train_log, loss_train_adm) + NORMALCOLOR)
                writer.add_scalar('Loss/train_abs_adm', loss_train_adm, i_epoch + 1)
            else:
                print(TRAINCOLOR + 'Epoch %3d -> Train Loss: %1.4f Train Loss Abs: %1.4f Train Loss Log: %1.4f' % (i_epoch, loss_train, loss_train_abs, loss_train_log) + NORMALCOLOR)
            writer.add_scalar('Loss/train', loss_train, i_epoch + 1)
            writer.add_scalar('Loss/train_abs', loss_train_abs, i_epoch + 1)
            writer.add_scalar('Loss/train_log', loss_train_log, i_epoch + 1)



            if (i_epoch + 1) % args.n_epoch_test == 0:
                if (i_epoch + 1) == args.n_epoch:   # if last epoch, we creare 2D images with points projections
                    loss_test, loss_test_abs, loss_test_log, loss_test_abs_gl, loss_test_abs_ml, loss_test_abs_hl, loss_test_adm = eval(model, PCC, test_set, params, args, test_list, mean_dataset, stats_path, stats_file, last_epoch=True)
                else:
                    loss_test, loss_test_abs, loss_test_log, _, _, _, loss_test_adm = eval(model, PCC, test_set, params, args, test_list, mean_dataset, stats_path, stats_file)
                gc.collect()
                if args.adm:
                    print(TESTCOLOR + 'Test Loss: %1.4f Test Loss Abs: %1.4f Test Loss Log: %1.4f Test Loss Adm: %1.4f' % (loss_test, loss_test_abs, loss_test_log, loss_test_adm) + NORMALCOLOR)
                    writer.add_scalar('Loss/test_abs_adm', loss_test_adm, i_epoch + 1)
                else:
                    print(TESTCOLOR + 'Test Loss: %1.4f Test Loss Abs: %1.4f Test Loss Log: %1.4f' % (
                    loss_test, loss_test_abs, loss_test_log) + NORMALCOLOR)
                writer.add_scalar('Loss/test', loss_test, i_epoch + 1)
                writer.add_scalar('Loss/test_abs', loss_test_abs, i_epoch + 1)
                writer.add_scalar('Loss/test_log', loss_test_log, i_epoch + 1)
        writer.flush()
        return model, loss_train, loss_train_abs, loss_train_log, loss_train_adm, loss_test, loss_test_abs, loss_test_log, loss_test_abs_gl, loss_test_abs_ml, loss_test_abs_hl, loss_test_adm



    df_gt = pd.read_csv(path + gt_file, sep=',', header=0)  # we open GT file
    placettes = df_gt['Name'].to_numpy()    # We extract the names of the plots to create train and test list


    # We use 5 folds cross validation
    folds = 5
    kf = KFold(n_splits=folds, random_state=42, shuffle=True)

    # We keep track of stats per fold
    loss_train_list = []
    loss_train_abs_list = []
    loss_train_log_list = []
    loss_train_adm_list = []

    loss_test_list = []
    loss_test_abs_list = []
    loss_test_log_list = []
    loss_test_abs_gl_list = []
    loss_test_abs_ml_list = []
    loss_test_abs_hl_list = []
    loss_test_adm_list = []



    fold_id = 1
    for train_ind, test_ind in kf.split(placettes):
        train_list = placettes[train_ind]
        test_list = placettes[test_ind]

        # generate the train and test dataset
        test_set = tnt.dataset.ListDataset(test_list,
                                           functools.partial(cloud_loader, dataset=dataset, df_gt=df_gt, train=False, args=args))
        train_set = tnt.dataset.ListDataset(train_list,
                                            functools.partial(cloud_loader, dataset=dataset, df_gt=df_gt, train=True, args=args))

        trained_model, loss_train, loss_train_abs, loss_train_log, loss_train_adm, loss_test, loss_test_abs, loss_test_log, loss_test_abs_gl, loss_test_abs_ml, loss_test_abs_hl, loss_test_abs_adm = train_full(args)

        # save the trained model
        PATH = stats_path + "model_ss_" + str(args.subsample_size) + "_dp_" + str(args.diam_pix) + "_fold_" + str(fold_id) + ".pt"
        torch.save(trained_model, PATH)

        # Save all loss stats
        print_stats(stats_file,
                    "Fold_" + str(fold_id) + " Train Loss " + str(loss_train) + " Loss abs " + str(loss_train_abs) + " Loss log " + str(loss_train_log),
                    print_to_console=True)
        if args.adm:
            print_stats(stats_file,
                        "Fold_" + str(fold_id) + " Test Loss " + str(loss_test) + " Loss abs " + str(
                            loss_test_abs) + " Loss log " + str(loss_test_log) + " Loss abs adm " + str(loss_test_abs_adm),
                        print_to_console=True)
        else:
            print_stats(stats_file,
                        "Fold_" + str(fold_id) + " Test Loss " + str(loss_test) + " Loss abs " + str(loss_test_abs) + " Loss log " + str(loss_test_log),
                        print_to_console=True)

        if args.nb_stratum == 2:
            print_stats(stats_file, "Fold_" + str(fold_id) + " Test Loss abs GL " + str(loss_test_abs_gl) + " Test Loss abs ML " + str(
                loss_test_abs_ml), print_to_console=True)
        else:
            print_stats(stats_file, "Fold_" + str(fold_id) + " Test Loss abs GL " + str(loss_test_abs_gl) + " Test Loss abs ML " + str(
                loss_test_abs_ml) + " Test Loss abs HL " + str(
                loss_test_abs_hl), print_to_console=True)

        loss_train_list.append(loss_train)
        loss_train_abs_list.append(loss_train_abs)
        loss_train_log_list.append(loss_train_log)
        loss_train_adm_list.append(loss_train_adm)
        
        loss_test_list.append(loss_test)
        loss_test_abs_list.append(loss_test_abs)
        loss_test_log_list.append(loss_test_log)
        loss_test_abs_gl_list.append(loss_test_abs_gl)
        loss_test_abs_ml_list.append(loss_test_abs_ml)
        loss_test_abs_hl_list.append(loss_test_abs_hl)
        loss_test_adm_list.append(loss_test_abs_adm)


        print_stats(stats_file,
                    "training time " + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))),
                    print_to_console=True)
        fold_id += 1

    #compute mean stats for 5 folds

    if args.adm:
        mean_cross_fold_train = np.mean(loss_train_list), np.mean(loss_train_abs_list), np.mean(loss_train_log_list), np.mean(loss_train_adm_list)
        print_stats(stats_file,
                    "Mean Train Loss " + str(mean_cross_fold_train[0]) + " Loss abs " + str(
                        mean_cross_fold_train[1]) + " Loss log " + str(mean_cross_fold_train[2]) + " Loss ADM " + str(mean_cross_fold_train[3]),
                    print_to_console=True)

        if args.nb_stratum == 2:
            mean_cross_fold_test = np.mean(loss_test_list), np.mean(loss_test_abs_list), np.mean(
                loss_test_log_list), np.mean(loss_test_abs_gl_list), np.mean(loss_test_adm_list)

            print_stats(stats_file,
                        "Mean Test Loss " + str(mean_cross_fold_test[0]) + " Loss abs " + str(
                            mean_cross_fold_test[1]) + " Loss log " + str(
                            mean_cross_fold_test[2]) + " Loss abs GL " + str(
                            mean_cross_fold_test[3]) + " Loss abs ML " + str(
                            mean_cross_fold_test[4]) + " Loss ADM " + str(mean_cross_fold_test[5]),
                        print_to_console=True)

        else:   # 3 stratum
            mean_cross_fold_test = np.mean(loss_test_list), np.mean(loss_test_abs_list), np.mean(
                loss_test_log_list), np.mean(loss_test_abs_gl_list), np.mean(loss_test_abs_ml_list), np.mean(loss_test_abs_hl_list), np.mean(loss_test_adm_list)

            print_stats(stats_file,
                        "Mean Test Loss " + str(mean_cross_fold_test[0]) + " Loss abs " + str(
                            mean_cross_fold_test[1]) + " Loss log " + str(
                            mean_cross_fold_test[2]) + " Loss abs GL " + str(
                            mean_cross_fold_test[3]) + " Loss abs ML " + str(
                            mean_cross_fold_test[4]) + " Loss abs HL " + str(
                            mean_cross_fold_test[5]) + " Loss ADM " + str(mean_cross_fold_test[6]),
                        print_to_console=True)


    else:
        mean_cross_fold_train = np.mean(loss_train_list), np.mean(loss_train_abs_list), np.mean(loss_train_log_list)
        print_stats(stats_file,
                    "Mean Train Loss " + str(mean_cross_fold_train[0]) + " Loss abs " + str(
                        mean_cross_fold_train[1]) + " Loss log " + str(mean_cross_fold_train[2]),
                    print_to_console=True)

        if args.nb_stratum == 2:
            mean_cross_fold_test = np.mean(loss_test_list), np.mean(loss_test_abs_list), np.mean(
                loss_test_log_list), np.mean(loss_test_abs_gl_list), np.mean(loss_test_abs_ml_list)

            print_stats(stats_file,
                        "Mean Test Loss " + str(mean_cross_fold_test[0]) + " Loss abs " + str(
                            mean_cross_fold_test[1]) + " Loss log " + str(
                            mean_cross_fold_test[2]) + " Loss abs GL " + str(
                            mean_cross_fold_test[3]) + " Loss abs ML " + str(
                            mean_cross_fold_test[4]),
                        print_to_console=True)

        else:  # 3 stratum
            mean_cross_fold_test = np.mean(loss_test_list), np.mean(loss_test_abs_list), np.mean(
                loss_test_log_list), np.mean(loss_test_abs_gl_list), np.mean(loss_test_abs_ml_list), np.mean(
                loss_test_abs_hl_list)

            print_stats(stats_file,
                        "Mean Test Loss " + str(mean_cross_fold_test[0]) + " Loss abs " + str(
                            mean_cross_fold_test[1]) + " Loss log " + str(
                            mean_cross_fold_test[2]) + " Loss abs GL " + str(
                            mean_cross_fold_test[3]) + " Loss abs ML " + str(
                            mean_cross_fold_test[4]) + " Loss abs HL " + str(
                            mean_cross_fold_test[5]),
                        print_to_console=True)


if __name__ == "__main__":
    main()






