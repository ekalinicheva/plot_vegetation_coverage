import warnings
warnings.simplefilter(action='ignore')

import functools
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from laspy.file import File
from torch.optim.lr_scheduler import StepLR
import time
import gc
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from model.loss_functions import loss_abs_gl_ml_hl, loss_loglikelihood


# We import from other files
from model.model import PointNet
from utils.point_cloud_classifier import PointCloudClassifier
from utils.useful_functions import *
from utils.create_final_images import *
from data_loader.loader import *
from utils.reproject_to_2d_and_predict_plot_coverage import *
from em_gamma.get_gamma_params import *
from utils.open_las import *



print(torch.cuda.is_available())
np.random.seed(42)


torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='model')
args = parser.parse_args()
args.n_epoch = 200            # number of training epochs
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
args.z_max = 25                 # maximum z value for data normalization
args.adm = True
args.nb_stratum = 3



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
results_path = path + "RESULTS_3_stratum/"
stats_path = results_path + run_name + "/"
print(stats_path)
stats_file = stats_path + "stats.txt"
create_dir(stats_path)
print_stats(stats_file, str(args), print_to_console=True) # save all the args parameters


#   Parameters of gamma distributions
# params = {'phi': 0.6261676951828907, 'a_g': 2.7055041947617378, 'a_v': 2.455886147981429, 'loc_g': -0.01741796054681828,
#           'loc_v': 0.06129981952874307, 'scale_g': 0.06420226677528383, 'scale_v': 2.2278027968946112}

# phi_g = 1 - params["phi"]
# phi_v = params["phi"]


def main():
    # # We open las files and create a training dataset
    all_points, dataset, mean_dataset = open_las(las_folder)

    # #   Parameters of gamma distributions for two stratum
    # params = get_gamma_params(z_all)
    params = {'phi': 0.47535938, 'a_g': 0.1787963, 'a_v': 3.5681678,
              'loc_g': 0, 'loc_v': 0, 'scale_g': 0.48213167,
              'scale_v': 1.46897231}

    # [0.1787963 3.5681678]
    # [0.48213167 1.46897231]
    # [0.52464062 0.47535938]
    # shape, scale, pi



    def train(model, PCC, optimizer, args):
        """train for one epoch"""
        model.train()

        # the loader function will take care of the batching
        loader = torch.utils.data.DataLoader(train_set, collate_fn=cloud_collate, \
                                             batch_size=args.batch_size, shuffle=True, drop_last=True)

        # will keep track of the loss
        loss_meter_abs = tnt.meter.AverageValueMeter()
        loss_meter_log = tnt.meter.AverageValueMeter()
        loss_meter = tnt.meter.AverageValueMeter()
        for index_batch, (cloud, gt) in enumerate(loader):
            if PCC.is_cuda:
                gt = gt.cuda()
            optimizer.zero_grad()  # put gradient to zero
            pred_pointwise, pred_pointwise_b = PCC.run(model, cloud)  # compute the prediction
            pred_pl = project_to_2d(pred_pointwise, cloud, pred_pointwise_b, PCC, args)

            # we compute two losses (negative loglikelihood and the absolute error loss for 3 stratum)


            loss_abs = loss_abs_gl_ml_hl(pred_pl, gt)
            loss_log, likelihood = loss_loglikelihood(pred_pointwise, cloud, params, PCC, args)
            loss = loss_abs + args.m * loss_log
            loss.backward()
            optimizer.step()

            loss_meter_abs.add(loss_abs.item())
            loss_meter_log.add(loss_log.item())
            loss_meter.add(loss.item())
            gc.collect()

        return loss_meter.value()[0], loss_meter_abs.value()[0], loss_meter_log.value()[0]


    def eval(model, PCC, args, last_epoch=True):
        """eval on test set"""

        model.eval()

        loader = torch.utils.data.DataLoader(test_set, collate_fn=cloud_collate, batch_size=1, shuffle=False)
        loss_meter_abs = tnt.meter.AverageValueMeter()
        loss_meter_log = tnt.meter.AverageValueMeter()
        loss_meter = tnt.meter.AverageValueMeter()


        for index_batch, (cloud, gt) in enumerate(loader):
            if PCC.is_cuda:
                gt = gt.cuda()

            # if it is the last epoch, we get time stats info
            if last_epoch:
                start_encoding_time = time.time()
                pred_pointwise, pred_pointwise_b = PCC.run(model, cloud)  # compute the prediction
                pred_pl = project_to_2d(pred_pointwise, cloud, pred_pointwise_b, PCC, args)
                end_encoding_time = time.time()
                print(end_encoding_time - start_encoding_time)
            else:
                pred_pointwise, pred_pointwise_b = PCC.run(model, cloud)  # compute the prediction
                pred_pl = project_to_2d(pred_pointwise, cloud, pred_pointwise_b, PCC, args)

            # we compute two losses (negative loglikelihood and the absolute error loss for 3 stratum)
            loss_abs = loss_abs_gl_ml_hl(pred_pl, gt)
            loss_log, likelihood = loss_loglikelihood(pred_pointwise, cloud, params, PCC, args)
            loss = loss_abs + args.m * loss_log

            loss_meter_abs.add(loss_abs.item())
            loss_meter_log.add(loss_log.item())
            loss_meter.add(loss.item())

            # If it is the last epoch, we create two images with ground and medium stratum. this code is adapted only for eval batch of size 1. To change=)
            if last_epoch:
                create_final_images_stratum3(pred_pl, gt, pred_pointwise_b, cloud, likelihood, test_list[index_batch], mean_dataset, stats_path, stats_file,
                                    args, create_raster=False)   #create final images with stratum values
        return loss_meter.value()[0], loss_meter_abs.value()[0], loss_meter_log.value()[0]


    def train_full(args):
        """The full training loop"""
        # initialize the model
        model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args)
        writer = SummaryWriter(results_path+ "runs/")
        # model = torch.load("/home/ekaterina/DATASET_regression/RESULTS/2021-04-13_164536/model_ss_4096_dp_32.pt")

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

            # train one epoch
            loss_train, loss_train_abs, loss_train_log = train(model, PCC, optimizer, args)
            print(TRAINCOLOR + 'Epoch %3d -> Train Loss: %1.4f Train Loss Abs: %1.4f Train Loss Log: %1.4f' % (i_epoch, loss_train, loss_train_abs, loss_train_log) + NORMALCOLOR)
            scheduler.step()

            if (i_epoch + 1) % args.n_epoch_test == 0:
                if (i_epoch + 1) == args.n_epoch:   # if last epoch, we creare 2D images with points projections
                    loss_test, loss_test_abs, loss_test_log = eval(model, PCC, args, last_epoch=True)
                else:
                    loss_test, loss_test_abs, loss_test_log = eval(model, PCC, args, last_epoch=False)
                gc.collect()
                print(TESTCOLOR + 'Test Loss: %1.4f Test Loss Abs: %1.4f Test Loss Log: %1.4f' % (loss_test, loss_test_abs, loss_test_log) + NORMALCOLOR)
                writer.add_scalar('Loss/test', loss_test, i_epoch + 1)
                writer.add_scalar('Loss/test_abs', loss_test_abs, i_epoch + 1)
                writer.add_scalar('Loss/test_log', loss_test_log, i_epoch + 1)
            writer.add_scalar('Loss/train', loss_train, i_epoch + 1)
            writer.add_scalar('Loss/train_abs', loss_train_abs, i_epoch + 1)
            writer.add_scalar('Loss/train_log', loss_train_log, i_epoch + 1)
        print_stats(stats_file, "Test loss = " + str(loss_train), print_to_console=False)

        writer.flush()
        return model, PCC, loss_train, loss_train_abs, loss_train_log, loss_test, loss_test_abs, loss_test_log


    df_gt = pd.read_csv(path + gt_file, sep=',', header=0)  # we open GT file
    placettes = df_gt['Name'].to_numpy()    # We extract the names of the plots to create train and test list

    # We use 5 folds cross validation
    folds = 5
    kf = KFold(n_splits=folds, random_state=42, shuffle=True)

    # We keep track of stats per fold
    loss_train_list = []
    loss_train_abs_list = []
    loss_train_log_list = []

    loss_test_list = []
    loss_test_abs_list = []
    loss_test_log_list = []


    fold_id = 1
    for train_ind, test_ind in kf.split(placettes):
        train_list = placettes[train_ind]
        test_list = placettes[test_ind]

        # generate the train and test dataset
        test_set = tnt.dataset.ListDataset(test_list,
                                           functools.partial(cloud_loader, dataset=dataset, df_gt=df_gt, train=False, args=args))
        train_set = tnt.dataset.ListDataset(train_list,
                                            functools.partial(cloud_loader, dataset=dataset, df_gt=df_gt, train=True, args=args))

        trained_model, PCC, loss_train, loss_train_abs, loss_train_log, loss_test, loss_test_abs, loss_test_log = train_full(args)

        # save the trained model
        PATH = stats_path + "model_ss_" + str(args.subsample_size) + "_dp_" + str(args.diam_pix) + "_fold_" + str(fold_id) + ".pt"
        torch.save(trained_model, PATH)

        # Save all loss stats
        print_stats(stats_file,
                    "Fold_" + str(fold_id) + " Train Loss " + str(loss_train) + " Loss abs " + str(loss_train_abs) + " Loss log " + str(loss_train_log),
                    print_to_console=True)
        print_stats(stats_file,
                    "Fold_"+ str(fold_id) + " Test Loss " + str(loss_test) + " Loss abs " + str(loss_test_abs) + " Loss log " + str(loss_test_log),
                    print_to_console=True)

        loss_train_list.append(loss_train)
        loss_train_abs_list.append(loss_train_abs)
        loss_train_log_list.append(loss_train_log)
        
        loss_test_list.append(loss_test)
        loss_test_abs_list.append(loss_test_abs)
        loss_test_log_list.append(loss_test_log)


        print_stats(stats_file,
                    "training time " + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))),
                    print_to_console=True)
        fold_id += 1

    #compute mean stats for 5 folds
    mean_cross_fold_train = np.mean(loss_train_list), np.mean(loss_train_abs_list), np.mean(loss_train_log_list)
    mean_cross_fold_test = np.mean(loss_test_list), np.mean(loss_test_abs_list), np.mean(loss_test_log_list)

    print(mean_cross_fold_train)
    print(mean_cross_fold_test)


    print_stats(stats_file,
                "Mean Train Loss " + str(mean_cross_fold_train[0]) + " Loss abs "  + str(mean_cross_fold_train[1]) + " Loss log "  + str(mean_cross_fold_train[2]),
                print_to_console=True)

    print_stats(stats_file,
                "Mean Test Loss " + str(mean_cross_fold_test[0]) + " Loss abs "  + str(mean_cross_fold_test[1]) + " Loss log "  + str(mean_cross_fold_test[2]),
                print_to_console=True)


if __name__ == "__main__":
    main()







