import warnings
warnings.simplefilter(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)



import functools
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from laspy.file import File
from torch_scatter import scatter_max, scatter_sum, scatter_mean
from torch.optim.lr_scheduler import StepLR
import os, time, re, datetime
import gc
from torch.utils.tensorboard import SummaryWriter



# We import from other files
from model import PointNet
from point_cloud_classifier import PointCloudClassifier
from useful_functions import *
from view_everything import *
from loader import *
from point_projection import *


print(torch.cuda.is_available())



params = {'phi': 0.6261676951828907, 'a_g': 2.7055041947617378, 'a_v': 2.455886147981429, 'loc_g': -0.01741796054681828, 'loc_v': 0.06129981952874307, 'scale_g': 0.06420226677528383, 'scale_v': 2.2278027968946112}
params = {'phi': 0.6136189904818085, 'a_g': 0.6657787175832519, 'a_v': 2.370970267759735, 'loc_g': -6.272105724597368e-29, 'loc_v': 0.19546896827852403, 'scale_g': 0.12812990193731877, 'scale_v': 2.2355462644645163}

fit_alpha_g, fit_loc_g, fit_beta_g = params["a_g"], params["loc_g"], params["scale_g"]
fit_alpha_v, fit_loc_v, fit_beta_v = params["a_v"], params["loc_v"], params["scale_v"]
phi_g = 1 - params["phi"]
phi_v = params["phi"]



path = "/home/ign.fr/ekalinicheva/DATASET_regression/"
# path = "/home/ekaterina/DATASET_regression/"


gt_file = "resultats_placettes_recherche1.csv"
las_folder = path + "placettes/"
len_list = []

# We open las files and create a training dataset
df_gt = pd.read_csv(path + gt_file, sep=',', header=0)
dataset = {}
las_files = os.listdir(las_folder)
all_points = np.empty((0, 9))
for las_file in las_files:
    if las_file:
        las = File(las_folder + las_file, mode='r')
        # print(las_file)
        las = File(las_folder + las_file, mode='r')
        x_las = las.X
        y_las = las.Y
        z_las = las.Z
        r = las.Red
        g = las.Green
        b = las.Blue
        nir = las.nir
        len_list.append((len(nir)))
        intensity = las.intensity
        # nbr_returns = las.num_returns
        nbr_returns = las.return_num
        # cl = laz.classification
        points_placette = np.asarray([x_las / 100, y_las / 100, z_las / 100, r, g, b, nir, intensity, nbr_returns]).T
        #There is a file with 2 points 60m above others (maybe birds), we delete these points
        if las_file=="Releve_Lidar_F70.las":
            points_placette = points_placette[points_placette[:, 2]<640]
        # We do the same for the intensity
        if las_file == "Releve_Lidar_F39.las":
            points_placette = points_placette[points_placette[:, -2] < 20000]

        # We directly substract z_min at local level
        xyz = points_placette[:, :3]
        knn = NearestNeighbors(500, algorithm='kd_tree').fit(xyz[:, :2])
        _, neigh = knn.radius_neighbors(xyz[:, :2], 0.5)
        z = xyz[:,2]
        zmin_neigh = []
        for n in range(len(z)):
            zmin_neigh.append(np.min(z[neigh[n]]))

        points_placette[:,2] = points_placette[:,2]-zmin_neigh
        all_points = np.append(all_points, points_placette, axis=0)
        dataset[os.path.splitext(las_file)[0]] = points_placette

# We extract the names of the plots to create train and test list
placettes = df_gt['Name'].to_numpy()

test_list = placettes


# generate the train and test dataset
test_set = tnt.dataset.ListDataset(test_list,
                                   functools.partial(cloud_loader, dataset=dataset, df_gt=df_gt, train=False))

def encode(model, PCC, args, image=True):
    """eval on test set"""

    model.eval()

    loader = torch.utils.data.DataLoader(test_set, collate_fn=cloud_collate, batch_size=1, shuffle=False)
    # loader = tqdm(loader, ncols=1000)


    for index_batch, (cloud, gt) in enumerate(loader):

        if PCC.is_cuda:
            gt = gt.cuda()

        pred_pointwise, pred_pointwise_b = PCC.run(model, cloud)  # compute the prediction

        """
        We do all the computation to obtain 
        pred_pl - [Bx4] prediction vector for the plot
        scores -  [(BxN)x2] probas that a point belongs to stratum 1 or stratum 2
        scores_list - [BxNx2] same as scores, but separated by batch
        """
        pred_pl, scores, scores_list = project_to_2d(pred_pointwise, cloud, pred_pointwise_b, PCC, args, params)


        # If it is the last epoch, we create two aggregated images_soil_coverage
        if image:
            for b in range(len(pred_pointwise_b)):
                pred_stats = test_list[index_batch] + ' Pred ' + np.array2string(np.round(np.asarray(pred_pl[b].cpu().detach().numpy().reshape(-1)), 2)) + ' GT ' + np.array2string(gt.cpu().numpy()[0])
                pred_cloud = pred_pointwise_b[b]
                current_cloud = cloud[b]
                xy = current_cloud[:2]
                xy = torch.floor((xy - torch.min(xy, dim=1).values.view(2, 1).expand_as(xy)) / (
                        torch.max(xy, dim=1).values - torch.min(xy, dim=1).values + 0.0001).view(2, 1).expand_as(
                    xy) * args.diam_pix).int()
                xy = xy.cpu().numpy()
                unique, index, inverse = np.unique(xy.T, axis=0, return_index=True, return_inverse=True)
                image_soil = np.full((args.diam_pix, args.diam_pix, 2), np.nan)
                image_med_veg = np.full((args.diam_pix, args.diam_pix, 2), np.nan)
                for i in np.unique(inverse):
                    where = np.where(inverse == i)[0]
                    k, m = xy.T[where][0]
                    maxpool = nn.MaxPool1d(len(where))
                    max_pool_val = maxpool(pred_cloud[:, where].unsqueeze(0)).cpu().detach().numpy().flatten()
                    proba_soil = max_pool_val[:2]/(max_pool_val[:2].sum())
                    proba_med_veg = np.asarray([max_pool_val[2], 1-max_pool_val[2]])
                    image_soil[m, k, :] = proba_soil
                    image_med_veg[m, k, :] = proba_med_veg
                image_soil = np.flip(image_soil, axis=0)
                image_med_veg = np.flip(image_med_veg, axis=0)
                # images_soil_coverage[test_list[index_batch]] = image_soil
                # images_med_veg_coverage[test_list[index_batch]] = image_med_veg
                text = ' Pred ' + np.array2string(np.round(np.asarray(pred_pl[b].cpu().detach().numpy().reshape(-1)), 2)) + ' GT ' + np.array2string(gt.cpu().numpy()[0])
                # text = 'Pred ' + ML
                # visualize(image_soil, image_med_veg, current_cloud, pred_cloud, test_list[index_batch], stats_path, text, scores=scores_list[b])
                visualize(image_soil, image_med_veg, current_cloud, pred_cloud, test_list[index_batch], stats_path, text)




def encode_full(args):
    """The full training loop"""

    model = torch.load("/home/ign.fr/ekalinicheva/DATASET_regression/RESULTS/2021-04-14_164807/model_ss_4096_dp_32.pt")

    PCC = PointCloudClassifier(args)

    encode(model, PCC, args, image=True)



torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='model')
args = parser.parse_args()
args.n_epoch = 0
args.n_epoch_test = 0
args.batch_size = 20
args.n_class = 4
args.input_feats = 'xyzrgbnir'
# args.input_feats = 'xyzrgbni'

args.n_input_feats = len(args.input_feats)
args.MLP_1 = [32, 32]
# args.MLP_1 = [32, 64]
# args.MLP_2 = [32, 64, 256]
args.MLP_2 = [64, 128]
# args.MLP_3 = [128, 64, 32]
args.MLP_3 = [64, 32]
args.subsample_size = 2048 * 2
args.cuda = 1
args.lr = 5e-4
args.wd = 0.001
args.diam_pix = 32
args.drop = 0.4
args.soft = True
args.m = 1    #loss regularization


# We keep track of everything
start_time = time.time()
print(time.strftime("%H:%M:%S", time.gmtime(start_time)))
run_name = str(time.strftime("%Y-%m-%d_%H%M%S"))
stats_path = path + "RESULTS/" + run_name + "/"
print(stats_path)
stats_file = stats_path + "stats.txt"
create_dir(stats_path)
print_stats(stats_file, str(args), print_to_console=True)


encode_full(args)

