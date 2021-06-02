# We import from other files
from utils.create_final_images import *
from data_loader.loader import *
from utils.reproject_to_2d_and_predict_plot_coverage import *
from model.loss_functions import *
from utils.useful_functions import create_dir
import torchnet as tnt
import time
import gc
import os

np.random.seed(42)


def infer_from_single_cloud(model, PCC, cloud, args):
    """
    Returns: prediction at the pixel level
    """
    model.eval()
    # loader = torch.utils.data.DataLoader(
    #     test_set, collate_fn=cloud_collate, batch_size=1, shuffle=False
    # )
    if PCC.is_cuda:
        gt = gt.cuda()

    pred_pointwise, pred_pointwise_b = PCC.run(model, cloud)  # compute the prediction

    _, _, pred_pixels = project_to_2d(
        pred_pointwise, cloud, pred_pointwise_b, PCC, args
    )  # compute plot prediction

    print(pred_pixels)

    return pred_pixels
