import warnings
warnings.simplefilter(action='ignore')

import torchnet as tnt
import gc
import time
from torch.utils.tensorboard import SummaryWriter


# We import from other files
from utils.create_final_images import *
from data_loader.loader import *
from utils.reproject_to_2d_and_predict_plot_coverage import *
from model.loss_functions import *


np.random.seed(42)


def eval(model, PCC, test_set, params, args, test_list, mean_dataset, stats_path, stats_file, last_epoch=False):
    """eval on test set"""

    model.eval()

    loader = torch.utils.data.DataLoader(test_set, collate_fn=cloud_collate, batch_size=1, shuffle=False)
    loss_meter_abs = tnt.meter.AverageValueMeter()
    loss_meter_log = tnt.meter.AverageValueMeter()
    loss_meter = tnt.meter.AverageValueMeter()
    loss_meter_abs_gl = tnt.meter.AverageValueMeter()
    loss_meter_abs_ml = tnt.meter.AverageValueMeter()
    loss_meter_abs_hl = tnt.meter.AverageValueMeter()
    loss_meter_abs_adm = tnt.meter.AverageValueMeter()


    for index_batch, (cloud, gt) in enumerate(loader):

        if PCC.is_cuda:
            gt = gt.cuda()


        start_encoding_time = time.time()
        pred_pointwise, pred_pointwise_b = PCC.run(model, cloud)  # compute the prediction
        end_encoding_time = time.time()
        if last_epoch:            # if it is the last epoch, we get time stats info
            print(end_encoding_time - start_encoding_time)


        pred_pl, pred_adm = project_to_2d(pred_pointwise, cloud, pred_pointwise_b, PCC, args) # compute plot prediction

        # we compute two losses (negative loglikelihood and the absolute error loss for 2 stratum)
        loss_abs = loss_absolute(pred_pl, gt, args)  # absolut loss
        loss_log, likelihood = loss_loglikelihood(pred_pointwise, cloud, params, PCC,
                                                  args)  # negative loglikelihood loss
        if args.adm:
            # we compute admissibility loss
            loss_adm = loss_abs_adm(pred_adm, gt)
            loss = loss_abs + args.m * loss_log + 0.5 * loss_adm
            loss_meter_abs_adm.add(loss_adm.item())
        else:
            loss = loss_abs + args.m * loss_log

        loss_meter.add(loss.item())
        loss_meter_abs.add(loss_abs.item())
        loss_meter_log.add(loss_log.item())
        gc.collect()

        if last_epoch:

            component_losses = loss_absolute(pred_pl, gt, args, level_loss=True) # gl_mv_loss gives separated losses for each stratum
            if args.nb_stratum == 2:
                loss_abs_gl, loss_abs_ml = component_losses
            else:
                loss_abs_gl, loss_abs_ml, loss_abs_hl = component_losses
                loss_abs_hl = loss_abs_hl[~torch.isnan(loss_abs_hl)]
                if loss_abs_hl.size(0) > 0:
                    loss_meter_abs_hl.add(loss_abs_hl.item())
            loss_meter_abs_gl.add(loss_abs_gl.item())
            loss_meter_abs_ml.add(loss_abs_ml.item())

            create_final_images(pred_pl, gt, pred_pointwise_b, cloud, likelihood, test_list[index_batch], mean_dataset, stats_path, stats_file,
                                args, adm=pred_adm)  # create final images with stratum values


    return loss_meter.value()[0], loss_meter_abs.value()[0], loss_meter_log.value()[0], loss_meter_abs_gl.value()[0], loss_meter_abs_ml.value()[0], loss_meter_abs_hl.value()[0], loss_meter_abs_adm.value()[0]
