import warnings
warnings.simplefilter(action='ignore')


import torchnet as tnt
import gc
from torch.utils.tensorboard import SummaryWriter


# We import from other files
from data_loader.loader import *
from utils.reproject_to_2d_and_predict_plot_coverage import *
from model.loss_functions import *

np.random.seed(42)

def train(model, PCC, train_set, params, optimizer, args):
    """train for one epoch"""
    model.train()

    # the loader function will take care of the batching
    loader = torch.utils.data.DataLoader(train_set, collate_fn=cloud_collate, \
                                         batch_size=args.batch_size, shuffle=True, drop_last=True)

    # will keep track of the loss
    loss_meter = tnt.meter.AverageValueMeter()
    loss_meter_abs = tnt.meter.AverageValueMeter()
    loss_meter_log = tnt.meter.AverageValueMeter()
    loss_meter_abs_adm = tnt.meter.AverageValueMeter()

    for index_batch, (cloud, gt) in enumerate(loader):

        if PCC.is_cuda:
            gt = gt.cuda()

        optimizer.zero_grad()  # put gradient to zero
        pred_pointwise, pred_pointwise_b = PCC.run(model, cloud)  # compute the pointwise prediction
        pred_pl, pred_adm, pred_pixels = project_to_2d(pred_pointwise, cloud, pred_pointwise_b, PCC, args)  # compute plot prediction

        # we compute two losses (negative loglikelihood and the absolute error loss for 2 or 3 stratum)
        loss_abs = loss_absolute(pred_pl, gt, args)
        loss_log, likelihood = loss_loglikelihood(pred_pointwise, cloud, params, PCC,
                                                  args)  # negative loglikelihood loss
        if args.ent:
            loss_e = loss_entropy(pred_pixels)

        if args.adm:
            # we compute admissibility loss
            loss_adm = loss_abs_adm(pred_adm, gt)
            if args.ent:
                loss = loss_abs + args.m * loss_log + 0.5 * loss_adm + args.e * loss_e
            else:
                loss = loss_abs + args.m * loss_log + 0.5 * loss_adm
            loss_meter_abs_adm.add(loss_adm.item())
        else:
            if args.ent:
                loss = loss_abs + args.m * loss_log + args.e * loss_e
            else:
                loss = loss_abs + args.m * loss_log

        loss.backward()
        optimizer.step()

        loss_meter_abs.add(loss_abs.item())
        loss_meter_log.add(loss_log.item())
        loss_meter.add(loss.item())
        gc.collect()

    return loss_meter.value()[0], loss_meter_abs.value()[0], loss_meter_log.value()[0], loss_meter_abs_adm.value()[0]