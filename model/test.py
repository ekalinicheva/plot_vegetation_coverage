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


def evaluate(
    model,
    PCC,
    test_set,
    params,
    args,
    test_list,
    mean_dataset,
    stats_path,
    stats_file,
    last_epoch=False,
    create_final_images_bool=True,
):
    """Eval on test set and inference if this is the last epoch
    Outputs are: average losses (printed), infered values (csv) , k trained models, stats, and images.
    Everything is saved under /experiments/ folder.
    """

    model.eval()

    loader = torch.utils.data.DataLoader(
        test_set, collate_fn=cloud_collate, batch_size=1, shuffle=False
    )
    loss_meter_abs = tnt.meter.AverageValueMeter()
    loss_meter_log = tnt.meter.AverageValueMeter()
    loss_meter = tnt.meter.AverageValueMeter()
    loss_meter_abs_gl = tnt.meter.AverageValueMeter()
    loss_meter_abs_ml = tnt.meter.AverageValueMeter()
    loss_meter_abs_hl = tnt.meter.AverageValueMeter()
    loss_meter_abs_adm = tnt.meter.AverageValueMeter()

    cloud_info_list = []
    for index_batch, (cloud, gt) in enumerate(loader):
        pl_id = test_list[index_batch]

        if PCC.is_cuda:
            gt = gt.cuda()

        pred_pointwise, pred_pointwise_b = PCC.run(
            model, cloud
        )  # compute the prediction
        end_encoding_time = time.time()

        pred_pl, pred_adm, pred_pixels = project_to_2d(
            pred_pointwise, cloud, pred_pointwise_b, PCC, args
        )  # compute plot prediction

        # we compute two losses (negative loglikelihood and the absolute error loss for 2 stratum)
        loss_abs = loss_absolute(pred_pl, gt, args)  # absolut loss
        loss_log, likelihood = loss_loglikelihood(
            pred_pointwise, cloud, params, PCC, args
        )  # negative loglikelihood loss

        if args.ent:
            loss_e = loss_entropy(pred_pixels)

        if args.adm:
            # we compute admissibility loss
            loss_adm = loss_abs_adm(pred_adm, gt)
            if args.ent:
                # Losses : coverage, log-likelihood, admissibility, entropy
                loss = loss_abs + args.m * loss_log + 0.5 * loss_adm + args.e * loss_e
            else:
                # Losses: coverage, log-likelihood, and admissibility losses
                loss = loss_abs + args.m * loss_log + 0.5 * loss_adm
            loss_meter_abs_adm.add(loss_adm.item())
        else:
            if args.ent:
                # losses: coverage, loss-likelihood, entropy
                loss = loss_abs + args.m * loss_log + args.e * loss_e
            else:
                # losses: coverage, loss-likelihood
                loss = loss_abs + args.m * loss_log

        loss_meter.add(loss.item())
        loss_meter_abs.add(loss_abs.item())
        loss_meter_log.add(loss_log.item())
        gc.collect()

        # This is where we get results
        if last_epoch:

            # give separate losses for each stratum
            component_losses = loss_absolute(
                pred_pl, gt, args, level_loss=True
            )  # gl_mv_loss gives separated losses for each stratum
            if args.nb_stratum == 2:
                loss_abs_gl, loss_abs_ml = component_losses
            else:
                loss_abs_gl, loss_abs_ml, loss_abs_hl = component_losses
                loss_abs_hl = loss_abs_hl[~torch.isnan(loss_abs_hl)]
                if loss_abs_hl.size(0) > 0:
                    loss_meter_abs_hl.add(loss_abs_hl.item())
            loss_meter_abs_gl.add(loss_abs_gl.item())
            loss_meter_abs_ml.add(loss_abs_ml.item())

            # create final plot to visualize results
            plot_path = os.path.join(stats_path, "img/")
            create_dir(plot_path)

            if create_final_images_bool:
                create_final_images(
                    pred_pl,
                    gt,
                    pred_pointwise_b,
                    cloud,
                    likelihood,
                    pl_id,
                    mean_dataset,
                    plot_path,
                    stats_file,
                    args,
                    adm=pred_adm,
                )  # create final images with stratum values

            # Keep and format prediction from pred_pl
            with torch.no_grad():
                pred_pl_cpu = pred_pl.cpu().numpy()[0]
                gt_cpu = gt.cpu().numpy()[0]
            cloud_info = {
                "pl_id": pl_id,
                "pred_veg_b": pred_pl_cpu[0],
                "pred_sol_nu": pred_pl_cpu[1],
                "pred_veg_moy": pred_pl_cpu[2],
                "pred_veg_h": pred_pl_cpu[3],
                "vt_veg_b": gt_cpu[0],
                "vt_sol_nu": gt_cpu[1],
                "vt_veg_moy": gt_cpu[2],
                "vt_veg_h": gt_cpu[3],
            }

            cloud_info_list.append(cloud_info)

    return (
        [
            loss_meter.value()[0],
            loss_meter_abs.value()[0],
            loss_meter_log.value()[0],
            loss_meter_abs_gl.value()[0],
            loss_meter_abs_ml.value()[0],
            loss_meter_abs_hl.value()[0],
            loss_meter_abs_adm.value()[0],
        ],
        cloud_info_list,
    )
