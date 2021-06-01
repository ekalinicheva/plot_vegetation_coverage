import numpy as np
from utils.useful_functions import print_stats


# We compute all possible mean stats per loss for all folds
def stats_for_all_folds(
    all_folds_loss_train_lists, all_folds_loss_test_lists, stats_file, args
):
    (
        loss_train_list,
        loss_train_abs_list,
        loss_train_log_list,
        loss_train_adm_list,
    ) = all_folds_loss_train_lists
    (
        loss_test_list,
        loss_test_abs_list,
        loss_test_log_list,
        loss_test_abs_gl_list,
        loss_test_abs_ml_list,
        loss_test_abs_hl_list,
        loss_test_adm_list,
    ) = all_folds_loss_test_lists

    if args.adm:
        mean_cross_fold_train = (
            np.mean(loss_train_list),
            np.mean(loss_train_abs_list),
            np.mean(loss_train_log_list),
            np.mean(loss_train_adm_list),
        )
        print_stats(
            stats_file,
            "Mean Train Loss "
            + str(mean_cross_fold_train[0])
            + " Loss abs "
            + str(mean_cross_fold_train[1])
            + " Loss log "
            + str(mean_cross_fold_train[2])
            + " Loss ADM "
            + str(mean_cross_fold_train[3]),
            print_to_console=True,
        )

        if args.nb_stratum == 2:
            mean_cross_fold_test = (
                np.mean(loss_test_list),
                np.mean(loss_test_abs_list),
                np.mean(loss_test_log_list),
                np.mean(loss_test_abs_gl_list),
                np.mean(loss_test_abs_ml_list),
                np.mean(loss_test_adm_list),
            )

            print_stats(
                stats_file,
                "Mean Test Loss "
                + str(mean_cross_fold_test[0])
                + " Loss abs "
                + str(mean_cross_fold_test[1])
                + " Loss log "
                + str(mean_cross_fold_test[2])
                + " Loss abs GL "
                + str(mean_cross_fold_test[3])
                + " Loss abs ML "
                + str(mean_cross_fold_test[4])
                + " Loss ADM "
                + str(mean_cross_fold_test[5]),
                print_to_console=True,
            )

        else:  # 3 stratum
            mean_cross_fold_test = (
                np.mean(loss_test_list),
                np.mean(loss_test_abs_list),
                np.mean(loss_test_log_list),
                np.mean(loss_test_abs_gl_list),
                np.mean(loss_test_abs_ml_list),
                np.mean(loss_test_abs_hl_list),
                np.mean(loss_test_adm_list),
            )

            print_stats(
                stats_file,
                "Mean Test Loss "
                + str(mean_cross_fold_test[0])
                + " Loss abs "
                + str(mean_cross_fold_test[1])
                + " Loss log "
                + str(mean_cross_fold_test[2])
                + " Loss abs GL "
                + str(mean_cross_fold_test[3])
                + " Loss abs ML "
                + str(mean_cross_fold_test[4])
                + " Loss abs HL "
                + str(mean_cross_fold_test[5])
                + " Loss ADM "
                + str(mean_cross_fold_test[6]),
                print_to_console=True,
            )

    else:
        mean_cross_fold_train = (
            np.mean(loss_train_list),
            np.mean(loss_train_abs_list),
            np.mean(loss_train_log_list),
        )
        print_stats(
            stats_file,
            "Mean Train Loss "
            + str(np.round(mean_cross_fold_train[0], 4))
            + " Loss abs "
            + str(np.round(mean_cross_fold_train[1], 4))
            + " Loss log "
            + str(np.round(mean_cross_fold_train[2], 4)),
            print_to_console=True,
        )

        if args.nb_stratum == 2:
            mean_cross_fold_test = (
                np.mean(loss_test_list),
                np.mean(loss_test_abs_list),
                np.mean(loss_test_log_list),
                np.mean(loss_test_abs_gl_list),
                np.mean(loss_test_abs_ml_list),
            )

            print_stats(
                stats_file,
                "Mean Test Loss "
                + str(mean_cross_fold_test[0])
                + " Loss abs "
                + str(mean_cross_fold_test[1])
                + " Loss log "
                + str(mean_cross_fold_test[2])
                + " Loss abs GL "
                + str(mean_cross_fold_test[3])
                + " Loss abs ML "
                + str(mean_cross_fold_test[4]),
                print_to_console=True,
            )

        else:  # 3 stratum
            mean_cross_fold_test = (
                np.round(np.mean(loss_test_list), 4),
                np.round(np.mean(loss_test_abs_list), 4),
                np.round(np.mean(loss_test_log_list), 4),
                np.round(np.mean(loss_test_abs_gl_list), 4),
                np.round(np.mean(loss_test_abs_ml_list), 4),
                np.round(np.mean(loss_test_abs_hl_list), 4),
            )

            print_stats(
                stats_file,
                "Mean Test Loss "
                + str(mean_cross_fold_test[0])
                + " Loss abs "
                + str(mean_cross_fold_test[1])
                + " Loss log "
                + str(mean_cross_fold_test[2])
                + " Loss abs GL "
                + str(mean_cross_fold_test[3])
                + " Loss abs ML "
                + str(mean_cross_fold_test[4])
                + " Loss abs HL "
                + str(mean_cross_fold_test[5]),
                print_to_console=True,
            )


# We compute all possible loss stats per fold
def stats_per_fold(
    all_folds_loss_train_lists,
    all_folds_loss_test_lists,
    final_train_losses_list,
    final_test_losses_list,
    stats_file,
    fold_id,
    args,
):
    if all_folds_loss_test_lists is None and all_folds_loss_test_lists is None:
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
    else:
        (
            loss_train_list,
            loss_train_abs_list,
            loss_train_log_list,
            loss_train_adm_list,
        ) = all_folds_loss_train_lists
        (
            loss_test_list,
            loss_test_abs_list,
            loss_test_log_list,
            loss_test_abs_gl_list,
            loss_test_abs_ml_list,
            loss_test_abs_hl_list,
            loss_test_adm_list,
        ) = all_folds_loss_test_lists

    loss_train, loss_train_abs, loss_train_log, loss_train_adm = final_train_losses_list
    (
        loss_test,
        loss_test_abs,
        loss_test_log,
        loss_test_abs_gl,
        loss_test_abs_ml,
        loss_test_abs_hl,
        loss_test_adm,
    ) = final_test_losses_list

    # Save all loss stats
    print_stats(
        stats_file,
        "Fold_"
        + str(fold_id)
        + " Train Loss "
        + str(loss_train)
        + " Loss abs "
        + str(loss_train_abs)
        + " Loss log "
        + str(loss_train_log),
        print_to_console=True,
    )
    if args.adm:
        print_stats(
            stats_file,
            "Fold_"
            + str(fold_id)
            + " Test Loss "
            + str(loss_test)
            + " Loss abs "
            + str(loss_test_abs)
            + " Loss log "
            + str(loss_test_log)
            + " Loss abs adm "
            + str(loss_test_adm),
            print_to_console=True,
        )
    else:
        print_stats(
            stats_file,
            "Fold_"
            + str(fold_id)
            + " Test Loss "
            + str(loss_test)
            + " Loss abs "
            + str(loss_test_abs)
            + " Loss log "
            + str(loss_test_log),
            print_to_console=True,
        )

    if args.nb_stratum == 2:
        print_stats(
            stats_file,
            "Fold_"
            + str(fold_id)
            + " Test Loss abs GL "
            + str(loss_test_abs_gl)
            + " Test Loss abs ML "
            + str(loss_test_abs_ml),
            print_to_console=True,
        )
    else:
        print_stats(
            stats_file,
            "Fold_"
            + str(fold_id)
            + " Test Loss abs GL "
            + str(loss_test_abs_gl)
            + " Test Loss abs ML "
            + str(loss_test_abs_ml)
            + " Test Loss abs HL "
            + str(loss_test_abs_hl),
            print_to_console=True,
        )

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
    loss_test_adm_list.append(loss_test_adm)

    all_folds_loss_train_lists = [
        loss_train_list,
        loss_train_abs_list,
        loss_train_log_list,
        loss_train_adm_list,
    ]
    all_folds_loss_test_lists = [
        loss_test_list,
        loss_test_abs_list,
        loss_test_log_list,
        loss_test_abs_gl_list,
        loss_test_abs_ml_list,
        loss_test_abs_hl_list,
        loss_test_adm_list,
    ]
    return all_folds_loss_train_lists, all_folds_loss_test_lists


# We perform tensorboard visualisation by writing the stats to the writer
def write_to_writer(writer, args, i_epoch, list_with_losses, train):
    TESTCOLOR = "\033[104m"
    TRAINCOLOR = "\033[100m"
    NORMALCOLOR = "\033[0m"

    if train:
        loss_train, loss_train_abs, loss_train_log, loss_train_adm = list_with_losses
        if args.adm:
            print(
                TRAINCOLOR
                + "Epoch %3d -> Train Loss: %1.4f Train Loss Abs: %1.4f Train Loss Log: %1.4f Train Loss Adm: %1.4f"
                % (i_epoch, loss_train, loss_train_abs, loss_train_log, loss_train_adm)
                + NORMALCOLOR
            )
            writer.add_scalar("Loss/train/abs_adm", loss_train_adm, i_epoch + 1)
        else:
            print(
                TRAINCOLOR
                + "Epoch %3d -> Train Loss: %1.4f Train Loss Abs: %1.4f Train Loss Log: %1.4f"
                % (i_epoch, loss_train, loss_train_abs, loss_train_log)
                + NORMALCOLOR
            )
        writer.add_scalar("Loss/train_total", loss_train, i_epoch + 1)
        writer.add_scalar("Coverage_MAE/train", loss_train_abs, i_epoch + 1)
        writer.add_scalar("Loss/train_log", loss_train_log, i_epoch + 1)

    else:
        (
            loss_test,
            loss_test_abs,
            loss_test_log,
            _,
            _,
            _,
            loss_test_adm,
        ) = list_with_losses
        if args.adm:
            print(
                TESTCOLOR
                + "Test Loss: %1.4f Test Loss Abs: %1.4f Test Loss Log: %1.4f Test Loss Adm: %1.4f"
                % (loss_test, loss_test_abs, loss_test_log, loss_test_adm)
                + NORMALCOLOR
            )
            writer.add_scalar("Loss/test/abs_adm", loss_test_adm, i_epoch + 1)
        else:
            print(
                TESTCOLOR
                + "Test Loss: %1.4f Test Loss Abs: %1.4f Test Loss Log: %1.4f"
                % (loss_test, loss_test_abs, loss_test_log)
                + NORMALCOLOR
            )
        writer.add_scalar("Loss/test/total", loss_test, i_epoch + 1)
        writer.add_scalar("Coverage_MAE/test", loss_test_abs, i_epoch + 1)
        writer.add_scalar("Loss/test/log", loss_test_log, i_epoch + 1)
    return writer
