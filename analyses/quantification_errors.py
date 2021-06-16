import os, sys

repo_absolute_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_absolute_path)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from utils.useful_functions import create_dir
from model.accuracy import (
    compute_mae,
    compute_mae2,
    compute_accuracy,
    compute_accuracy2,
)  # do not "import *"
from functools import partial


##############################################################
##############################################################

experiment_rel_path = "./experiments/RESULTS_3_strata/only_stratum/PROD/learning/2021-05-31_11h40m21s/PCC_inference_all_placettes.csv"

# TODO: go from % to ratio to avoid inconsistency
# here everything is in % for clarity of plot
bins_centers = np.array([0, 10, 25, 33, 50, 75, 90, 100])
bins_borders = np.append((bins_centers[:-1] + bins_centers[1:]) / 2, 105)
assert list(
    map(lambda x: abs(x[0] - x[1]), zip(bins_borders[:-1], bins_centers[:-1]))
) == list(map(lambda x: abs(x[0] - x[1]), zip(bins_borders, bins_centers[1:])))

# we round up to be coherent with current metrics
bins_borders = np.floor(bins_borders + 0.5).astype(int)

# create the associated mapping from center to borders
bb = [0] + bins_borders.tolist()
center_to_border_dict = {
    center: borders for center, borders in zip(bins_centers, zip(bb[:-1], bb[1:]))
}

# adapt error functions to %
compute_accuracy2_perc = partial(
    compute_accuracy2, margin=10, center_to_border_dict=center_to_border_dict
)
compute_accuracy_perc = partial(
    compute_accuracy, center_to_border_dict=center_to_border_dict
)
compute_mae2_perc = partial(compute_mae2, center_to_border_dict=center_to_border_dict)
compute_mae_perc = partial(compute_mae)  # to be coherent in naming convention here

error_funcs = [
    compute_mae_perc,
    compute_accuracy_perc,
    compute_mae2_perc,
    compute_accuracy2_perc,
]
error_funcs_names = ["mae", "acc", "mae2", "acc2"]

##############################################################
##############################################################


def study_quantification_error_1(df, output_fig_path=""):
    # total error under uniform distribution
    x = np.linspace(0, 100, 2001)
    y_classes = np.digitize(x, bins_borders)
    y_quant = bins_centers[y_classes[:, None]].squeeze()
    error = np.abs(x - y_quant)
    print(f"Quantification error #1 = {error.mean().round(2)}%")

    # Error by class
    errors_by_class = np.zeros(shape=(9,))
    for i in range(9):
        errors_by_class[i] = error[y_classes == i].mean()
    errors_by_class = errors_by_class.round(2)
    errors_by_class_mapper = {
        val_quant: mean_error
        for val_quant, mean_error in zip(bins_centers, errors_by_class)
    }
    # printing!
    l = list(
        zip(
            bins_centers,
            len(bins_centers) * ["->"],
            errors_by_class,
            len(bins_centers) * ["%pts"],
        )
    )
    print(l)

    # actual error in dataset
    df_errors = df.copy()
    df_errors[["vt_veg_b", "vt_veg_moy", "vt_veg_h"]] = df_errors[
        ["vt_veg_b", "vt_veg_moy", "vt_veg_h"]
    ].replace(errors_by_class_mapper)
    print(
        f"Actual error due to quantization: {df_errors[['vt_veg_b','vt_veg_moy','vt_veg_h']].values.mean()}"
    )
    print(df_errors[["vt_veg_b", "vt_veg_moy", "vt_veg_h"]].describe().loc["mean"])

    # plot
    plt.title("L'erreur de quantification dépend de la valeur de couverture")
    plt.plot(x, y_quant, label="Couverture (discretisée, %)")
    plt.plot(x, x, label="Couverture (continue, %)")
    plt.plot(x, error, label="Erreur de quantification (pp)")
    plt.xlabel("Couverture (%)")
    plt.legend()
    plt.tight_layout()

    # plt.show()
    if output_fig_path:
        plt.savefig(output_fig_path, dpi=150, transparent=True)
        print(f"Quantification error #1 plot saved to {output_fig_path}")
    return errors_by_class_mapper


def describe_possible_measurement_error_distribution(
    stdev_of_error_list=[0.0000001, 5, 10, 12.5, 15, 20],
    above_error_list=[2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 50],
    msrt_error_description_path="",
):
    """Output a csv of prob(|e|>value) for different stdev of measuremen errors"""

    errors = np.empty((len(above_error_list), len(stdev_of_error_list)))

    for idx_stdev, stdev_of_error in enumerate(stdev_of_error_list):
        imprecision_generator = norm(0, stdev_of_error)
        for idx_above_error, above_error in enumerate(above_error_list):
            prob = 1 - (
                imprecision_generator.cdf(above_error)
                - imprecision_generator.cdf(-above_error)
            )
            errors[idx_above_error, idx_stdev] = prob

    df_errors = pd.DataFrame(
        data=errors,
        index=[f"|e|>{s}" for s in above_error_list],
        columns=[f"σ={s:.1f}" for s in stdev_of_error_list],
    ).round(2)
    df_errors.to_csv(
        msrt_error_description_path,
    )
    print(f"Measurement error description saved to {msrt_error_description_path}")


def compute_expected_error_based_on_measurement_error_stdev(
    stdev_of_error=10, error_func=compute_mae
):
    """
    Return expected MAE (1 or 2, chosen with error_func(y_pred, y_gt)) based on the assumption of a gaussian measurement error,
    amplified by quantification of ground truth coverage at time of measurement.
    """

    # Gaussian error
    imprecision_generator = norm(0, stdev_of_error)

    expected_E_list = []

    # for each coverage class
    for center, (lower_b, upper_b) in center_to_border_dict.items():
        E_list = []

        # for each value in a coverage class
        for real_coverage in np.arange(
            lower_b, upper_b + 0.1, 0.25
        ):  # we always include both border in this range
            # print(f"Real coverage : {real_coverage}")
            E = 0.0
            W = 0.0

            # for different imprecision values
            for imprecision_delta in np.arange(-50, 50, 0.25):
                w = imprecision_generator.pdf(imprecision_delta)
                coverage_with_imprecision = real_coverage + imprecision_delta
                # we accept that 0% and 100% are more likely errors when close to them
                coverage_with_imprecision = max(coverage_with_imprecision, 0)
                coverage_with_imprecision = min(coverage_with_imprecision, 100)
                if lower_b <= coverage_with_imprecision <= upper_b:
                    # GT within real coverage class
                    E = E + w * error_func(
                        real_coverage,
                        center,
                        center_to_border_dict=center_to_border_dict,
                    )
                else:
                    # GT outside real coverage class
                    closest_center = bins_centers[
                        np.argsort(abs(bins_centers - coverage_with_imprecision))[1]
                    ]
                    E = E + w * error_func(
                        real_coverage,
                        closest_center,
                        center_to_border_dict=center_to_border_dict,
                    )  # quantification error N°2
                W = W + w
            E_list.append(E / W)
        expected_E_in_class = np.mean(E_list)

        print(f"e={expected_E_in_class:.2}% for range [{lower_b};{upper_b}]")
        expected_E_list.append(expected_E_in_class)
    expected_E = np.mean(expected_E_list).round(2)
    print(
        f"Expected indicator ({error_func.func.__name__}) with evaluation error of stdev={stdev_of_error} is {expected_E}"
    )
    return expected_E


def get_all_expected_error_based_on_measurement_error_stdev(
    stdev_of_error_list=[0.0000001, 5, 10, 12.5, 15, 20],
    expected_errors_path="",
):
    global error_funcs
    global error_funcs_names
    expected_Es = np.empty((len(error_funcs), len(stdev_of_error_list)))
    for idx_stdev, stdev_of_error in enumerate(stdev_of_error_list):
        for idx_func, error_func in enumerate(error_funcs):
            expected_E = compute_expected_error_based_on_measurement_error_stdev(
                stdev_of_error=stdev_of_error, error_func=error_func
            )
            expected_Es[idx_func, idx_stdev] = expected_E

    df_errors = pd.DataFrame(
        data=expected_Es,
        index=error_funcs_names,
        columns=[f"σ={s:.1f}" for s in stdev_of_error_list],
    ).round(2)
    df_errors.to_csv(
        expected_errors_path,
    )
    print(f"Expected errors description saved to {expected_errors_path}")


def main():
    # get data
    format_percentage_to_float = lambda x: float(x.replace("%", ""))
    cols_to_format = [
        "pred_veg_b",
        "pred_sol_nu",
        "pred_veg_moy",
        "pred_veg_h",
        "vt_veg_b",
        "vt_sol_nu",
        "vt_veg_moy",
        "vt_veg_h",
        "error_veg_b",
        "error_veg_moy",
        "error_veg_b_and_moy",
    ]
    df = pd.read_csv(
        experiment_rel_path,
        converters={key: format_percentage_to_float for key in cols_to_format},
    )
    # create paths
    analyses_path = "/".join(experiment_rel_path.split("/")[:-1] + ["analyses/"])
    create_dir(analyses_path)
    output_fig_path = os.path.join(analyses_path, "quantification_error_1.png")
    msrt_error_description_path = os.path.join(
        analyses_path, "msrt_error_description.csv"
    )
    expected_errors_path = os.path.join(
        analyses_path, "expected_errors_under_gaussian_msrt_error.csv"
    )
    # run anlayses
    study_quantification_error_1(df, output_fig_path=output_fig_path)
    describe_possible_measurement_error_distribution(
        msrt_error_description_path=msrt_error_description_path
    )

    # get_all_expected_error_based_on_measurement_error_stdev(
    #     expected_errors_path=expected_errors_path
    # )


if __name__ == "__main__":
    main()
