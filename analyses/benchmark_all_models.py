import time
import pandas as pd
from argparse import ArgumentParser
import os
import sys
import glob

# repo_absolute_path = "/home/CGaydon/Documents/LIDAR PAC/plot_vegetation_coverage/"  # for inline debug only TODO: remove

repo_absolute_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_absolute_path)
from model.accuracy import calculate_performance_indicators
from utils.useful_functions import create_dir


parser = ArgumentParser(description="describe_perf")
parser.add_argument(
    "--results_folder_path",
    default=os.path.join(
        repo_absolute_path,
        "experiments/**/*placettes*.csv",
    ),
    type=str,
    help="Path (abs or rel) to the folder containing csv file with results",
)
parser.add_argument(
    "--benchmark_file_path",
    default=os.path.join(
        repo_absolute_path,
        f"experiments/benchmarks/models_benchmark_at_{time.strftime('%Y-%m-%d_%Hh%Mm%Ss')}.csv",
    ),
    type=str,
    help="Path (abs or rel) to the folder containing csv file with results",
)
args, _ = parser.parse_known_args()


def format_cols(df):
    dict_PY = {
        "nom": "pl_id",
        "COUV BASSE": "vt_veg_b",
        "COUV INTER": "vt_veg_moy",
        "COUV HAUTE": "vt_veg_h",
        "couverture basse calibree": "pred_veg_b",
        "couverture inter calibree": "pred_veg_moy",
        "Taux de couverture haute lidar": "pred_veg_h",
    }
    df = df.rename(dict_PY, axis=1)
    cols_of_interest = [
        "pl_id",
        "vt_veg_b",
        "vt_veg_moy",
        "vt_veg_h",
        "pred_veg_b",
        "pred_veg_moy",
        "pred_veg_h",
    ]
    # Check columns and select them
    assert all(coln in df for coln in cols_of_interest)
    df = df[cols_of_interest]
    # Convert if necessary
    if df["vt_veg_b"].max() > 1:
        df[["vt_veg_b", "vt_veg_moy", "vt_veg_h"]] = (
            df[["vt_veg_b", "vt_veg_moy", "vt_veg_h"]] / 100
        )
    if df["pred_veg_b"].dtype == object:
        if any(df["pred_veg_b"].str.contains("%")):
            df[["pred_veg_b", "pred_veg_moy", "pred_veg_h"]] = df[
                ["pred_veg_b", "pred_veg_moy", "pred_veg_h"]
            ].applymap(lambda x: float(x.replace("%", "")) / 100)
        else:
            sys.exit("ERROR: UNKNOWN CASE")

    return df


def main():
    results_file_paths = glob.glob(args.results_folder_path, recursive=True)
    results_file_paths = [
        f for f in results_file_paths if ("(copie)" not in f) and ("/DEV/" not in f)
    ]
    print(results_file_paths)
    if len(results_file_paths) == 0:
        sys.exit(f"No result file found via regex {args.results_folder_path}")
    means = []
    for fname in results_file_paths:
        print(fname)
        df = pd.read_csv(fname)
        df = format_cols(df)
        df = calculate_performance_indicators(df)
        means.append(df.mean())
    df_out = pd.DataFrame(
        means,
        index=[
            f.replace(repo_absolute_path, "").replace(".csv", "")
            for f in results_file_paths
        ],
    )
    create_dir(os.path.dirname(args.benchmark_file_path))
    df_out.to_csv(args.benchmark_file_path, index=True)


if __name__ == "__main__":
    main()
