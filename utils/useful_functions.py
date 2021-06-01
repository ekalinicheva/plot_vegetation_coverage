import os
import time


# Print stats to file
def print_stats(stats_file, text, print_to_console=True):
    with open(stats_file, "a") as f:
        if isinstance(text, list):
            for t in text:
                f.write(t + "\n")
                if print_to_console:
                    print(t)
        else:
            f.write(text + "\n")
            if print_to_console:
                print(text)
    f.close()


# Function to create a new folder if does not exists
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_new_experiment_folder(args):

    # We write results to different folders depending on the chosen parameters
    results_path = os.path.join(
        args.path, f"experiments/RESULTS_{2 if args.nb_stratum == 2 else 3}_strata/"
    )

    if args.adm:
        results_path = os.path.join(results_path, f"admissibility/{args.mode}/")
    else:
        results_path = os.path.join(results_path, f"only_stratum/{args.mode}/")

    # We keep track of time and stats
    start_time = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(start_time)))

    run_name = str(time.strftime("%Y-%m-%d_%Hh%Mm%Ss"))
    stats_path = os.path.join(results_path, run_name) + "/"
    print("Results folder: ", stats_path)
    stats_file = os.path.join(stats_path, "stats.txt")

    create_dir(stats_path)

    # add to args
    args.results_path = results_path
    args.stats_path = stats_path
    args.stats_file = stats_file
