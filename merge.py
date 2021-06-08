# for now merge.py is a separate script, but it should be part of infer at some point

import warnings

warnings.simplefilter(action="ignore")

import numpy as np
import os
import torch
import matplotlib

# Weird behavior: loading twice in cell appears to remove an elsewise occuring error.
for i in range(2):
    try:
        matplotlib.use("TkAgg")  # rerun this cell if an error occurs.
    except:
        print("!")

print(torch.cuda.is_available())
np.random.seed(42)
torch.cuda.empty_cache()

# We import from other files
from config import args
from utils.useful_functions import create_new_experiment_folder, print_stats
