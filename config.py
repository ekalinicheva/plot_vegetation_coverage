from argparse import ArgumentParser
import os

# This script defines all parameters for data loading, model definition, sand I/O operations.

MODE = "DEV"  # DEV or PROD


parser = ArgumentParser(description="model")  # Byte-compiled / optimized / DLL files

# Ignore formating for this block of code
# fmt: off

repo_absolute_path = os.path.dirname(os.path.abspath(__file__))
print(repo_absolute_path)

# System Parameters
parser.add_argument('--path', default=repo_absolute_path, type=str, help="Repo absolute path directory")
parser.add_argument('--dataset_folder_path', default=os.path.join(repo_absolute_path, "/data/placettes_dataset_20210526"), type=str, help="Name of folder with decompressed files. Put in repo root.")

parser.add_argument('--gt_filename', default="./placettes_metadata.csv", type=str, help="Name of ground truth file. Put in LAS folder i.e. in 'dataset'.")
parser.add_argument('--cuda', default=0, type=int, help="Whether we use cuda (1) or not (0)")
parser.add_argument('--folds', default=5, type=int, help="Number of folds for cross validation model training")

# Model Parameters
parser.add_argument('--n_class', default=4, type=int,
                    help="Size of the model output vector. In our case 4 - different vegetation coverage types")
parser.add_argument('--input_feats', default='xyzrgbnir', type=str,
                    help="Point features that we keep. in this code, we keep them all. permuting those letters will break everything. To be modified")
parser.add_argument('--subsample_size', default=4096, type=int, help="Subsample cloud size")
parser.add_argument('--diam_pix', default=32, type=int,
                    help="Size of the output stratum raster (its diameter in pixels)")
parser.add_argument('--m', default=1, type=float,
                    help="Loss regularization. The weight of the negative loglikelihood loss in the total loss")
parser.add_argument('--norm_ground', default=True, type=bool,
                    help="Whether we normalize low vegetation and bare soil values, so LV+BS=1 (True) or we keep unmodified LV value (False) (recommended)")
parser.add_argument('--adm', default=False, type=bool, help="Whether we compute admissibility or not")
parser.add_argument('--nb_stratum', default=3, type=int,
                    help="[2, 3] Number of vegetation stratum that we compute 2 - ground level + medium level; 3 - ground level + medium level + high level")
parser.add_argument('--ECM_ite_max', default=5, type=int, help='Max number of EVM iteration')
parser.add_argument('--NR_ite_max', default=10, type=int, help='Max number of Netwon-Rachson iteration')

# Network Parameters
parser.add_argument('--MLP_1', default=[32, 32], type=list,
                    help="Parameters of the 1st MLP block (output size of each layer). See PointNet article")
parser.add_argument('--MLP_2', default=[64, 128], type=list,
                    help="Parameters of the 2nd MLP block (output size of each layer). See PointNet article")
parser.add_argument('--MLP_3', default=[64, 32], type=list,
                    help="Parameters of the 3rd MLP block (output size of each layer). See PointNet article")
parser.add_argument('--drop', default=0.4, type=float, help="Probability value of the DropOut layer of the model")
parser.add_argument('--soft', default=True, type=bool,
                    help="Whether we use softmax layer for the model output (True) of sigmoid (False)")

# Optimization Parameters
parser.add_argument('--wd', default=0.001, type=float, help="Weight decay for the optimizer")
parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('--step_size', default=50, type=int,
                    help="After this number of steps we decrease learning rate. (Period of learning rate decay)")
parser.add_argument('--lr_decay', default=0.1, type=float,
                    help="We multiply learning rate by this value after certain number of steps (see --step_size). (Multiplicative factor of learning rate decay)")
parser.add_argument('--n_epoch', default=50 if MODE=="PROD" else 2, type=int, help="Number of training epochs")
parser.add_argument('--n_epoch_test', default=5 if MODE=="PROD" else 1, type=int, help="We evaluate every -th epoch")
parser.add_argument('--batch_size', default=20, type=int, help="Size of the training batch")

args, _ = parser.parse_known_args()


assert (args.nb_stratum in [2, 3]), "Number of stratum should be 2 or 3!"
assert (args.lr_decay < 1), "Learning rate decrease should be smaller than 1, as learning rate should decrease"

print(f"Arguments were imported in {MODE} mode")
