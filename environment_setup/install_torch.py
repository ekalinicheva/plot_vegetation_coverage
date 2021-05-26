import os
os.system("pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html")
os.system("pip install torchnet")

# Install torch_scatter
import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)
print(TORCH)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)
print(CUDA)

os.system(f"pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html")
