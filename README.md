## Install dependencies

1. On NVIDIA RTX 3060

Refer to the ufficial PyTorch documentation to install the correct version of CUDA: https://pytorch.org/get-started/locally/. Here we used an Anaconda environment (https://www.anaconda.com/). 

- conda create -n myenv python=3.12.2
- conda activate myenv
- conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
- conda install -c pytorch torchtext
- conda install numpy cffi
- pip install pysoundfile
- pip install wandb