# load conda into environment
eval "$(conda shell.bash hook)"

# set CUDA_HOME (to PyTorch cuda version)
export CUDA_HOME=/usr/local/cuda-11.8

# make directories for apex
mkdir -p ~/lib && cd ~/lib
git clone https://github.com/NVIDIA/apex
cd apex

# install apex
conda activate cpc && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
