## Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

## Build extensions
sh scripts/build_extensions.sh

## Install clean versions of torch and torchvision
python3 -m pip uninstall -y torch torchvision
python3 -m pip freeze | grep nvidia- | xargs pip uninstall -y
python3 -m pip install torch torchvision


## Download data
sh scripts/download_data.sh

## Fix CUDNN
sh scripts/fix_cudnn.sh

## Remove unnecessary files
rm -r occtopology_extension.egg-info
rm -r distance_extension.egg-info
rm -r curvature_extension.egg-info
rm -r utils_extension.egg-info
rm -r build

## Create output directories
mkdir outputs
mkdir outputs/models
mkdir outputs/figures
mkdir outputs/meshes
mkdir outputs/results
mkdir outputs/points