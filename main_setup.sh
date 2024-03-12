pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

sh scripts/build_extensions.sh

python3 -m pip uninstall -y torch torchvision
python3 -m pip freeze | grep nvidia- | xargs pip uninstall -y
python3 -m pip install torch torchvision

pip install google-cloud-storage

sh scripts/download_data.sh
sh scripts/fix_cudnn.sh

rm -r occtopology_extension.egg-info
rm -r distance_extension.egg-info
rm -r curvature_extension.egg-info
rm -r utils_extension.egg-info

rm -r build

mkdir outputs
mkdir outputs/models
mkdir outputs/figures
mkdir outputs/meshes
mkdir outputs/results