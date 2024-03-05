pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

sh scripts/build_extensions.sh

python3 -m pip uninstall torch torchvision
python3 -m pip freeze | grep nvidia- | xargs pip uninstall -y
python3 -m pip install torch torchvision