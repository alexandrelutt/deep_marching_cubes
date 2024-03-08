# Research project - Deep Marching Cubes

## Setup requirements

Please follow these instructions to install the required packages:

```bash
python -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
sh main_setup.sh
```

To train the model and upload outputs to GCP, please run
```bash
python run.py
sh scripts/download_outputs.sh
```