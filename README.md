# Research project - Deep Marching Cubes

## Setup requirements

Please follow these instructions to install the required packages:

```bash
cd deep_marching_cubes
python -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
sh main_setup.sh
```

To train the model and upload outputs to GCP, please use Tmux:
```bash
sudo apt install tmux
tmux
```
Then press Ctrl+b and d
```bash
source venv/bin/activate
python run.py
```
After training:
```bash
tmux attach
sh scripts/download_outputs.sh
```
Then press Ctrl+b and :
```bash
kill-session
```