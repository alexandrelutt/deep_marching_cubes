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

Then, install the [data](https://s3.eu-central-1.amazonaws.com/avg-projects/deep_marching_cubes_data.zip) and store it in a `all_data` folder.

To train the model and upload outputs to GCP, please use Tmux:
```bash
sudo apt install tmux
tmux
source venv/bin/activate
python run.py
```
You can now leave GCP VM window. After training, reconnect to the VM:
```bash
tmux attach
sh scripts/download_outputs.sh
```
Then press Ctrl+b and :
```bash
kill-session
```
If you want to exit the tmux window, press Ctrl+b and d.
