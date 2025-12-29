GH-DSTPP

This project presents GH-DSTPP, a GH-spatio-temporal diffusion point process model for modeling and generating spatio-temporal event sequences. The model integrates graph-based structures with diffusion processes to capture complex dependencies in spatio-temporal event data. Contributors to this project are from [Your Organization/Institution Name].
The code is tested under a Linux desktop with PyTorch 1.7+ and Python 3.7+.

Installation
Environment
Tested OS: Linux
Python >= 3.7
PyTorch == 1.7.1 (or compatible versions)
Tensorboard
Dependencies
Install PyTorch 1.7.1 with the appropriate CUDA version for your system.
Install required Python packages using:

pip install -r requirements.txt
Model Training
Use the following command to train GH-DSTPP on the Earthquake dataset:

python app.py --dataset Earthquake --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --total_epochs 2000
To train on other datasets, replace the --dataset parameter:

python app.py --dataset COVID19 --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --total_epochs 2000

python app.py --dataset Citibike --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --total_epochs 2000

python app.py --dataset Independent --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --total_epochs 2000
Trained models are saved in the ModelSave/ directory.
Training logs are saved in the logs/ directory for monitoring with Tensorboard.

Key Components
Encoder Layer (Layers.py): Implements the core encoder with multi-head attention and position-wise feed-forward networks, supporting spatio-temporal context integration.
Diffusion Model (DiffusionModel.py): Handles the diffusion process for spatio-temporal event generation, with separate processing for temporal and spatial components.
Sequence Generation (MHP.py): Incorporates mechanisms for generating event sequences using Ogata's thinning method, adapted for diffusion-based point processes.
Model Architecture (Models.py): Defines the overall model structure, combining encoders, recurrent layers, and diffusion components.

Note

The implementation leverages concepts from denoising diffusion probabilistic models (DDPM) and graph neural networks for enhanced spatio-temporal modeling.
If you use this code in your research, please cite our paper (once published):
plaintext
@inproceedings{yourpaper202X,
  author = {Author 1 and Author 2 and Author 3},
  title = {GH-DSTPP: Graph-enhanced Spatio-temporal Diffusion Point Processes},
  year = {202X},
  booktitle = {Proceedings of the XXXX Conference},
  pages = {XXXX–XXXX},
}
