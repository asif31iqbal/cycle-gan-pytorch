# CycleGAN PyTorch
A PyTorch implementation of [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf) and applying it to font style transfer.

## Setup
I have personally used the [Anaconda](https://www.anaconda.com) distribution for downloading all required packages and creating my virtual environment for this. You can use `pip` for your purposes. For a proper execution of my code, Python version `3.7` is needed.

To mimic my setup steps, do the following:

### Setting up the environment and packages
- Install Python 3.7
- Install Anaconda by following the instructions [here](https://conda.io/docs/user-guide/install/index.html) 
- Create a conda virtual environment `conda create --name {your_env_name}`
- Activate the environment `conda activate {your_env_name}`
- Install following packages:
```
conda install numpy
conda install pandas
conda install scikit-learn
conda install matplotlib
conda install seaborn
conda install pytorch torchvision -c pytorch
conda install jupyter
```

apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos

### Download the data
- Clone this repository and cd to the root directory of the cloned repository
- To download the datasets used in the original paper, execute `download.sh` with parameter which should be one of `apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos`. Note that for my experimentation I have only used `apple2orange` and `summer2winter_yosemite`.
- The data for font style transfer that I used is already included in the report under `datasets/arial2times` and `datasets/arial2times_word`.

### Run training
Once you have set up the virtual env, cloned the reposiotry and downloaded the data, you can now run training:

```
# apple2orange
python train.py --dataset apple2orange --epochs 50 --constant_lr_epochs 25 --lr 0.0004 --cycle_loss_lambda 5

# summer2winter_yosemite
python train.py --dataset summer2winter_yosemite --epochs 50 --constant_lr_epochs 25 --lr 0.0002 --cycle_loss_lambda 10 --identity_loss_lambda 5

# arial2times_word
python train.py --dataset arial2times_word --epochs 50 --constant_lr_epochs 25 --lr 0.0002 --cycle_loss_lambda 10 --identity_loss_lambda 5
```
Please note the full list of parameters in `train.py`

### Run testing

``
python test.py --dataset apple2orange
python test.py --dataset summer2winter_yosemite
python test.py --dataset arial2times_word
```




