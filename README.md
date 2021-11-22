# PLCiL
Implementation of the Pseudo-Labeling for Class incremental Learning.

Link to paper: [[BMVC website]](https://www.bmvc2021-virtualconference.com/conference/papers/paper_1327.html)

![PLCiL](images/PLCiL.png)

# Requirements

* Python >= 3.7
* PyTorch >= 1.9
* Hydra >= 1.1 

Other packages can be found in 

# Datasets

The repertory of each dataset should be specified in a yaml file located in the directory `./config/datasetpath/`.
We provide an example with `./config/datasetpath/workstation.yaml`.

CIFAR100 is downladed automatically to the specified path. 

IMAGENET and Places-365 can be downloaded from the official sources. 

Note that for IMAGENET downsampled to 32x32, you should download the npz version.


# Training
The hyperparameter and output management relies on the Hydra package. All configurations files are in .yaml and stored in `./config/`.
The choice of hyperparameters is detailled in appendix A.1. The class order depends of the seed.

To train on CIFAR-100-full using a WideResnet-28-8 as backbone:
```bash
python train_PLCiL.py dataset=cifar100 dataset.n_lab=500 dataset.incremental_step=10 datasetpath=workstation \
    model=wrn28-8 params.epochs=150 params.memory=2000 params.batch_size=32 params.mu=7 params.lr=0.03 \
    params.lamb=1.0 params.weight_decay=0.0001 params.tau=0.8 params.size_unlab=100000 \
    seed=12345 num_workers=8 gpu=0 save_model=False
```


To train on CIFAR-100-20% using a WideResnet-28-8 as backbone:
```bash
python train_PLCiL.py dataset=cifar100 dataset.n_lab=100 dataset.incremental_step=10 datasetpath=workstation \
    model=wrn28-8 params.epochs=150 params.memory=2000 params.batch_size=32 params.mu=7 params.lr=0.03 \
    params.lamb=1.0 params.weight_decay=0.0001 params.tau=0.8 params.size_unlab=100000 \
    seed=12345 num_workers=8 gpu=0 save_model=False
```

To train on IMAGENET-100-10% using a ResNet-18 as backbone. 
```bash
python train_PLCiL.py dataset=imagenet100 dataset.unlab_scenario=disjoint dataset.n_lab=130 dataset.incremental_step=10 datasetpath=workstation \
    model=resnet18 params.epochs=70 params.memory=2000 params.batch_size=32 params.mu=7 params.lr=0.03 \
    params.lamb=3.0 params.weight_decay=0.0001 params.tau=0.7 params.size_unlab=100000 \
    seed=12345 num_workers=8 gpu=0 save_model=False
```
By default, we use the scenario "unlab_scenario=disjoint" meaning that the unlabeled data is built from IMAGENET-1000 without the classes from IMAGENET-100.
To try different scenarios as showcased in table 3, use "unlab_scenario=disjoint, equal, full or places365".


# Citation

```bibtex
 @INPROCEEDINGS{lechat2021PLCiL,
  author={Alexis Lechat and Stéphane Herbin and Frédéric Jurie},
  title={Pseudo-Labeling for Class Incremental Learning},
  booktitle={Proceedings of the British Machine Vision Conference, {BMVC}, 2021}, 
  year={2021}}
```

# Acknowledgement

This code uses the following implementations :


* CTAugment implementation from [[github]](https://github.com/google-research/fixmatch) 
* Datasets sampler from [[github]](https://github.com/CoinCheung/fixmatch-pytorch)
* Wide-ResNet implementation from [[github]](https://github.com/meliketoy/wide-resnet.pytorch) modified to fit the WRN28-8 proposed by [[Sohn et al. 2020]](https://arxiv.org/abs/2001.07685)

