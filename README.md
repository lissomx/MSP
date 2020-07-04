# MSP

This code corresponding to the paper: **Latent Space Factorisation and Manipulation via Matrix Subspace Projection (ICML2020)**.

The main web page is here https://xiao.ac/proj/msp.

This code is based on

```
python 3.7
pytorch (version >= 1.4.0)
torchvision (version >= 0.4.1)
```

### Step 1: Preparing CelebA Dataset

To train and test the model, you should download the CelebA dataset (from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

You only need to put the two file: *img_align_celeba.zip* and *list_attr_celeba.txt* in the folder *./CelebA_Dataset/* .

### Step 2: Training

Please run *train_CelebA.py* to train the model like:

```console
> python3 train_CelebA.py
```

You can use the parameter *-pg* to show the training progress.

```console
> python3 train_CelebA.py -pg
```

The trained model will be saved in *./model_save/* .

### Step 3: Testing

The file *testing_CelebA.py* can be used to generate the example pictures (including the picture used in the ICML paper).
```console
> python3 testing_CelebA.py
```
The generated pictures will be in *./Outputs/* .

## Paper and Citation

This work has been published in ICML2020. [Here is the paper of the near camera-ready version](https://xiao.ac/_data/msp/MSP-icml2020-near-camera-ready.pdf). If you find MSP interesting, please consider citing:

> &nbsp;
> @inproceedings{li2020msp,
  title={Latent Space Factorisation and Manipulation via Matrix Subspace Projection },
  author={Li, Xiao and Lin, Chenghua and Li, Ruizhe and Wang, Chaozheng and Guerin, Frank},
  booktitle={Proceedings of the 37th International Conference on Machine Learning},
  year={2020}
}
> &nbsp;

## Acknowledgement

This work is supported by the award made by the UK Engineering and Physical SciencesResearch Council (Grant number: EP/P011829/1).
