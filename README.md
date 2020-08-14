# MSP

This code corresponding to the paper: **Latent Space Factorisation and Manipulation via Matrix Subspace Projection (ICML2020)**.

The main website is here https://xiao.ac/proj/msp.

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

Alternatively, you can download the pre-trained model (from https://s3.eu-west-2.amazonaws.com/nn.models/MSP_CelebA.tch), and put the file *MSP_CelebA.tch* in *./model_save/* .

### Step 3: Testing

The file *testing_CelebA.py* can be used to generate the example pictures (including the picture used in the ICML paper).
```console
> python3 testing_CelebA.py
```
The generated pictures will be in *./Outputs/* .

### Textural Experiments (TBD)

The code for the text experiment is being collated and will be released soon.

## Paper and Citation

This work has been published in ICML2020. [Here is the paper of the near camera-ready version](https://arxiv.org/abs/1907.12385). If you find MSP interesting, please consider citing:

> &nbsp;
> @incollection{icml2020_1832,
 author = {Li, Xiao and Lin, Chenghua and Li, Ruizhe and Wang, Chaozheng and Guerin, Frank},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {3211--3221},
 title = {Latent Space Factorisation and Manipulation via Matrix Subspace Projection},
 year = {2020}
}
> &nbsp;

## Acknowledgement

This work is supported by the award made by the UK Engineering and Physical SciencesResearch Council (Grant number: EP/P011829/1).
