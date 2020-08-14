import argparse
import operator
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import torch
import random
import os
import sys
import scipy
# from scipy import linalg, matrix
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image,ImageFont,ImageDraw

from Dataset_CelebA import CelebA
from M_ModelAE_Cnn import CnnVae as AE
from M_ModelGan_PatchGan import PatchGan as Gan

parser = argparse.ArgumentParser(description='C0AE for CelebA')
parser.add_argument('-bz', '--batch-size', type=int, default=70,
                    help='input batch size for training (default: 128)')
parser.add_argument('-iz', '--image-size', type=int, default=256,
                    help='size to resize for CelebA pics (default: 256)')
parser.add_argument('-nf', type=int, default=64,
                    help='output channel number of the first cnn layer (default: 64)')
args = parser.parse_args()

location = "./"

print(args)

celeba_zip = "CelebA_Dataset/img_align_celeba.zip"
celeba_txt = "CelebA_Dataset/list_attr_celeba.txt"
model_save = 'model_save'
output_dir = 'Outputs'


print("CelebA zip file: ", os.path.abspath(celeba_zip))
print("CelebA txt file: ", os.path.abspath(celeba_txt))
print("model save: ", os.path.abspath(model_save))
print("output dir: ", os.path.abspath(output_dir))


batch_size = args.batch_size
image_size = args.image_size
nf = args.nf
lr = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = CelebA(celeba_zip, celeba_txt, image_size)
label_size = dataset.label_size

# test_data = Subset(dataset, range(182638, 202599))
test_data = Subset(dataset, [184183,198267,182691]) # 184183,198267,182691 are the pictures used in the icml paper
dataloader_test = DataLoader(test_data, batch_size=120)

model = AE(image_size, label_size, nf, nc=3)

def make_continue(cl, total=5):
    # e.g.:
    #   cl = [(15,-1,1),(12,2,-2)]
    #   total = 5
    step = [(i[2]-i[1])/(total-1) for i in cl]
    output = []
    for i in range(total):
        o = [(x[0], x[1]+i*step[j]) for j, x in enumerate(cl)]
        output.append(o)
    # output of e.g.:
    # [
    #   [(15, -1), (12, 2)],
    #   [(15, -0.5), (12, 1)],
    #   [(15, 0), (12, 0)],
    #   [(15, 0.5), (12, -1)],
    #   [(15, 1), (12, -2)],
    # ]
    return output


def randnGenerate(n_pic=40):
    U = model.get_U()
    for i, (_, label) in enumerate(dataloader_test):
        label = label[:n_pic].to(device)*2
        b_size = label.shape[0]
        label_size = label.shape[1]
        hidden_size = 2048 # 2048 corresponds to the defalt model settings
        label_size = label.shape[1]
        noise = torch.rand([b_size,hidden_size-label_size]).to(device)

        z0 = torch.cat([label,noise*0], dim=1)
        z1 = torch.cat([label,noise*0.05], dim=1)
        z2 = torch.cat([label,noise*0.1], dim=1)
        z3 = torch.cat([label,noise*0.15], dim=1)
        prod0 = model.decoder(z0 @ U)
        prod1 = model.decoder(z1 @ U)
        prod2 = model.decoder(z2 @ U)
        prod3 = model.decoder(z3 @ U)
        pics = torch.cat([prod0,prod1,prod2,prod3])
        pics.add_(1.0).div_(2.0)
        vutils.save_image(pics, f'{output_dir}/randnGenerate.jpg', nrow=n_pic, padding=4)
        break

def label_morphing(step=5):
    if not os.path.isdir(f'{output_dir}/Morphing_Labels/'):
        os.makedirs(f'{output_dir}/Morphing_Labels/')
    test_label = [
        [(8, 2, -2), (9, -2, 2), "Hair_Colour"], 
        [(15, -2, 2), "Glasses"], 
        # [(3, 2, -2), "Bags_Under_Eyes"], 
        # [(26, -2, 2), "Pale_Skin"], 
        # [(4, -2, 2), (5, 2, -2), (28, -2, 2), "Bald_Bangs"], 
        # [(23, 2, -2), "Narrow_Eyes"], 
        # [(36, 2, -2), "Lipstick"], 
        # [(21, -2, 2), "Mouth_Open"], 
        # [(31, -2, 2), "Smiling"], 
        [(20, -2, 2), "Male"],  
        # [(16, -1.7, 1.7), (22, -1.7, 1.7), (30, -1.7, 1.7), (24, 1.7, -1.7), "Beard"], 
        # [(16, -2, 2), (22, -2, 2), (30, -2, 2), (24, 2, -2), "Beard"], 
        # [(16, 0, 2), (22, 0, 2), (30, 0, 2), (24, 2, -2), (15, 0, 2), "Glasses_Beard"],
    ]
    # [0] := 182638
    test_data = Subset(dataset, range(182950, 182950+6))
    total_it = len(test_data)
    bar_data = tqdm(test_data, total=total_it)
    imgs = []
    for i, (data, label) in enumerate(bar_data):
        # imgs.append(data.add(1.0).div(2.0))
        with torch.no_grad():
            data = data.unsqueeze(0).to(device)
            for m in test_label:
                for s in range(step):
                    new_label1 = [(i,1.3*(a*(step-1-s)+b*s)/(step-1)) for i,a,b in m[:-1]]
                    prod = model.predict(data, new_label1).squeeze()
                    prod.add_(1.0).div_(2.0)
                    imgs.append(prod)
            vutils.save_image(data.add(1.0).div(2.0), f'{output_dir}/Morphing_Labels/Morphing_label_orgi_{i}.jpg',padding=0)
    img = torch.stack(imgs)
    vutils.save_image(img, f'{output_dir}/Morphing_Labels/Morphing_label.jpg', nrow=step, padding=4)

def pic_morphing(step=7):
    if not os.path.isdir(f'{output_dir}/Morphing_Pics/'):
        os.makedirs(f'{output_dir}/Morphing_Pics/')
    imgs = []
    ids = [(182950,182960), (182952,182962), (182953,182963), (182980,182990), (182950+57,182950+77), (182950+59,182950+79), (182950+60,182950+80), (182950+63,182950+83),
            (182950+100,182950+120),(182950+101,182950+121),(182950+103,182950+123),(182950+109,182950+129),(182950+110,182950+130),(182950+111,182950+131)]
    for i,(i1,i2) in enumerate(ids):
        with torch.no_grad():
            data1 = dataset[i1][0].unsqueeze(0).to(device)
            data2 = dataset[i2][0].unsqueeze(0).to(device)
            z1,_ = model.encode(data1)
            z2,_ = model.encode(data2)
            for s in range(step):
                z = (z1*(step-1-s)+z2*s)/(step-1)
                prod = model.decoder(z)
                prod.add_(1.0).div_(2.0)
                imgs.append(prod.squeeze())
            vutils.save_image(data1.add(1.0).div(2.0), f'{output_dir}/Morphing_Pics/Morphing_pic1_orgi_{i}.jpg',padding=0)
            vutils.save_image(data2.add(1.0).div(2.0), f'{output_dir}/Morphing_Pics/Morphing_pic2_orgi_{i}.jpg',padding=0)
    img = torch.stack(imgs)
    vutils.save_image(img, f'{output_dir}/Morphing_Pics/Morphing_Pics.jpg', nrow=step, padding=4)

def duoLabelChange():
    if not os.path.isdir(f'{output_dir}/Duo_Labels_Changing/'):
        os.makedirs(f'{output_dir}/Duo_Labels_Changing/')
    test_label = [
        [(20, 1), (16, 1), (22, 1), (30, 1), (24, -1)], 
        [(20, 2), (18, 2)], 
        [(21, 2), (31, 2)],
        [(20, 2), (4, 2), (28, 2), (5, -2)],
        [(8, 2), (9, -2), (15, -2)],

        [(20, -1), (16, 2), (22, 2), (30, 2), (24, -1)], 
        [(20, 2), (18, -2)], 
        [(21, 2), (31, -2)],
        [(20, 2), (4, -2), (28, -2), (5, 2)],
        [(8, -2), (9, 2), (15, -2)],

        [(20, 2), (16, -2), (22, -2), (30, -2), (24, 2)],
        [(20, -2), (18, 2)],
        [(21, -2), (31, 2)],
        [(20, -2), (4, 2), (28, 2), (5, -2)],
        [(8, 2), (9, -2), (15, 2)],

        [(20, -2), (16, -2), (22, -2), (30, -2), (24, 2)], 
        [(20, 2), (18, -2)], 
        [(21, -2), (31, -2)],
        [(20, -2), (4, -2), (28, -2), (5, 2)],
        [(8, -2), (9, 2), (15, 2)],
    ]

    for i, (data, label) in enumerate(dataloader_test):
        data = data[:5].to(device)
        b_size = data.shape[0]

    # data, _ = iter(dataloader_test).next()
    # data = data.to(device)
    # b_size = data.shape[0]
    # blank = torch.zeros(b_size, 3, image_size, image_size, device=device)
    # prod = model.predict(data).add_(1.0).div_(2.0)
        with torch.no_grad():
            pic = []
            for tl in test_label:
                # prod = model.predict(data, [])
                prod = model.predict(data, tl)
                pic.append(prod)
            comparison = torch.stack(pic)

        data = data.add_(1.0).div_(2.0).cpu()
        comparison = comparison.add_(1.0).div_(2.0).cpu()
        for j in range(b_size):
            id = i*b_size+j
            vutils.save_image(data[j], f'{output_dir}/Duo_Labels_Changing/{id}_orig.jpg', padding=0)
            vutils.save_image(comparison[:,j], f'{output_dir}/Duo_Labels_Changing/{id}.jpg', nrow=5, padding=4, pad_value=1)

        # vutils.save_image(comparison.cpu(), f'{output_dir}/MSP_CelebA_test_{epoch}.jpg', nrow=b_size, padding=5)
        
        break


print("loading model...")
model.load_state_dict(torch.load(f'{location}/{model_save}/MSP_CelebA.tch',map_location='cpu'))
model.to(device)
model.eval()

print("processing...")
with torch.no_grad():
    randnGenerate()
    label_morphing(7)
    pic_morphing()
    duoLabelChange()
    pass
