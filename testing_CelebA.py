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
parser.add_argument('--epochs', type=int, default=600,
                    help='number of epochs to train (default: 600)')
parser.add_argument('-s', '--save', action='store_true', default=True,
                    help='save model every epoch')
parser.add_argument('-l', '--load', action='store_true',
                    help='load model at the begining')
parser.add_argument('-nf', type=int, default=64,
                    help='output channel number of the first cnn layer (default: 64)')
parser.add_argument('-ep', type=int, default=1,
                    help='starting ep index for outputs')
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

test_data = Subset(dataset, range(182638, 202599))
dataloader_test = DataLoader(test_data, batch_size=120)

model = AE(image_size, label_size, nf, nc=3).to(device)

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


def reconstruction(epoch):
    test_label = [
        [(8, 2, -2), (9, -2, 2), "Hair_Colour"], 
        [(15, -2, 2), "Glasses"],
        [(4, -2, 2), (5, 2, -2), (28, -2, 2), "Bald_Bangs"],
        [(23, 2, -2), "Narrow_Eyes"],
        [(21, -2, 2), "Mouth_Open"], 
        [(31, -2, 2), "Smiling"],
        [(20, -2, 2), "Male"],
        [(16, -1.7, 1.7), (22, -1.7, 1.7), (30, -1.7, 1.7), (24, 1.7, -1.7), "Beard"], 
    ]
    model.eval()
    data, _ = iter(dataloader_test).next()
    data = data.to(device)
    b_size = data.shape[0]
    blank = torch.zeros(b_size, 3, image_size, image_size, device=device)
    prod = model.predict(data).add_(1.0).div_(2.0)
    with torch.no_grad():
        pic = [(data+1)*0.5, prod, blank]
        for tl in test_label:
            for item in make_continue(tl[: -1], 2):
                prod = model.predict(data, item)
                prod.add_(1.0).div_(2.0)
                pic.append(prod)
            pic.append(blank)
        comparison = torch.cat(pic)
    vutils.save_image(comparison.cpu(
    ), f'{location}/{output_dir}/MSP_recon_CelebA0_{epoch}.png', nrow=b_size, padding=4)
    pass

def imageView():
    total_it = len(test_data)
    bar_data = tqdm(test_data, total=total_it)
    pics = []
    fid = 0
    font = ImageFont.load_default()
    tt2 = transforms.ToTensor()
    for i, (data, label) in enumerate(bar_data):
        with torch.no_grad():
            if i % 6 == 0:
                im = Image.new(mode = "RGB", size = (256, 256*3))
                draw = ImageDraw.Draw(im)
                draw.text((200, 384),f"{i}",(255,255,255),font=font)
                imv = tt2(im)
                pics.append(imv.unsqueeze(0))
            data = data.unsqueeze(0)
            prod1 = model.predict(data, [(15, label[15]*-2)])
            prod2 = model.predict(data, [(20, -2), (16, 1.5), (22, 1.5), (30, 1.5), (24, -1.5)])
            img = torch.cat([data,prod1,prod2], dim=2)
            img.add_(1.0).div_(2.0)
            pics.append(img)
        if len(pics) == 140:
            vutils.save_image(torch.cat(pics), f'{output_dir}/ImageView/ImageView_{fid}.png', nrow=7, padding=4)
            pics = []
            fid = i + 1
    vutils.save_image(pics, f'{output_dir}/ImageView/ImageView_{fid}.png', nrow=7, padding=4)

def imageView2():
    name = "RemoveGlass"
    total_it = len(test_data)
    bar_data = tqdm(test_data, total=total_it)
    pic = []
    ids = []
    fid = 0
    id_record = ""
    for i, (data, label) in enumerate(bar_data):
        if label[15] != 1:
            continue
        with torch.no_grad():
            data = data.unsqueeze(0)
            prod1 = model.predict(data, [(15, 0)])
            prod2 = model.predict(data, [(15, -1)])
            prod3 = model.predict(data, [(15, -2)])
            img = torch.cat([data,prod1,prod2,prod3], dim=2)
            img.add_(1.0).div_(2.0)
            pic.append(img)
            ids.append(i)
        if len(pic) == 40:
            vutils.save_image(torch.cat(pic), f'{output_dir}/{name}/Pic_{fid}.png', nrow=10, padding=4)
            id_record += f"Pic_{fid}:\n {str(ids)}\n\n"
            pic = []
            ids = []
            fid+=1
    vutils.save_image(torch.cat(pic), f'{output_dir}/{name}/Pic_{fid}.png', nrow=10, padding=4)
    id_record += f"Pic_{fid}:\n {str(ids)}\n"
    with open(f'{output_dir}/{name}/Ids.txt', "w") as f:
        f.write(id_record)
    
def imageSelect():
    name = "M_FemaleBeard"
    font = ImageFont.load_default()
    tt2 = transforms.ToTensor()
    total_it = len(test_data)
    bar_data = tqdm(test_data, total=total_it)
    S = []
    pics = []
    with torch.no_grad():
        for i, (data, label) in enumerate(bar_data):
            if label[20]<0:
                continue
            data = data.unsqueeze(0)
            prod = model.predict(data, [(20, -2), (16, 1), (22, 1), (30, 1), (24, -1)])
            label_prod, _ = model.encode(prod)
            label_prod = label_prod[0, [20,16,22,30,24]].clamp(-1.,1.)
            score1 = label_prod * torch.tensor([-1.0,1.0,1.0,1.0,-1.0])
            score2 = (data-prod)**2
            score = score1.mean() * (1-score2.mean())
            S.append((i, score))
        S.sort(key = lambda x:x[1], reverse=True)

        bar_data = tqdm(S, total=len(S))
        fid = 1
        for i,_ in bar_data:
            im = Image.new(mode = "RGB", size = (256, 256))
            draw = ImageDraw.Draw(im)
            draw.text((200, 200),f"{i}",(255,255,255),font=font)
            imv = tt2(im)
            pics.append(imv.unsqueeze(0))
            data = test_data[i][0].unsqueeze(0)
            prod = model.predict(data, [(20, -2), (16, 1), (22, 1), (30, 1), (24, -1)])
            data.add_(1.0).div_(2.0)
            prod.add_(1.0).div_(2.0)
            pics.append(data)
            pics.append(prod)
            if len(pics) == 360:
                vutils.save_image(torch.cat(pics), f'{output_dir}/{name}/{fid}.png', nrow=9, padding=4)
                pics = []
                fid += 1
        vutils.save_image(torch.cat(pics), f'{output_dir}/{name}/{fid}.png', nrow=9, padding=4)
    pass



def randnGenerate():
    for i, (data, label) in enumerate(dataloader_test):
        label_size = label.shape[1]
        z, _ = model.encode(data)
        break
    l = z[20:60,:label_size]
    s = torch.randn([40,2048-40])
    z7 = torch.cat([l,s*0.018], dim=1)
    z6 = torch.cat([l,s*0.016], dim=1)
    z0 = torch.cat([l,s*0.014], dim=1)
    z1 = torch.cat([l,s*0.012], dim=1)
    z2 = torch.cat([l,s*0.01], dim=1)
    z3 = torch.cat([l,s*0.008], dim=1)
    z4 = torch.cat([l,s*0.006], dim=1)
    z5 = torch.cat([l,s*0.004], dim=1)
    prod0 = model.decoder(z0)
    prod1 = model.decoder(z1)
    prod2 = model.decoder(z2)
    prod3 = model.decoder(z3)
    prod4 = model.decoder(z4)
    prod5 = model.decoder(z5)
    prod6 = model.decoder(z6)
    prod7 = model.decoder(z7)
    pics = torch.cat([prod7,prod6,prod0,prod1,prod2,prod3,prod4,prod5])
    pics.add_(1.0).div_(2.0)
    vutils.save_image(pics, f'{output_dir}/randnGenerate.jpg', nrow=40, padding=4)

def jianbian():
    for i, (data, label) in enumerate(dataloader_test):
        z, _ = model.encode(data)
        break
    z0 = z[0:20,:]
    z5 = z[20:40,:]
    z1 = z5*0.2 + z0*0.8
    z2 = z5*0.4 + z0*0.6
    z3 = z5*0.6 + z0*0.4
    z4 = z5*0.8 + z0*0.2
    prod0 = model.decoder(z0)
    prod1 = model.decoder(z1)
    prod2 = model.decoder(z2)
    prod3 = model.decoder(z3)
    prod4 = model.decoder(z4)
    prod5 = model.decoder(z5)
    pics = torch.cat([prod0,prod1,prod2,prod3,prod4,prod5])
    pics.add_(1.0).div_(2.0)
    vutils.save_image(pics, f'{output_dir}/Jianbian.jpg', nrow=20, padding=4)

def findImage():
    data_set = test_data
    def load_img(path):
        tt1 = transforms.Resize((225,225))
        tt2 = transforms.ToTensor()
        tt3 = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        with open(path,"rb") as f:
            img = Image.open(f)
            img = tt1(img)
            img = tt2(img)
            img = img[:3]
            img = tt3(img)
        return img
    imgs = [f"{i+1}.png" for i in range(4)]
    imgs = [load_img(i) for i in imgs]

    rst = []
    total_it = len(data_set)
    bar_data = tqdm(data_set, total=total_it)

    for i, p in enumerate(bar_data):
        sc = min([((p[0]-i)**2).sum() for i in imgs])
        rst.append((i,sc))
    rst.sort(key = lambda x:x[1])
    rst = [i for i,_ in rst[:60]]

    print(rst)
    
    batch = [data_set[i][0] for i in rst]
    batch = torch.stack(batch)
    batch.add_(1.0).div_(2.0)
    vutils.save_image(batch, f'{output_dir}/FindImage.jpg', nrow=10, padding=4)
    
    pass

def label_morphing(step=5):
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
            data = data.unsqueeze(0)
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
    imgs = []
    ids = [(182950,182960), (182952,182962), (182953,182963), (182980,182990), (182950+57,182950+77), (182950+59,182950+79), (182950+60,182950+80), (182950+63,182950+83),
            (182950+100,182950+120),(182950+101,182950+121),(182950+103,182950+123),(182950+109,182950+129),(182950+110,182950+130),(182950+111,182950+131)]
    for i,(i1,i2) in enumerate(ids):
        with torch.no_grad():
            data1 = dataset[i1][0].unsqueeze(0)
            data2 = dataset[i2][0].unsqueeze(0)
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

print("loading model...")
model.load_state_dict(torch.load(f'{location}/{model_save}/M_CnnVae_MSP0_CelebA.tch',map_location='cpu'))
model.eval()

print("processing...")
# reconstruction(-1)
imageView()
jianbian()
randnGenerate()
findImage()
imageView2()
imageSelect()
label_morphing(7)
pic_morphing()
pass
