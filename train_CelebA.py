import argparse
import math
from tqdm import tqdm
import operator
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
from scipy import linalg, matrix
import time

from Dataset_CelebA import CelebA
from M_ModelAE_Cnn import CnnVae as AE
from M_ModelGan_PatchGan import PatchGan as Gan

parser = argparse.ArgumentParser(description='C0AE for CelebA')
parser.add_argument('-bz', '--batch-size', type=int, default=70,
                    help='input batch size for training (default: 70)')
parser.add_argument('-iz', '--image-size', type=int, default=256,
                    help='size to resize for CelebA pics (default: 256)')
parser.add_argument('--epochs', type=int, default=80,
                    help='number of epochs to train (default: 80)')
parser.add_argument('-s', '--save', action='store_true', default=True,
                    help='save model every epoch')
parser.add_argument('-l', '--load', action='store_true',
                    help='load model at the begining')
parser.add_argument('-nf', type=int, default=64,
                    help='output channel number of the first cnn layer (default: 64)')
parser.add_argument('-ep', type=int, default=1,
                    help='starting ep index for outputs')
parser.add_argument('-pg', action='store_true',
                    help='show tqdm bar')
args = parser.parse_args()

# args.load = True
# args.save = False
# args.pg = True

print(args)

celeba_zip = "CelebA_Dataset/img_align_celeba.zip"
celeba_txt = "CelebA_Dataset/list_attr_celeba.txt"
model_save = 'model_save/'
output_dir = 'Outputs/'

print("CelebA zip file: ", os.path.abspath(celeba_zip))
print("CelebA txt file: ", os.path.abspath(celeba_txt))
print("model save: ", os.path.abspath(model_save))
print("output dir: ", os.path.abspath(output_dir))

if not os.path.isdir(model_save):
    os.makedirs(model_save)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

batch_size = args.batch_size
image_size = args.image_size
nf = args.nf
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CelebA(celeba_zip, celeba_txt, image_size)
label_size = dataset.label_size

train_data = Subset(dataset, range(0, 182638))
test_data = Subset(dataset, range(182638, 202600))

dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test_data, batch_size=min(40, batch_size))


model = AE(image_size, label_size, nf, nc=3).to(device)
discriminator = Gan(image_size, nf=nf, layer=int(math.log2(image_size)-5)).to(device)

# print(model)
# print(discriminator)

optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer2 = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)


def train(ep):
    model.train()
    discriminator.train()

    epoch_loss = 0
    epoch_loss_rec = 0
    epoch_loss_vae = 0
    epoch_loss_msp = 0
    epoch_loss_pch = 0
    epoch_acc = 0
    epoch_D0 = []
    epoch_D1 = []
    total = 0

    bar_data = dataloader_train
    if args.pg:
        total_it = len(dataloader_train)
        bar_data = tqdm(dataloader_train, total=total_it)
    for i, (data, label) in enumerate(bar_data):
        b_size = data.shape[0]
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        prod, z, *other = model(data)

        # train discriminator
        optimizer2.zero_grad()
        real = discriminator(data)
        fake = discriminator(prod.detach())
        D=(real.mean().item(),fake.mean().item())
        const = torch.ones_like(real)
        loss_real = discriminator.loss(real, const)
        loss_fake = discriminator.loss(fake, const-1)
        loss_gan = loss_real + loss_fake
        loss_gan.sum().backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
        optimizer2.step()
        optimizer2.zero_grad()
        # end

        Loss, l_rec, l_vae, l_msp = model.loss(prod, data, label, z, *other)
        fake = discriminator(prod)
        Loss_pch = discriminator.loss(fake, const, False).sum()

        acc = model.acc(z, label)
        epoch_loss += Loss.item()
        epoch_loss_rec += l_rec
        epoch_loss_vae += l_vae
        epoch_loss_msp += l_msp
        epoch_loss_pch += Loss_pch.item()
        epoch_acc += acc
        epoch_D0.append(D[0])
        epoch_D1.append(D[1])
        total += b_size

        if args.pg:
            bar_data.set_description(f"ep{ep} -- Loss: {Loss.item()/b_size:.0f}, loss_rec: {l_rec/b_size:.0f},  loss_vae: {l_vae/b_size:.0f}, loss_msp: {l_msp/b_size:.0f}, loss_gan: {Loss_pch.item()/b_size:.0f}|r{D[0]:.3f}|f{D[1]:.3f}, acc: {acc:.4f}")
            
        (Loss+Loss_pch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    mean_D0,mean_D1 = sum(epoch_D0)/i, sum(epoch_D1)/i
    std_D0,std_D1 = sum([abs(mean_D0-i) for i in epoch_D0])/i, sum([abs(mean_D1-i) for i in epoch_D1])/i
    return epoch_loss/total, epoch_loss_rec/total, epoch_loss_vae/total, epoch_loss_msp/total, epoch_loss_pch/total, epoch_acc/i, mean_D0,mean_D1,std_D0,std_D1


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
        [(16, -2, 2), (22, -2, 2), (30, -2, 2), (24, 2, -2), "Beard"], 
    ]
    model.eval()
    discriminator.eval()
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
    ), f'{output_dir}/MSP_CelebA_test_{epoch}.jpg', nrow=b_size, padding=4)


if args.load:
    print("loading model...")
    ep_prefix = 32
    model.load_state_dict(torch.load(f'{model_save}/MSP_CelebA.tch'))
    pack = torch.load(f'{model_save}/MSP_CelebA.opt.tch')
    optimizer.load_state_dict(pack["optimizer"])
    discriminator.load_state_dict(pack["discriminator"])
    optimizer2.load_state_dict(pack["optimizer2"])
    args.ep = pack["ep"]+1
    model.eval()
    discriminator.eval()
    print("loaded")

reconstruction(0)

print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | starting training ...")
for ep in range(args.ep, args.epochs):
    loss, loss_rec, loss_vae, loss_msp, loss_pch, acc, mean_D0,mean_D1,std_D0,std_D1 = train(ep)
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    print(f"{localtime} | M ep:{ep} == loss: {loss:.0f}, loss_rec: {loss_rec:.0f}, loss_vae: {loss_vae:.0f}, loss_msp: {loss_msp:.0f}, loss_pch: {loss_pch:.0f}, acc: {acc:.4f} == Gan: r{mean_D0:.2f}±{std_D0:.2f} | f{mean_D1:.2f}±{std_D1:.2f}")

    reconstruction(ep)
    if args.save:
        torch.save(model.state_dict(), f"{model_save}/MSP_CelebA.tch")
        torch.save({
            "ep" : ep,
            "optimizer" : optimizer.state_dict(),
            "discriminator" : discriminator.state_dict(),
            "optimizer2" : optimizer2.state_dict()
            }, f'{model_save}/MSP_CelebA.opt.tch')
pass
