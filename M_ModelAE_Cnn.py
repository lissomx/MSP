import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class Encoder(nn.Module):
    # only for square pics with width or height is n^(2x)
    def __init__(self, image_size, nf, hidden_size=None, nc=3):
        super(Encoder, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        sequens = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        while(True):
            image_size = image_size/2
            if image_size > 4:
                sequens.append(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False))
                sequens.append(nn.BatchNorm2d(nf * 2))
                sequens.append(nn.LeakyReLU(0.2, inplace=True))
                nf = nf * 2
            else:
                if hidden_size is None:
                    self.hidden_size = int(nf)
                sequens.append(nn.Conv2d(nf, self.hidden_size, int(image_size), 1, 0, bias=False))
                break
        self.main = nn.Sequential(*sequens)

    def forward(self, input):
        return self.main(input).squeeze(3).squeeze(2)


class Decoder(nn.Module):
    # only for square pics with width or height is n^(2x)
    def __init__(self, image_size, nf, hidden_size=None, nc=3):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        sequens = [
            nn.Tanh(),
            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False),
        ]
        while(True):
            image_size = image_size/2
            sequens.append(nn.ReLU(True))
            sequens.append(nn.BatchNorm2d(nf))
            if image_size > 4:
                sequens.append(nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False))
            else:
                if hidden_size is None:
                    self.hidden_size = int(nf)
                sequens.append(nn.ConvTranspose2d(self.hidden_size, nf, int(image_size), 1, 0, bias=False))
                break
            nf = nf*2
        sequens.reverse()
        self.main = nn.Sequential(*sequens)

    def forward(self, z):
        z = z.unsqueeze(2).unsqueeze(2)
        output = self.main(z)
        return output

    def loss(self, predict, orig):
        batch_size = predict.shape[0]
        a = predict.view(batch_size, -1)
        b = orig.view(batch_size, -1)
        L = F.mse_loss(a, b, reduction='sum')
        return L


class CnnVae(nn.Module):
    def __init__(self, image_size, label_size, nf, hidden_size=None, nc=3):
        super(CnnVae, self).__init__()
        self.encoder = Encoder(image_size, nf, hidden_size)
        self.decoder = Decoder(image_size, nf, hidden_size)
        self.image_size = image_size
        self.nc = nc
        self.label_size = label_size
        self.hidden_size = self.encoder.hidden_size

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.M = nn.Parameter(torch.empty(label_size, self.hidden_size))
        nn.init.xavier_normal_(self.M)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        prod = self.decoder(z)
        return prod, z, mu, logvar

    def _loss_vae(self, mu, logvar):
        # https://arxiv.org/abs/1312.6114
        # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD
    
    def _loss_msp(self, label, z):
        L1 = F.mse_loss((z @ self.M.t()).view(-1), label.view(-1), reduction="none").sum()
        L2 = F.mse_loss((label @ self.M).view(-1), z.view(-1), reduction="none").sum()
        return L1 + L2

    def loss(self, prod, orgi, label, z, mu, logvar):
        L_rec = self.decoder.loss(prod, orgi)
        L_vae = self._loss_vae(mu, logvar)
        L_msp = self._loss_msp(label, z) 
        _msp_weight = orgi.numel()/(label.numel()+z.numel())
        Loss = L_rec + L_vae + L_msp * _msp_weight
        return Loss, L_rec.item(), L_vae.item(), L_msp.item()

    def acc(self, z, l):
        zl = z @ self.M.t()
        a = zl.clamp(-1, 1)*l*0.5+0.5
        return a.round().mean().item()

    def predict(self, x, new_ls=None, weight=1.0):
        z, _ = self.encode(x)
        if new_ls is not None:
            zl = z @ self.M.t()
            d = torch.zeros_like(zl)
            for i, v in new_ls:
                d[:,i] = v*weight - zl[:,i]
            z += d @ self.M
        prod = self.decoder(z)
        return prod

    def predict_ex(self, x, label, new_ls=None, weight=1.0):
        return self.predict(x,new_ls,weight)

