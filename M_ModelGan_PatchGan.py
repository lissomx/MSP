import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F


class PatchGan(nn.Module):
    # only for square pics with width or height is n^(2x)
    def __init__(self, image_size, nf=16, layer=3, nc=3):
        super(PatchGan, self).__init__()
        self.image_size = image_size
        sequens = [
            nn.Conv2d(nc, nf, 3, 2, 1, bias=False),
        ]
        for ly in range(layer):
            nf = nf * 2
            image_size = image_size / 2
            sequens.append(nn.LeakyReLU(0.2, True))
            sequens.append(nn.Conv2d(int(nf/2), nf, 3, 2, 1, bias=False))
            sequens.append(nn.BatchNorm2d(nf))
        sequens += [
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, 1, 3, 1, 1, bias=False)
        ]

        self.main = nn.Sequential(*sequens)

    def forward(self, input):
        output = self.main(input)
        return output.view(output.shape[0],-1)

    def loss(self, prod, orgi, training_discriminator=True):
        b_size = prod.shape[0]
        prod = prod if training_discriminator else prod.clamp(0,1)
        L = F.mse_loss(prod.view(-1), orgi.view(-1), reduction="none")
        L = L.view(b_size,-1).sum(dim=1)
        return L * 4
