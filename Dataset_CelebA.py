import re
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import zipfile
from PIL import Image

# 0 5_o_Clock_Shadow
# 1 Arched_Eyebrows
# 2 Attractive
# 3 Bags_Under_Eyes
# 4 Bald
# 5 Bangs
# 6 Big_Lips
# 7 Big_Nose
# 8 Black_Hair
# 9 Blond_Hair
# 10 Blurry
# 11 Brown_Hair
# 12 Bushy_Eyebrows
# 13 Chubby
# 14 Double_Chin
# 15 Eyeglasses
# 16 Goatee
# 17 Gray_Hair
# 18 Heavy_Makeup
# 19 High_Cheekbones
# 20 Male
# 21 Mouth_Slightly_Open
# 22 Mustache
# 23 Narrow_Eyes
# 24 No_Beard
# 25 Oval_Face
# 26 Pale_Skin
# 27 Pointy_Nose
# 28 Receding_Hairline
# 29 Rosy_Cheeks
# 30 Sideburns
# 31 Smiling
# 32 Straight_Hair
# 33 Wavy_Hair
# 34 Wearing_Earrings
# 35 Wearing_Hat
# 36 Wearing_Lipstick
# 37 Wearing_Necklace
# 38 Wearing_Necktie
# 39 Young


class CelebA(Dataset):
    def __init__(self, path_zip, path_txt, image_size, crop_size=178):
        # path_zip: the path of 'img_align_celeba.zip'
        # path_txt: the path of 'list_attr_celeba.txt'
        # image_size (int or [int,int]): size of ortputed image
        # corp_size (int or [int,int]): copr the CelebA images before resizing.

        self.image_size = image_size
        self.transforms = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        print("loading labels...")
        with open(path_txt, "r") as f:
            _ = f.readline()
            self.label_name = f.readline().strip().split(" ")
            self.label_name = self.label_name
            self.label_size = len(self.label_name)
            self.labels = torch.empty((202599, self.label_size))
            for l in f.readlines():
                name, *labels = re.split(r"\s+", l.strip())
                name = re.search(r'[0-9]+', name).group(0)
                name = int(name)-1
                labels = [1. if i == "1" else -1. for i in labels]
                self.labels[name] = torch.tensor(labels)

        print("loading", path_zip)
        self.zip_file = zipfile.ZipFile(path_zip, "r")

    def __getitem__(self, index):
        return self._read_zipped_image(self.zip_file, abs(index), index < 0), self.labels[index]

    def __len__(self):
        return 202599

    def _read_zipped_image(self, zip_file, seq_id, flip=False):
        # 0 <= seq_id <= 202599
        with self.zip_file.open(f"img_align_celeba/{seq_id+1:06}.jpg") as f:
            # e.g. img_align_celeba.zip/img_align_celeba/000001.jpg
            img = Image.open(f)
            if flip:
                img = transforms.functional.hflip(img)
            img = self.transforms(img)
        return img
