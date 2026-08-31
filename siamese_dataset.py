from torch.utils.data import Dataset
import random
import torch


class SiameseDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        imgA, labelA = self.data[index]

        same_class_flag = random.randint(0, 1)  # pair with same class?

        if same_class_flag:  # yes, pair with same class
            labelB = -1
            while labelB != labelA:
                imgB, labelB = random.choice(self.data)

        else:  # no, pair with different class
            labelB = labelA
            while labelB == labelA:
                imgB, labelB = random.choice(self.data)

        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        pair_label = torch.tensor([labelA != labelB], dtype=torch.float32)

        return imgA, imgB, pair_label
