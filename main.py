import torch
from torch import optim
from siamese_dataset import SiameseDataset
from torch.utils.data import DataLoader
from siamese_model import SiameseNetwork, ContrastiveLoss

# import numpy as np
# import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def main():
    mnist_train = MNIST(root="./data", train=True, download=True)
    mnist_test = MNIST(root="./data", train=False, download=True)
    transform = transforms.Compose([transforms.ToTensor()])

    siamese_train = SiameseDataset(mnist_train, transform)
    siamese_test = SiameseDataset(mnist_test, transform)

    train_dataloader = DataLoader(
        siamese_train, shuffle=True, num_workers=8, batch_size=64
    )
    model = SiameseNetwork().cpu()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        total_loss = 0

        for imgA, imgB, label in train_dataloader:

            imgA, imgB, label = imgA.cpu(), imgB.cpu(), label.cpu()
            optimizer.zero_grad()
            outputA, outputB = model(imgA, imgB)
            loss_contrastive = criterion(outputA, outputB, label)
            loss_contrastive.backward()

            total_loss += loss_contrastive.item()
            optimizer.step()

        print(f"Epoch {epoch}; Loss {total_loss}")


if __name__ == "__main__":
    main()
