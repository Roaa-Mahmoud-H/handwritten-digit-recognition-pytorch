"""MNIST dataset and preprocessing helpers."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = 0.1307
STD = 0.3081


def get_dataloaders(data_dir="data", batch_size=128, seed=42):
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])

    train_aug = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
    train_clean = datasets.MNIST(data_dir, train=True, download=True, transform=eval_transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_clean), generator=generator).tolist()
    train_ds = Subset(train_aug, indices[:55000])
    val_ds = Subset(train_clean, indices[55000:])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
