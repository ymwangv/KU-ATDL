import torch
from torch.utils.data import DataLoader

from model import Detector
from model import dino_backbones, clip_backbones
from dataset import CPPEDataset, BloodCellDataset
from utils import train_and_validate

import argparse

backbones_ = {
    'dinov2_s': dino_backbones,
    'dinov2_b': dino_backbones,
    'clip_base': clip_backbones
}

datasets_ = {
    'cppe': CPPEDataset,
    'blood': BloodCellDataset,
}

"""
Usage
1. python main.py --dataset blood --backbone dinov2_s --batch_size 16 --num_epochs 100
2. python main.py --dataset cppe --backbone dinov2_s --batch_size 16 --num_epochs 100
3. python main.py --dataset blood --backbone clip_base --batch_size 16 --num_epochs 100
4. python main.py --dataset cppe --backbone clip_base --batch_size 16 --num_epochs 100
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=datasets_.keys(), help="Dataset name")
    parser.add_argument("--backbone", type=str, required=True, choices=backbones_.keys(), help="Backbone name")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")

    args = parser.parse_args()

    dataset_name = args.dataset
    backbone_name = args.backbone
    dataset_dataset = datasets_[dataset_name]
    backbone_dict = backbones_[backbone_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = dataset_dataset(backbone=backbone_name, image_set='train')
    val_dataset = dataset_dataset(backbone=backbone_name, image_set='test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda x: tuple(zip(*x)))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=lambda x: tuple(zip(*x)))

    model = Detector(num_classes=len(train_dataset.classes), backbone_name=backbone_name, backbones_dict=backbone_dict,
                     device=device)

    train_and_validate(model, train_dataloader, val_dataloader, device, dataset_name=dataset_name,
                       backbone_name=backbone_name, num_epochs=args.num_epochs, lr=args.learning_rate)

    torch.save(model.state_dict(), f"train_model_{dataset_name}_{backbone_name}.pt")


main()
