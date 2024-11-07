import torch
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets


# class BrainTumorDataset(Dataset):
#     def __init__(self, image_set="train"):
#         brain_tumor = load_dataset('mmenendezg/brain-tumor-object-detection')
#         if image_set == "train":
#             self.dataset = concatenate_datasets([brain_tumor['train'], brain_tumor['validation']])
#         else:
#             self.dataset = brain_tumor['test'].select(range(30))
#         self.resize_shape = (448, 448)
#         self.transform = transforms.Compose([
#             transforms.Resize(self.resize_shape),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         self.classes = [
#             "background",
#             'No',
#             'Yes',
#         ]
#         self.label_map = {name: i for i, name in enumerate(self.classes)}
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         image = self.dataset[idx]['image'].convert("RGB")
#
#         orig_width, orig_height = image.size
#         new_width, new_height = self.resize_shape
#
#         scale_x = new_width / orig_width
#         scale_y = new_height / orig_height
#
#         boxes = []
#         for obj in self.dataset[idx]['objects']['bbox']:
#             x_min, y_min, width, height = obj
#             x_max = x_min + width
#             y_max = y_min + height
#
#             xmin = x_min * scale_x
#             ymin = y_min * scale_y
#             xmax = x_max * scale_x
#             ymax = y_max * scale_y
#
#             boxes.append([xmin, ymin, xmax, ymax])
#
#         labels = []
#         for obj in self.dataset[idx]['objects']['label']:
#             labels.append(obj + 1)
#
#         formatted_target = {
#             "boxes": torch.tensor(boxes, dtype=torch.float32),
#             "labels": torch.tensor(labels, dtype=torch.int64)
#         }
#
#         image = self.transform(image)
#
#         return image, formatted_target


class BloodCellDataset(Dataset):
    def __init__(self, backbone, image_set="train"):
        self.backbone = backbone

        blood_cell = load_dataset("keremberke/blood-cell-object-detection", name="full")

        if image_set == "train":
            self.dataset = concatenate_datasets([blood_cell['train'], blood_cell['validation']])
        else:
            self.dataset = blood_cell['test']

        if backbone in ['dinov2_s', 'dinov2_b']:
            self.resize_shape = (448, 448)
        elif backbone == 'clip_base':
            self.resize_shape = (224, 224)

        self.transform = transforms.Compose([
            transforms.Resize(self.resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.classes = [
            "background",
            'platelets',
            'rbc',
            'wbc',
        ]
        self.label_map = {name: i for i, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image = self.dataset[idx]['image'].convert("RGB")

        orig_width, orig_height = image.size
        new_width, new_height = self.resize_shape

        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        boxes = []
        for obj in self.dataset[idx]['objects']['bbox']:
            x_min, y_min, width, height = obj
            x_max = x_min + width
            y_max = y_min + height

            xmin = x_min * scale_x
            ymin = y_min * scale_y
            xmax = x_max * scale_x
            ymax = y_max * scale_y

            boxes.append([xmin, ymin, xmax, ymax])

        labels = []
        for obj in self.dataset[idx]['objects']['category']:
            labels.append(obj + 1)

        formatted_target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        image = self.transform(image)

        return image, formatted_target


class CPPEDataset(Dataset):
    def __init__(self, backbone, image_set="train"):
        self.backbone = backbone

        cppe = load_dataset("cppe-5")

        self.dataset = cppe[image_set]

        if backbone in ['dinov2_s', 'dinov2_b']:
            self.resize_shape = (448, 448)
        elif backbone == 'clip_base':
            self.resize_shape = (224, 224)

        self.transform = transforms.Compose([
            transforms.Resize(self.resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.classes = [
            "Background",
            "Coverall",
            "Face_Shield",
            "Gloves",
            "Goggles",
            "Mask"
        ]
        self.label_map = {name: i for i, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image'].convert("RGB")

        orig_width, orig_height = image.size
        new_width, new_height = self.resize_shape

        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        boxes = []
        for obj in self.dataset[idx]['objects']['bbox']:
            x_min, y_min, width, height = obj
            x_max = x_min + width
            y_max = y_min + height

            xmin = x_min * scale_x
            ymin = y_min * scale_y
            xmax = x_max * scale_x
            ymax = y_max * scale_y

            boxes.append([xmin, ymin, xmax, ymax])

        labels = []
        for obj in self.dataset[idx]['objects']['category']:
            labels.append(obj + 1)

        formatted_target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        image = self.transform(image)

        return image, formatted_target
