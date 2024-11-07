import torch
import torch.nn as nn
from torch.hub import load
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from transformers import CLIPModel
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")

clip_backbones = {
    'clip_base': {
        'name': 'openai/clip-vit-base-patch32',
        'embedding_size': 768,
        'patch_size': 32
    }
}

dino_backbones = {
    'dinov2_s': {
        'name': 'dinov2_vits14',
        'embedding_size': 384,
        'patch_size': 14
    },
    'dinov2_b': {
        'name': 'dinov2_vitb14',
        'embedding_size': 768,
        'patch_size': 14
    },
}


class CLIPBackbone(nn.Module):
    def __init__(self, backbone, backbones):
        super(CLIPBackbone, self).__init__()
        self.out_channels = backbones[backbone]['embedding_size']
        self.patch_size = backbones[backbone]['patch_size']
        self.clip_model = CLIPModel.from_pretrained(backbones[backbone]['name'])
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size = x.shape[0]
        patch_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)

        vision_outputs = self.clip_model.vision_model(x)
        patch_embeddings = vision_outputs.last_hidden_state
        x = patch_embeddings[:, 1:, :]
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, -1, int(patch_dim[0]), int(patch_dim[1]))
        return {"0": x}


class DinoV2Backbone(nn.Module):
    def __init__(self, backbone, backbones):
        super(DinoV2Backbone, self).__init__()
        self.out_channels = backbones[backbone]['embedding_size']
        self.patch_size = backbones[backbone]['patch_size']
        self.dinov2 = load('facebookresearch/dinov2', backbones[backbone]['name'])
        for param in self.dinov2.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size = x.shape[0]
        patch_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)

        x = self.dinov2.forward_features(x)
        x = x['x_norm_patchtokens']
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, self.out_channels, int(patch_dim[0]), int(patch_dim[1]))

        return {"0": x}


class Detector(nn.Module):
    def __init__(self, num_classes, backbone_name, backbones_dict, device):
        super(Detector, self).__init__()
        self.device = device
        self.num_classes = num_classes

        if backbone_name in ['dinov2_s', 'dinov2_b']:
            self.backbone = DinoV2Backbone(backbone_name, backbones_dict)
        elif backbone_name == 'clip_base':
            self.backbone = CLIPBackbone(backbone_name, backbones_dict)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        if backbone_name in ['dinov2_s', 'dinov2_b']:
            self.detection_head = FasterRCNN(
                backbone=self.backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                min_size=448,
                max_size=448,
            )
        elif backbone_name == 'clip_base':
            self.detection_head = FasterRCNN(
                backbone=self.backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                min_size=224,
                max_size=224,
            )

        self.to(device)

    def forward(self, images, targets=None):
        return self.detection_head(images, targets)


def test_detector():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 21

    model = Detector(num_classes=num_classes, backbone_name='dinov2_s', backbones_dict=dino_backbones, device=device)
    print(model)

    dummy_image = [torch.rand(3, 448, 448).to(device)]
    dummy_targets = [{'boxes': torch.FloatTensor([[0, 0, 10, 10]]).to(device), 'labels': torch.tensor([1]).to(device)},
                     {'boxes': torch.FloatTensor([[10, 10, 20, 20]]).to(device),
                      'labels': torch.tensor([2]).to(device)}]

    model.train()
    output = model(dummy_image, dummy_targets)
    for key, value in output.items():
        print(f"{key}: {value}")

    model.eval()
    output = model(dummy_image, dummy_targets)
    pprint(output)


if __name__ == '__main__':
    test_detector()
