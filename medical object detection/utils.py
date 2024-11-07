import torch
import torch.optim as optim
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import logging
from datetime import datetime


def evaluate(model, data_loader, device):
    map_metric = MeanAveragePrecision(iou_thresholds=[0.5])
    model.eval()

    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate mAP
            outputs = model(images)
            target_format = [
                {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu()
                }
                for target in targets
            ]
            preds = [
                {
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu()
                }
                for output in outputs
            ]
            map_metric.update(preds, target_format)

    map_score = map_metric.compute()

    return map_score['map']


def train_and_validate(model, train_loader, val_loader, device, dataset_name, backbone_name, num_epochs, lr):
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}_{backbone_name}.txt"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
    )

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"Total Trainable Parameters: {trainable_params}")
    print(f"Total Frozen Parameters: {frozen_params}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            losses.backward()
            optimizer.step()

        map_score = evaluate(model, val_loader, device)
        log_message = f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation mAP: {map_score:.4f}"
        print(log_message)
        logging.info(log_message)

    print("Training and validation completed")
