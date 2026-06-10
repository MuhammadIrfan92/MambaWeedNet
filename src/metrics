import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def compute_iou_batch(preds, masks, num_classes=3):
    preds = preds.view(-1)
    masks = masks.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = masks == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        ious.append(float("nan") if union == 0 else (intersection / union).item())
    return np.nanmean(ious)


def metrics_from_confusion_matrix(cm):
    ious, precision, recall, f1 = [], [], [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        ious.append(iou); precision.append(p); recall.append(r); f1.append(f)
    return {
        "confusion_matrix": cm.tolist(),
        "pixel_accuracy": float(np.trace(cm) / np.sum(cm)),
        "iou": ious,
        "miou": float(np.mean(ious)),
        "precision": precision,
        "mean_precision": float(np.mean(precision)),
        "recall": recall,
        "mean_recall": float(np.mean(recall)),
        "f1": f1,
        "mean_f1": float(np.mean(f1)),
    }


def evaluate_model(model, loader, device, num_classes=3, return_masks=False):
    model.eval()
    all_preds, all_labels = [], []
    pred_masks, true_masks = [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            all_preds.append(preds.cpu().view(-1))
            all_labels.append(masks.cpu().view(-1))
            if return_masks:
                pred_masks.append(preds.cpu())
                true_masks.append(masks.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    results = metrics_from_confusion_matrix(cm)
    if return_masks:
        results["pred_masks"] = torch.cat(pred_masks, dim=0)
        results["true_masks"] = torch.cat(true_masks, dim=0)
    return results
