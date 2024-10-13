import torch
from skimage.metrics import hausdorff_distance
from utils import intersection, union

def calculate_hausdorff(gt, pred):
    # Ensure pred and gt are binary masks (1 for object, 0 for background)
    gt_np = gt.cpu().numpy() if torch.is_tensor(gt) else gt
    pred_np = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    return hausdorff_distance(gt_np, pred_np)

def calculate_iou(pred, gt):
    inter = torch.sum(intersection(pred, gt)).float()
    uni = torch.sum(union(pred, gt)).float()
    iou = inter / uni
    return iou
