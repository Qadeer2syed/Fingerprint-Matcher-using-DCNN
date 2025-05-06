import torch.nn.functional as F
import torch

def contrastive_loss(e1, e2, label, margin=1.0):
    # label: 1 = match, 0 = non-match
    dist_sq = F.pairwise_distance(e1, e2, p=2)
    loss_pos = label * dist_sq
    loss_neg = (1-label) * torch.clamp(margin - dist_sq, min=0.0)
    return torch.mean(loss_pos + loss_neg)