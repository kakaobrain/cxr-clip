import torch.nn as nn


class Classification(nn.Module):
    def __init__(self, loss_ratio=1.0, **kwargs):
        super(Classification, self).__init__()
        self.name = "classification"
        self.loss_ratio = loss_ratio
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, cls_pred, target_class, **kwargs):
        # valid = target_class > -1
        # loss = self.bce(cls_pred[valid], target_class[valid])
        target_class[target_class < 0] = 0
        loss = self.bce(cls_pred, target_class)
        return loss
