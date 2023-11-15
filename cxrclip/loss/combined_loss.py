from typing import List

import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, loss_list: List[nn.Module]):
        super(CombinedLoss, self).__init__()
        self.loss_list = loss_list

        # loss_ratio_list = []
        # for loss in self.loss_list:
        #     assert hasattr(loss, "name")
        #     assert hasattr(loss, "loss_ratio")
        # loss_ratio = getattr(loss, "loss_ratio")
        # loss_ratio_list.append(loss_ratio)

        # assert sum(loss_ratio_list) == 1.0, f"Sum of {loss_ratio_list} is not 1.0."

    def forward(self, **kwargs):
        loss_dict = dict()
        total_loss = 0.0
        for loss in self.loss_list:
            cur_loss = loss(**kwargs)
            loss_dict[loss.name] = cur_loss
            total_loss += cur_loss * loss.loss_ratio

        loss_dict["total"] = total_loss
        return loss_dict
