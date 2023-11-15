import torch
import torch.nn as nn
from torch.nn import functional as F

from cxrclip import util

all_gather_func = util.DistAutogradAllGatherFunction(partial=False)

def all_gather(tensor):
    world_size = util.GlobalEnv.get().world_size
    if world_size > 1:
        tensor_list = all_gather_func.apply(tensor)
        all_tensor = torch.cat(tensor_list, 0)
    else:
        all_tensor = tensor
    return all_tensor

class CXRClip(nn.Module):
    def __init__(self, label_smoothing=0.0, i2i_weight=0.0, t2t_weight=0.0, loss_ratio=1.0):
        super(CXRClip, self).__init__()
        self.name = "contrastive"
        self.label_smoothing = label_smoothing
        self.loss_ratio = loss_ratio
        self.i2i_weight = i2i_weight
        self.t2t_weight = t2t_weight

    def forward(self, image_embeddings, text_embeddings, text_embeddings2, image_view_embeddings, labels, logit_scale, is_train, **kwargs):
        world_rank = util.GlobalEnv.get().world_rank
        batch_size = labels.size(0)

        all_image_embeddings = all_gather(image_embeddings)
        all_text_embeddings = all_gather(text_embeddings)
        all_text_embeddings2 = all_gather(text_embeddings2)
        all_image_view_embeddings = all_gather(image_view_embeddings)

        with torch.no_grad():
            labels = labels + (world_rank * batch_size)

        loss_i2t = 0
        loss_t2i = 0

        # I1 - T1
        logits_per_image = logit_scale * image_embeddings @ all_text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ all_image_embeddings.T

        label_smoothing = self.label_smoothing if is_train else 0.0
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        # I2 - T1
        logits_per_image = logit_scale * image_view_embeddings @ all_text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ all_image_view_embeddings.T

        label_smoothing = self.label_smoothing if is_train else 0.0
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        # I1 - T2
        logits_per_image = logit_scale * image_embeddings @ all_text_embeddings2.T
        logits_per_text = logit_scale * text_embeddings2 @ all_image_embeddings.T

        label_smoothing = self.label_smoothing if is_train else 0.0
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        # I2 - T2
        logits_per_image = logit_scale * image_view_embeddings @ all_text_embeddings2.T
        logits_per_text = logit_scale * text_embeddings2 @ all_image_view_embeddings.T

        label_smoothing = self.label_smoothing if is_train else 0.0
        loss_i2t += F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
        loss_t2i += F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

        loss_i2t = loss_i2t / 4.0
        loss_t2i = loss_t2i / 4.0

        # ICL
        loss_i2i = 0

        logits_per_i2i1 = logit_scale * image_embeddings @ all_image_view_embeddings.T
        logits_per_i1i2 = logit_scale * image_view_embeddings @ all_image_embeddings.T

        loss_i2i += F.cross_entropy(logits_per_i2i1, labels)
        loss_i2i += F.cross_entropy(logits_per_i1i2, labels)

        loss_i2i = loss_i2i / 2.0

        # TCL
        loss_t2t = 0

        logits_per_t2t1 = logit_scale * text_embeddings2 @ all_text_embeddings.T
        logits_per_t1t2 = logit_scale * text_embeddings @ all_text_embeddings2.T

        loss_t2t += F.cross_entropy(logits_per_t2t1, labels)
        loss_t2t += F.cross_entropy(logits_per_t1t2, labels)

        loss_t2t = loss_t2t / 2.0

        if is_train:
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "loss/contrastive/steps_i2t", loss_i2t, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "loss/contrastive/steps_t2i", loss_t2i, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "loss/contrastive/steps_i2i", loss_i2i, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "loss/contrastive/steps_t2t", loss_t2t, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "params/logit_scale", logit_scale, util.GlobalEnv.get().summary_writer.global_step
            )
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                "params/temperature", 1.0 / logit_scale, util.GlobalEnv.get().summary_writer.global_step
            )

        # contrastive loss
        loss = (loss_i2t + loss_t2i) / 2.0  # shape: (batch_size,)
        loss += loss_i2i * self.i2i_weight
        loss += loss_t2t * self.t2t_weight

        return loss.mean()
