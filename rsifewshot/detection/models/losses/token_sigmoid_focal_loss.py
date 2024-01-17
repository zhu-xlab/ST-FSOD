import torch
import torch.nn as nn
import torch.nn.functional as F
from rsidet.models import LOSSES
import pdb

def token_sigmoid_binary_focal_loss(pred_logits, targets, alpha, gamma, text_mask=None, weight=None):
    # binary version of focal loss
    # copied from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor with the reduction option applied.
    """
    assert (targets.dim() == 3)
    assert (pred_logits.dim() == 3)  # batch x from x to

    bs, n, _ = pred_logits.shape
    if text_mask is not None:
        assert (text_mask.dim() == 2)
        text_mask = (text_mask > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, pred_logits.size(1), 1)  # copy along the image channel dimension
        pred_logits = torch.masked_select(pred_logits, text_mask)
        targets = torch.masked_select(targets, text_mask)
        if weight is not None:
            weight = torch.masked_select(weight.view(bs, n, -1), text_mask)

        # print(pred_logits.shape)
        # print(targets.shape)

    p = torch.sigmoid(pred_logits)
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none",
                                                 weight=weight)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss

@LOSSES.register_module()
class TokenSigmoidFocalLoss(nn.Module):
    def __init__(self, alpha, gamma, loss_weight=1.0):
        super(TokenSigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, logits, targets, text_masks=None, version="binary", **kwargs):
        if version == "binary":
            loss_func = token_sigmoid_binary_focal_loss
        elif version == "softmax":
            loss_func = token_sigmoid_softmax_focal_loss
        elif version == "binaryv2":
            loss_func = token_sigmoid_binary_focal_loss_v2
        else:
            raise NotImplementedError
        loss = loss_func(logits, targets, self.alpha, self.gamma, text_masks, **kwargs)
        return loss.sum() * self.loss_weight

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
