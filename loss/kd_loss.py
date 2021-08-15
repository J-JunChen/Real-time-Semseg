import torch
from torch import nn
import torch.nn.functional as F


def kd_loss_fn(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    
    copy from https://github.com/peterliht/knowledge-distillation-pytorch
    """
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              nn.CrossEntropyLoss(ignore_index=255)(outputs, labels) * (1. - alpha)

    return KD_loss