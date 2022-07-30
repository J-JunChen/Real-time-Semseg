import torch
from torch import nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction='mean'):
        super(KDLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, outputs, labels, teacher_outputs, alpha, temperature):
        T = temperature
        KD_loss = F.kl_div(F.log_softmax(outputs/T, dim=1),
                            F.softmax(teacher_outputs/T, dim=1),
                            reduction=self.reduction) * (alpha * T * T) + \
                  F.cross_entropy(outputs, labels, 
                            ignore_index=self.ignore_index) * (1. - alpha)

        return KD_loss


class KD_SA_Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(KD_SA_Loss, self).__init__()
        self.reduction = reduction
    
    def forward(self, student_attn, teacher_attn):
        KD_loss = F.kl_div(student_attn, teacher_attn, reduction=self.reduction)

        return KD_loss

# class KD_SA_Loss(nn.Module):
#     def __init__(self, ignore_index=-100, reduction='mean'):
#         super(KD_SA_Loss, self).__init__()
#         self.ignore_index = ignore_index
#         self.reduction = reduction
    
#     def forward(self, outputs, labels, teacher_outputs, sqk_attn, svv_attn, tqk_attn, tvv_attn, alpha, temperature):
#         T = temperature
#         KD_loss = F.kl_div(F.log_softmax(outputs/T, dim=1),
#                             F.softmax(teacher_outputs/T, dim=1),
#                             reduction=self.reduction) * (alpha * T * T) + \
#                   F.cross_entropy(outputs, labels, 
#                             ignore_index=self.ignore_index) * (1. - alpha) + \
#                   F.kl_div(sqk_attn, tqk_attn, reduction=self.reduction) + \
#                   F.kl_div(svv_attn, tvv_attn, reduction=self.reduction)

#         return KD_loss