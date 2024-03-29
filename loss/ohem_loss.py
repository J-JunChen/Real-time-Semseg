#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_index=-100):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criterion(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


" reference code: https://github.com/MichaelFan01/STDC-Seg "
# class OhemCELoss(nn.Module):
#     def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
#         super(OhemCELoss, self).__init__()
#         self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
#         self.n_min = n_min
#         self.ignore_lb = ignore_lb
#         self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

#     def forward(self, logits, labels):
#         N, C, H, W = logits.size()
#         loss = self.criteria(logits, labels).view(-1)
#         loss, _ = torch.sort(loss, descending=True)
#         if loss[self.n_min] > self.thresh:
#             loss = loss[loss>self.thresh]
#         else:
#             loss = loss[:self.n_min]
#         return torch.mean(loss)

# class WeightedOhemCELoss(nn.Module):
#     def __init__(self, thresh, n_min, num_classes, ignore_lb=255, *args, **kwargs):
#         super(WeightedOhemCELoss, self).__init__()
#         self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
#         self.n_min = n_min
#         self.ignore_lb = ignore_lb
#         self.num_classes = num_classes
#         # self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

#     def forward(self, logits, labels):
#         N, C, H, W = logits.size()
#         criteria = nn.CrossEntropyLoss(weight=enet_weighing(labels, self.num_classes).cuda(), ignore_index=self.ignore_lb, reduction='none')
#         loss = criteria(logits, labels).view(-1)
#         loss, _ = torch.sort(loss, descending=True)
#         if loss[self.n_min] > self.thresh:
#             loss = loss[loss>self.thresh]
#         else:
#             loss = loss[:self.n_min]
#         return torch.mean(loss)

# class SoftmaxFocalLoss(nn.Module):
#     def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.nll = nn.NLLLoss(ignore_index=ignore_lb)

#     def forward(self, logits, labels):
#         scores = F.softmax(logits, dim=1)
#         factor = torch.pow(1.-scores, self.gamma)
#         log_score = F.log_softmax(logits, dim=1)
#         log_score = factor * log_score
#         loss = self.nll(log_score, labels)
#         return loss

