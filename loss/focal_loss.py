# source code from: https://github.com/LiQiufu/WaveSNet

import torch
import torch.nn as nn
import torch.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, batch_average=False, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=None): # gamma=2, aplha=0.25 in official Pytorch
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss



class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, alpha=0.25 , ignore_index=255):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, input, target):
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        logpt = -criterion(input, target.long())
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha > 0:
            loss *= self.alpha
        return loss


# source code from: https://amaarora.github.io/2020/06/29/FocalLoss.html
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=1).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.25).item())
    print(loss.FocalLoss(a, b, gamma=1, alpha=0.25).item())
    print(loss.FocalLoss(a, b, gamma=0.5, alpha=0.25).item())




