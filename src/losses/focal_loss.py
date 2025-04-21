import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Cite: GT CS 7643 Deep Learning Assignment 2
"""

def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    effective_number = 1.0 - np.power(beta, cls_num_list)
    
    per_cls_weights = (1.0 - beta) / np.array(effective_number) # reweight
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list) # normalize
    per_cls_weights = torch.tensor(per_cls_weights).float()
        
    return per_cls_weights

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None

        CE = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-CE)
        loss = (1 - pt) ** self.gamma * CE # focal term
        loss = loss.mean() # to scalar
        
        return loss
