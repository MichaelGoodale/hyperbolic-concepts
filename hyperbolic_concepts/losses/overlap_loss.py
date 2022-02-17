import torch
import torch.nn.functional as F

from torch import Tensor, nn

from typing import Callable

class OverlapLoss(nn.Module):

    def __init__(self, model, invalids: Tensor, margin=0.5):
        ''' concepts: a NXD tensor where N is number of concepts and D is dimensionality of the space
            radii: a N tensor where each row is the radius of the corresponding concept in conepts
            invalids: a NXN boolean tensor where invalids[i, j] is False if concept_i and concept_j should overlap'''
        super(OverlapLoss, self).__init__()
        self.model = model
        self.margin = margin
        self.invalids = invalids

    def forward(self, n_samples=256) -> Tensor:
        radii = self.model.radii.expand(self.model.concepts.shape[0], -1)
        overlap_length = (self.margin + radii + radii.T - self.model.distance(self.model.concepts, self.model.concepts))
        overlap_length, _ = torch.topk(F.relu(overlap_length[self.invalids]), n_samples)
        return overlap_length.mean() 

