import torch

from torch import Tensor

from typing import Callable

class OverlapLoss(torch.nn.Module):

    def __init__(self, concepts: Tensor, radii: Tensor, distance: Callable, margin=0.5):
        ''' concepts: a NXD tensor where N is number of concepts and D is dimensionality of the space
            radii: a N tensor where each row is the radius of the corresponding conept in conepts'''
        super(OverlapLoss, self).__init__()
        self.concepts = concepts
        self.radii = radii
        self.distance = distance
        self.margin = margin

    def forward(self, invalids: Tensor, n_samples=256) -> Tensor:
        ''' invalids: a NXN boolean tensor where invalids[i, j] is False if concept_i and concept_j should overlap'''
        radii = self.radii.expand(self.concepts.shape[0], -1)
        overlap_length = (self.margin + radii + radii.T - self.distance(self.concepts, self.concepts))
        overlap_length, _ = torch.topk(F.relu(overlap_length[invalids]), n_samples)
        return overlap_length.mean() 

