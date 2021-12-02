import torch

from torch import Tensor

from typing import Callable

class HierarchyLoss(torch.nn.Module):

    def __init__(self, concepts: Tensor, radii: Tensor, distance: Callable, margin: float =0.5):
        ''' concepts: a NXD tensor where N is number of concepts and D is dimensionality of the space
            radii: a N tensor where each row is the radius of the corresponding conept in conepts'''
        super(OverlapLoss, self).__init__()
        self.concepts = concepts
        self.radii = radii
        self.distance = distance
        self.margin = margin

    def forward(self, valids: Tensor) -> Tensor:
        ''' valids: a NXN boolean tensor where invalids[i, j] is True if concept_i < concept_j '''
        radii = self.radii.expand(self.concepts.shape[0], -1) 
        distances =  self.distance(self.concepts, self.concepts) - radii.T + radii
        pos_loss = torch.square(F.relu(distances[valids]))

        invalids = ~valids
        invalids.fill_diagonal_(False)
        neg_loss = torch.square(F.relu(self.margin - distances[invalids]))

        neg_loss, _ = torch.topk(neg_loss, k=min(len(pos_loss), len(neg_loss)))
        hier_loss = (pos_loss.sum()+neg_loss.sum())/(len(pos_loss) + len(neg_loss))
        return hier_loss
