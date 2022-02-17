import torch
import torch.nn.functional as F

from torch import Tensor, nn

from typing import Callable

class HierarchyLoss(nn.Module):

    def __init__(self, model, valids: Tensor, margin: float =0.5):
        ''' concepts: a NXD tensor where N is number of concepts and D is dimensionality of the space
            radii: a N tensor where each row is the radius of the corresponding conept in conepts
            valids: a NXN boolean tensor where invalids[i, j] is True if concept_i < concept_j '''
        super(HierarchyLoss, self).__init__()
        self.margin = margin
        self.model = model 

        self.valids = valids
        self.invalids = ~self.valids
        self.invalids.fill_diagonal_(False)


    def forward(self) -> Tensor:
        radii = self.model.radii.expand(self.model.concepts.shape[0], -1) 
        distances =  self.model.distance(self.model.concepts, self.model.concepts) - radii.T + radii
        pos_loss = torch.square(F.relu(distances[self.valids]))

        neg_loss = torch.square(F.relu(self.margin - distances[self.invalids]))

        neg_loss, _ = torch.topk(neg_loss, k=min(len(pos_loss), len(neg_loss)))
        hier_loss = (pos_loss.sum()+neg_loss.sum())/(len(pos_loss) + len(neg_loss))
        return hier_loss
