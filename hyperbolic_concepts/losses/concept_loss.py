import torch
import torch.nn.functional as F

from torch import Tensor, nn

from typing import Callable

class ConceptClassificationLoss(nn.Module):

    def __init__(self, margin=0.5, negative_sample_ratio:float = 10.0):
        ''' concepts: a NXD tensor where N is number of concepts and D is dimensionality of the space
            radii: a N tensor where each row is the radius of the corresponding conept in conepts'''
        super(ConceptClassificationLoss, self).__init__()
        self.margin = margin
        self.negative_sample_ratio = negative_sample_ratio 

    def forward(self, embeddings: Tensor, targets: Tensor) -> Tensor:
        ''' embeddings: a NXD tensor where N is batch_size and D is dimensionality of the space
            targets: a NXC boolean tensor where N is batch_size and C is the number of concepts'''
        distance = model.distance(embeddings, model.concept)
        radii = model.radius.expand(len(embeddings), -1)
        pos_radius_loss = torch.square(F.relu(self.margin + distance[targets]  - radii[targets]))
        neg_radius_loss = torch.square(F.relu(self.margin - distance[~targets] + radii[~targets]))

        n_neg_samples = int(min(self.negative_sample_ratio*len(pos_radius_loss), (~targets).sum()))
        neg_radius_loss, _ = torch.topk(neg_radius_loss, k=n_neg_samples)

        perm = torch.randperm(len(neg_radius_loss))[:len(pos_radius_loss)]
        neg_radius_loss = neg_radius_loss[perm]

        return (pos_radius_loss.sum() + neg_radius_loss.sum()) / (len(pos_radius_loss) + len(neg_radius_loss))
