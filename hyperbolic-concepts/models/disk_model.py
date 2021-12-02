import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl

import geoopt
from geoopt import ManifoldParameter, Manifold
from geoopt.optim import RiemannianAdam, RiemannianSGD

from ..losses import HierarchyLoss, ConceptClassificationLoss, OverlapLoss

class HyperbolicDiskClassification(pl.LightningModule):

    def __init__(self, concept_hierarchy: Tensor, concept_dim_size: int=5, inner_size=10, outer_size=50, margin=0.5, 
            negative_sample_ratio=10.0):
        ''' concept_hierarchy: NxN bool tensor where i,j = True means i < j '''
        super().__init__()
        self.feature_extractor = models.efficientnet_b0(pretrained=True)
        self.input_size = self.feature_extractor.classifier[1].in_features
        self.feature_extractor.classifier = nn.Identity()

        self.n_concepts = len(concept_hierarchy) 
        self.concept_dim_size = concept_dim_size
        self.manifold = geoopt.PoincareBall(1.0, learnable=False)

        self.radius_ = nn.Parameter(
            torch.log(0.1*torch.ones((n_concepts,)))
        )

        self.concepts_ = geoopt.ManifoldParameter(
            self.manifold.random_normal(self.n_concepts, concept_dim_size, std=0.25), manifold=self.manifold
        )

        self.head = nn.Sequential(
            nn.Linear(self.input_size, outer_size),
            nn.ReLU(),
            nn.Linear(outer_size, inner_size),
        )

        self.projection = nn.Sequential(
            ProjectToManifold(self.manifold),
            MobiusLinear(inner_size, concept_dim_size, dtype=torch.double)
        )

        no_overlap = (~concept_hierarchy) & ((~concept_hierarchy).T) # Things that aren't dominated or don't dominated each other, shouldn't overlap.

        self.overlap_loss = OverlapLoss(self.concepts, self.radius, self.distance, no_overlap)
        self.hierarchy_loss = HierarchyLoss(self.concept, self.radius, self.distance, concept_hierarchy,
                margin=margin)
        self.concept_loss = ConceptClassificationLoss(self.concept, self.radius, self.distance, \
                margin=margin, negative_sample_ratio=negative_sample_ratio)

    @property
    def concept(self):
        return self.manifold.expmap0(self.concepts_)

    @property
    def radius(self):
        return self.radius_.exp()

    def distance(self, a, b):
        return self.manifold.dist(
            a.unsqueeze(1).expand(-1, len(b), -1),
            b.expand(len(a), -1, -1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x).double()
        return self.projection(x)


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        image, target = batch
        embedding = self.forward(image)
        return loss
      
    def configure_optimizers(self):
      optimizer = RiemannianAdam(self.parameters(), lr=1e-3)
      return optimizer

