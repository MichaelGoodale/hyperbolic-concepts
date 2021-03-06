import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl

import geoopt
from geoopt import ManifoldParameter, Manifold
from geoopt.optim import RiemannianAdam, RiemannianSGD

from ..losses import HierarchyLoss, ConceptClassificationLoss, OverlapLoss
from ..layers import ProjectToManifold, MobiusLinear

class HyperbolicDiskClassification(pl.LightningModule):

    def __init__(self, concept_hierarchy: Tensor, concept_dim_size: int=5, inner_size=10, outer_size=50, margin=0.5, 
            negative_sample_ratio=10.0, loss_ratios=None):
        ''' concept_hierarchy: NxN bool tensor where i,j = True means i < j '''
        super().__init__()

        self.n_concepts = len(concept_hierarchy)
        self.concept_dim_size = concept_dim_size

        if loss_ratios is None:
            loss_ratios = {"concept_loss": 0.7125,
                           "hierarchy_loss": 0.2375,
                           "overlap_loss": 0.05}
        self.loss_ratios = loss_ratios

        self.manifold = geoopt.PoincareBall(1.0, learnable=False)

        self.feature_extractor = models.efficientnet_b0(pretrained=True)
        self.input_size = self.feature_extractor.classifier[1].in_features
        self.feature_extractor.classifier = nn.Identity()

        self.radius_ = nn.Parameter(
            torch.log(0.1*torch.ones((self.n_concepts,)))
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

        self.overlap_loss = OverlapLoss(no_overlap)
        self.hierarchy_loss = HierarchyLoss(concept_hierarchy,
                margin=margin)
        self.concept_loss = ConceptClassificationLoss(margin=margin,
                negative_sample_ratio=negative_sample_ratio)

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

    def losses(self, embedding, target):
        loss  = self.loss_ratios['hierarchy_loss'] * self.hierarchy_loss(self) 
        loss += self.loss_ratios['overlap_loss']   * self.overlap_loss(self)
        loss += self.loss_ratios['concept_loss']   * self.concept_loss(self, embedding, target)
        return loss

    def training_step(self, batch, batch_idx):
        image, target = batch
        embedding = self.forward(image)
        loss = self.losses(embedding, target)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        embedding = self.forward(image)
        loss = self.losses(embedding, target)
        return loss
      
    def configure_optimizers(self):
      optimizer = RiemannianAdam(self.parameters(), lr=1e-3)
      return optimizer

