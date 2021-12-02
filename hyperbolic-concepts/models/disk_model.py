class HyperbolicDiskClassification(pl.LightningModule):

    def __init__(self, n_concepts: int=10, concept_dim_size: int=5, inner_size=10, outer_size=50)
          super().__init__()
          self.feature_extractor = models.efficientnet_b0(pretrained=True)
          self.input_size = self.feature_extractor.classifier[1].in_features
          self.feature_extractor.classifier = nn.Identity()

          self.n_concepts = n_concepts
          self.concept_dim_size = concept_dim_size
          self.manifold = geoopt.PoincareBall(1.0, learnable=False)

          self.radius_ = torch.nn.Parameter(
              torch.log(0.1*torch.ones((n_concepts,)))
          )

          self.concepts_ = geoopt.ManifoldParameter(
              self.manifold.random_normal(n_concepts, concept_dim_size, std=0.25), manifold=self.manifold
          )

          self.head = torch.nn.Sequential(
              torch.nn.Linear(self.input_size, outer_size),
              torch.nn.ReLU(),
              torch.nn.Linear(outer_size, inner_size),
          )

          self.projection = torch.nn.Sequential(
              ProjectToManifold(self.manifold),
              MobiusLinear(inner_size, concept_dim_size, dtype=torch.double)
          )

    @property
    def concept(self):
      return self.manifold.expmap0(self.concepts_)

    @property
    def radius(self):
      return self.radius_.exp()

    def distance(self, a, b):
        return self.manifold.dist(
            a.unsqueeze(1).expand(-1, self.n_concepts, -1),
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

