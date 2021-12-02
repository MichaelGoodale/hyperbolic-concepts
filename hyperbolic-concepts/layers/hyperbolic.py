import torch.nn
import geoopt
'''From https://github.com/geoopt/geoopt'''

# package.nn.modules.py
def create_ball(ball=None, c=None):
    """
    Helper to create a PoincareBall.
    Sometimes you may want to share a manifold across layers, e.g. you are using scaled PoincareBall.
    In this case you will require same curvature parameters for different layers or end up with nans.
    Parameters
    ----------
    ball : geoopt.PoincareBall
    c : float
    Returns
    -------
    geoopt.PoincareBall
    """
    if ball is None:
        assert c is not None, "curvature of the ball should be explicitly specified"
        ball = geoopt.PoincareBall(c)
    # else trust input
    return ball


class MobiusLinear(torch.nn.Linear):
    def __init__(self, *args, nonlin=None, ball=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # for manifolds that have parameters like Poincare Ball
        # we have to attach them to the closure Module.
        # It is hard to implement device allocation for manifolds in other case.
        self.ball = create_ball(ball, c)

        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            nonlin=self.nonlin,
            ball=self.ball,
        )

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()


# package.nn.functional.py
def mobius_linear(input, weight, bias=None, nonlin=None, *, ball: geoopt.PoincareBall):
    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    if nonlin is not None:
        output = ball.logmap0(output)
        output = nonlin(output)
        output = ball.expmap0(output)
    return output

class ProjectToManifold(torch.nn.Module):
    def __init__(self, manifold: Lorentz):
        super().__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.expmap0(input)

class ManifoldEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, manifold: Manifold):
        super().__init__()
        self.manifold=manifold
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = geoopt.ManifoldParameter(
            torch.empty(num_embeddings, embedding_dim), manifold=self.manifold
        )

    def forward(self, x: Tensor):
        self.weight[x]

