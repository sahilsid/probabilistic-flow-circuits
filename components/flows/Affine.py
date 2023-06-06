
import math
import torch
import torch.nn as nn
from torch.distributions import Transform, constraints
from pyro.nn import DenseNN
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
import torch.nn.functional as F

class ConditionedScale(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    volume_preserving = True

    def __init__(self, D=None):
        super().__init__(cache_size=1)
        self.D = D
        
    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        return x*self.D

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. The Householder transformation, H, is "involutory," i.e.
        H^2 = I. If you reflect a point around a plane, then the same operation will
        reflect it back
        """

        return y*torch.reciprocal(self.D)

    def log_abs_det_jacobian(self, x, y):
        r"""
        Calculates the elementwise determinant of the log jacobian. Householder flow
        is measure preserving, so :math:`\log(|detJ|) = 0`
        """
        log_det = self.D.abs().log().sum(dim=-1)
        return log_det*torch.ones(
            x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device
        )


class Scale(ConditionedScale, TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    volume_preserving = False

    def __init__(self, input_dim):
        super().__init__()
        self.D = nn.Parameter(torch.Tensor(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.D.size(-1))
        self.D.data.uniform_(-stdv, stdv)



class IndexedScale(ConditionalTransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, n_index, input_dim, min_scale=0.1, max_scale=5.,):
        super().__init__()
        self.input_dim = input_dim
        self.n_index   = n_index
        self.W  = nn.Parameter(torch.randn(self.n_index, self.input_dim))
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def get_diagonal(self, context):
        return torch.index_select(self.W,0,context)

    def condition(self, context):
        D = self.get_diagonal(context)
        D = torch.nn.functional.softplus(D)
        return ConditionedScale(D)



class ConditionalScale(ConditionalTransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, n_index, input_dim, min_scale=0.1, max_scale=5.,):
        super().__init__()
        self.input_dim = input_dim
        self.n_index   = n_index
        self.nn = DenseNN(1, [64,128,64], param_dims=[self.input_dim])
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def get_diagonal(self, context):
        context = context.view(-1,1).to(torch.float32)
        return self.nn(context)

    def condition(self, context):
        D = self.get_diagonal(context)
        D = torch.nn.functional.softplus(D)
        return ConditionedScale(D)


class ConditionedTranslate(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    volume_preserving = True

    def __init__(self, b=None):
        super().__init__(cache_size=1)
        self.b = b
        
    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        return x + self.b

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. The Householder transformation, H, is "involutory," i.e.
        H^2 = I. If you reflect a point around a plane, then the same operation will
        reflect it back
        """

        return y - self.b

    def log_abs_det_jacobian(self, x, y):
        r"""
        Calculates the elementwise determinant of the log jacobian. Householder flow
        is measure preserving, so :math:`\log(|detJ|) = 0`
        """
        return torch.zeros(
            x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device
        )


class Translate(ConditionedTranslate, TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    volume_preserving = True

    def __init__(self, input_dim):
        super().__init__()
        self.b = nn.Parameter(torch.Tensor(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.b.size(-1))
        self.b.data.uniform_(-stdv, stdv)



class IndexedTranslate(ConditionalTransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, n_index, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.n_index   = n_index
        self.W  = nn.Parameter(torch.randn(self.n_index, self.input_dim))

    def get_bias(self, context):
        return torch.index_select(self.W,0,context)

    def condition(self, context):
        b = self.get_bias(context)
        return ConditionedTranslate(b)


class ConditionalTranslate(ConditionalTransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, n_index, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.n_index   = n_index
        self.nn = DenseNN(1,[64,128,64], param_dims=[self.input_dim])

    def get_bias(self, context):
        context = context.view(-1,1).to(torch.float32)
        return self.nn(context)

    def condition(self, context):
        b = self.get_bias(context)
        return ConditionedTranslate(b)
