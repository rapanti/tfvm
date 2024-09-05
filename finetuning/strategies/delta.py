"""
Modified from https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/regularization/delta.py
"""

import functools

import torch
import torch.nn as nn


class L2Regularization(nn.Module):
    r"""The L2 regularization of parameters :math:`w` can be described as:

    .. math::
        {\Omega} (w) = \dfrac{1}{2}  \Vert w\Vert_2^2 ,

    Args:
        model (torch.nn.Module):  The model to apply L2 penalty.

    Shape:
        - Output: scalar.
    """

    def __init__(self, model: nn.Module):
        super(L2Regularization, self).__init__()
        self.model = model

    def forward(self):
        output = 0.0
        for param in self.model.parameters():
            output += 0.5 * torch.norm(param) ** 2
        return output


class SPRegularization(nn.Module):
    r"""The SP (Starting Point) regularization from `Explicit inductive bias for transfer learning with convolutional networks
    (ICML 2018) <https://arxiv.org/abs/1802.01483>`_

    The SP regularization of parameters :math:`w` can be described as:

    .. math::
        {\Omega} (w) = \dfrac{1}{2}  \Vert w-w_0\Vert_2^2 ,

    where :math:`w_0` is the parameter vector of the model pretrained on the source problem, acting as the starting point (SP) in fine-tuning.


    Args:
        source_model (torch.nn.Module):  The source (starting point) model.
        target_model (torch.nn.Module):  The target (fine-tuning) model.

    Shape:
        - Output: scalar.
    """

    def __init__(self, source_model: nn.Module, target_model: nn.Module, weight: float = 0.5):
        super().__init__()
        assert weight > 0.0
        self.weight = weight
        self.target_model = target_model
        self.source_model = source_model

    def forward(self, **kwargs):
        output = 0.0
        source_weight = dict(self.source_model.named_parameters())
        for name, param in self.target_model.named_parameters():
            if param.requires_grad:
                output += self.weight * torch.norm(param - source_weight[name]) ** 2

        return output


class BehavioralRegularization(nn.Module):
    r"""The behavioral regularization from `DELTA:DEep Learning Transfer using Feature Map with Attention
    for convolutional networks (ICLR 2019) <https://openreview.net/pdf?id=rkgbwsAcYm>`_

    It can be described as:

    .. math::
        {\Omega} (w) = \sum_{j=1}^{N}   \Vert FM_j(w, \boldsymbol x)-FM_j(w^0, \boldsymbol x)\Vert_2^2 ,

    where :math:`w^0` is the parameter vector of the model pretrained on the source problem, acting as the starting point (SP) in fine-tuning,
    :math:`FM_j(w, \boldsymbol x)` is feature maps generated from the :math:`j`-th layer of the model parameterized with :math:`w`, given the input :math:`\boldsymbol x`.


    Inputs:
        layer_outputs_source (OrderedDict):  The dictionary for source model, where the keys are layer names and the values are feature maps correspondingly.

        layer_outputs_target (OrderedDict):  The dictionary for target model, where the keys are layer names and the values are feature maps correspondingly.

    Shape:
        - Output: scalar.

    """

    def __init__(self, source_model: nn.Module, weight: float = 0.01):
        super().__init__()
        self.source_model = source_model
        self.weight = weight

    def forward(self, x, feature, **kwargs):
        # output = 0.0
        # for fm_src, fm_tgt in zip(layer_outputs_source.values(), layer_outputs_target.values()):
        #     output += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2)
        source_feature = self.source_model.forward_features(x).detach()
        output = self.weight * torch.norm(feature - source_feature) ** 2
        return output


class AttentionBehavioralRegularization(nn.Module):
    r"""The behavioral regularization with attention from `DELTA:DEep Learning Transfer using Feature Map with Attention
    for convolutional networks (ICLR 2019) <https://openreview.net/pdf?id=rkgbwsAcYm>`_

    It can be described as:

    .. math::
        {\Omega} (w) = \sum_{j=1}^{N}  W_j(w) \Vert FM_j(w, \boldsymbol x)-FM_j(w^0, \boldsymbol x)\Vert_2^2 ,

    where
    :math:`w^0` is the parameter vector of the model pretrained on the source problem, acting as the starting point (SP) in fine-tuning.
    :math:`FM_j(w, \boldsymbol x)` is feature maps generated from the :math:`j`-th layer of the model parameterized with :math:`w`, given the input :math:`\boldsymbol x`.
    :math:`W_j(w)` is the channel attention of the :math:`j`-th layer of the model parameterized with :math:`w`.

    Args:
        channel_attention (list): The channel attentions of feature maps generated by each selected layer. For the layer with C channels, the channel attention is a tensor of shape [C].

    Inputs:
        layer_outputs_source (OrderedDict):  The dictionary for source model, where the keys are layer names and the values are feature maps correspondingly.

        layer_outputs_target (OrderedDict):  The dictionary for target model, where the keys are layer names and the values are feature maps correspondingly.

    Shape:
        - Output: scalar.

    """

    def __init__(self, channel_attention):
        super().__init__()
        self.channel_attention = channel_attention

    def forward(self, layer_outputs_source, layer_outputs_target):
        output = 0.0
        for i, (fm_src, fm_tgt) in enumerate(
            zip(layer_outputs_source.values(), layer_outputs_target.values())
        ):
            b, c, h, w = fm_src.shape
            fm_src = fm_src.reshape(b, c, h * w)
            fm_tgt = fm_tgt.reshape(b, c, h * w)

            distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
            distance = c * torch.mul(self.channel_attention[i], distance**2) / (h * w)
            output += 0.5 * torch.sum(distance)

        return output


def get_attribute(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))
