# Copied unchanged from particle_transformer (https://github.com/jet-universe/particle_transformer),
# networks/example_PFN.py at commit fa63c86d594ef875c2d25c04bd5afbab5c73b611; see NOTICE.md.
#
# MIT License
#
# Copyright (c) 2022 Huilin Qu, Congqiao Li, Sitian Qian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn


class ParticleFlowNetwork(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, input_dims, num_classes,
                 Phi_sizes=(100, 100, 128),
                 F_sizes=(100, 100, 100),
                 use_bn=True,
                 for_inference=False,
                 **kwargs):

        super(ParticleFlowNetwork, self).__init__(**kwargs)
        # input bn
        self.input_bn = nn.BatchNorm1d(input_dims) if use_bn else nn.Identity()
        # per-particle functions
        phi_layers = []
        for i in range(len(Phi_sizes)):
            phi_layers.append(nn.Sequential(
                nn.Conv1d(input_dims if i == 0 else Phi_sizes[i - 1], Phi_sizes[i], kernel_size=1),
                nn.BatchNorm1d(Phi_sizes[i]) if use_bn else nn.Identity(),
                nn.ReLU())
            )
        self.phi = nn.Sequential(*phi_layers)
        # global functions
        f_layers = []
        for i in range(len(F_sizes)):
            f_layers.append(nn.Sequential(
                nn.Linear(Phi_sizes[-1] if i == 0 else F_sizes[i - 1], F_sizes[i]),
                nn.ReLU())
            )
        f_layers.append(nn.Linear(F_sizes[-1], num_classes))
        if for_inference:
            f_layers.append(nn.Softmax(dim=1))
        self.fc = nn.Sequential(*f_layers)

    def forward(self, points, features, lorentz_vectors, mask):
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        x = self.input_bn(features)
        x = self.phi(x)
        if mask is not None:
            x = x * mask.bool().float()
        x = x.sum(-1)
        return self.fc(x)


def get_model(data_config, **kwargs):
    Phi_sizes = (128, 128, 128)
    F_sizes = (128, 128, 128)
    input_dims = len(data_config.input_dicts['pf_features'])
    num_classes = len(data_config.label_value)
    model = ParticleFlowNetwork(input_dims, num_classes, Phi_sizes=Phi_sizes,
                                F_sizes=F_sizes, use_bn=kwargs.get('use_bn', False))

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
