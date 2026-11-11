# Copied unchanged from particle_transformer (https://github.com/jet-universe/particle_transformer),
# networks/example_ParticleNet.py at commit fa63c86d594ef875c2d25c04bd5afbab5c73b611; see NOTICE.md.
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
from weaver.nn.model.ParticleNet import ParticleNet

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleNet.py
'''


class ParticleNetWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleNet(**kwargs)

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(points, features, mask)


def get_model(data_config, **kwargs):
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
    ]
    fc_params = [(256, 0.1)]

    pf_features_dims = len(data_config.input_dicts['pf_features'])
    num_classes = len(data_config.label_value)
    model = ParticleNetWrapper(
        input_dims=pf_features_dims,
        num_classes=num_classes,
        conv_params=kwargs.get('conv_params', conv_params),
        fc_params=kwargs.get('fc_params', fc_params),
        use_fusion=kwargs.get('use_fusion', False),
        use_fts_bn=kwargs.get('use_fts_bn', True),
        use_counts=kwargs.get('use_counts', True),
        for_inference=kwargs.get('for_inference', False)
    )

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
