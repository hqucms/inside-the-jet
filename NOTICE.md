# Third-party material

`itj` is released under the MIT licence (see `LICENSE`). It contains or derives from the following third-party
code, configuration files and data, each under its own licence.

| material | source | licence | used in |
|---|---|---|---|
| ParticleNet and Particle Transformer | [weaver-core](https://github.com/hqucms/weaver-core) v0.5.4 | MIT, Copyright (c) 2020 Huilin Qu | `src/itj/models/part.py`, `src/itj/models/particlenet.py` |
| JetClass data configs and example networks | [particle_transformer](https://github.com/jet-universe/particle_transformer) commit `fa63c86` | MIT, Copyright (c) 2022 Huilin Qu, Congqiao Li, Sitian Qian | `src/itj/configs/JetClass_*.yaml`, `examples/weaver/networks/*.py` |
| JetClass jets | [JetClass](https://doi.org/10.5281/zenodo.6619768) (Qu, Li and Qian) | CC BY 4.0 | `tests/data/jetclass_sample.parquet`, `tests/data/jetclass_sample_scores.npz` |

## weaver-core v0.5.4

`src/itj/models/part.py` and `src/itj/models/particlenet.py` contain the `ParticleTransformer` and `ParticleNet` modules
of `weaver/nn/model/ParticleTransformer.py` and `weaver/nn/model/ParticleNet.py` at tag v0.5.4, with the changes listed
in their module docstrings (later model versions and segmentation removed; intermediate outputs and attention weights on
request, finite gradients, pair-feature ablation, `trim=False`), wrapped in the `itj` model API. The state-dict keys are
unchanged, so the pretrained JetClass checkpoints load.

```text
MIT License

Copyright (c) 2020 Huilin Qu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## particle_transformer, commit fa63c86

`src/itj/configs/JetClass_kin.yaml`, `JetClass_kinpid.yaml` and `JetClass_full.yaml` are byte-for-byte copies of
`data/JetClass/JetClass_<set>.yaml` at commit `fa63c86d594ef875c2d25c04bd5afbab5c73b611` (the configs the pretrained
models were trained with). `examples/weaver/networks/example_ParticleTransformer.py`, `example_ParticleNet.py` and
`example_PFN.py` are copies of `networks/<file>` at the same commit with this licence added as a header. The
pretrained checkpoints that `jetgen fetch models` downloads (`models/<name>.pt` at the same commit) are not part of
this repository.

```text
MIT License

Copyright (c) 2022 Huilin Qu, Congqiao Li, Sitian Qian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## JetClass

`tests/data/jetclass_sample.parquet` holds the first 50 jets of each of the validation files
`ZJetsToNuNu_120.parquet`, `WToQQ_120.parquet` and `TTBar_120.parquet` of JetClass, unmodified (all 41 branches);
`tests/data/jetclass_sample_scores.npz` holds the scores of the pretrained models on those jets. How they were made
is in `tests/data/README.md`. The files that `jetgen fetch jetclass` downloads are not part of this repository.

JetClass by Huilin Qu, Congqiao Li and Sitian Qian, licensed under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0): doi:[10.5281/zenodo.6619768](https://doi.org/10.5281/zenodo.6619768). The data set is described in
H. Qu, C. Li and S. Qian, "Particle Transformer for Jet Tagging", ICML 2022,
[arXiv:2202.03772](https://arxiv.org/abs/2202.03772).
