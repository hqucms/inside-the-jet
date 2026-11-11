# Test data

Two small files that let the fast tests check the JetClass conventions and the pretrained models without
downloading anything.

| file | content | size |
|---|---|---|
| `jetclass_sample.parquet` | 150 JetClass validation jets: rows 0-49 of `ZJetsToNuNu_120.parquet` (QCD), `WToQQ_120.parquet` (W) and `TTBar_120.parquet` (top), in this order, all 41 branches | 285 kB |
| `jetclass_sample_scores.npz` | weaver 0.5.4 softmax scores (float32, 150 x 10, JetClass class order) of the pretrained `ParT_kin` and `ParticleNet_kin` on those jets, and their `jet_pt` to check the order | 13 kB |

## Licence and attribution

The jets are from JetClass by Huilin Qu, Congqiao Li and Sitian Qian, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): doi:[10.5281/zenodo.6619768](https://doi.org/10.5281/zenodo.6619768)
(H. Qu, C. Li and S. Qian, "Particle Transformer for Jet Tagging", [arXiv:2202.03772](https://arxiv.org/abs/2202.03772)).
The jets are unmodified; only the rows were selected. The scores are computed from them with the pretrained models
of the [particle_transformer](https://github.com/jet-universe/particle_transformer) repository (MIT) at commit
`fa63c86`. See also `NOTICE.md`.

## How the files were made

`jetclass_sample.parquet` is the first 50 rows of each of the three JetClass validation files (the files of the
CERN JetClassMini mirror that `jetgen fetch jetclass --split val` downloads; MD5s in `itj.jetclass.MANIFEST`),
concatenated with pyarrow and written without any other change, so its Arrow schema (names, order, types,
`large_list<item: ... not null>`, every field `not null`) and its key-value metadata are those of the JetClass files:

```python
import pyarrow as pa
import pyarrow.parquet as pq

files = ["ZJetsToNuNu_120.parquet", "WToQQ_120.parquet", "TTBar_120.parquet"]
table = pa.concat_tables([pq.ParquetFile(f"data/jetclass/val/{f}").read_row_group(0).slice(0, 50) for f in files])
# use_compliant_nested_type=False keeps the list item name "item" of the JetClass files
pq.write_table(table, "tests/data/jetclass_sample.parquet", compression="zstd", use_compliant_nested_type=False)
```

With pyarrow 25.0.1 this gives the committed file byte for byte.

`tests/test_io.py` checks the schema and that `itj.io.load` reads it as JetClass data; `tests/test_features.py` checks
that `itj.features.inputs` gives the arrays of weaver 0.5.4 on it (`pytest -m weaver`).

`jetclass_sample_scores.npz` comes from `weaver --predict` (weaver-core 0.5.4, CPU) with the stock data config
`src/itj/configs/JetClass_kin.yaml`, the network files `example_ParticleTransformer.py` and `example_ParticleNet.py`
of particle_transformer `fa63c86` and the checkpoints `ParT_kin.pt` and `ParticleNet_kin.pt` (`jetgen fetch models`)
on `jetclass_sample.parquet`; the `scores` column of each prediction file is stored under the model name.
`tests/test_pretrained.py` compares the `itj` models with these scores.
