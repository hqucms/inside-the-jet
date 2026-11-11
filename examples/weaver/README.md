# weaver on the toy and on JetClass

[weaver](https://github.com/hqucms/weaver-core) is the training framework of the Particle Transformer (ParT) and of the
JetClass paper. This page has the commands of task T4.6: a short CPU run on the toy and on JetClass, a prediction
with the pretrained `ParT_kin` that `itj` must reproduce, and the optional GPU trainings. weaver in one page (the
configuration files and the options) is in [docs/jetclass.md](../../docs/jetclass.md#weaver-in-one-page).

The network configurations in [networks/](networks) are the example wrappers of the ParT repository
([jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer), commit `fa63c86`, MIT
licence, see [NOTICE.md](../../NOTICE.md)): `example_ParticleTransformer.py`, `example_ParticleNet.py` and
`example_PFN.py`. The data configurations are those of `itj` (`jetgen configs` prints their paths):

| file | classes | inputs |
|---|---|---|
| `src/itj/configs/jets3_kin.yaml` | QCD, W, top (`label_QCD`, `label_Wqq`, `label_Tbqq`) | the 7 kinematic features of ParT |
| `src/itj/configs/jets3_kinpid.yaml`, `jets3_full.yaml` | the same | plus the particle types (13), plus the impact parameters (17) |
| `src/itj/configs/JetClass_kin.yaml`, `_kinpid`, `_full` | the ten JetClass classes | the stock configurations of the ParT repository |

## Before the first run

1. Install the extra (Python 3.10 to 3.13; torch first, see [docs/install.md](../../docs/install.md#optional-weaver)):
   `python -m pip install -e ".[weaver]"`, then check with `python -c "import weaver"`.
2. Make the data of week 2: `python examples/make_task_data.py --week 2`. The commands below need the split files
   `data/t2_split/` and `data/jetclass/split/` (`jetgen split`) and the checkpoint `models/pretrained/ParT_kin.pt`.
3. Run every command from the repository root. They are single lines; the Command Prompt versions (Windows) use no
   single quotes and `--gpus ""`.

**Always use the split files.** weaver reads all files of a fetch, concatenates them and only then applies a
`selection`, so an index-based selection in the YAML would mix the files and train on the test jets of every file but
the first. `jetgen split` writes the `itj` split (test = the first 20 % of each file, val = the next 10 %, train = the
rest) as files, and each file becomes its own named group (`name:path`). Then the jets that weaver tests on are
exactly those of `itj.io.load(path, part="test")`.

## A short CPU run on the toy

A ParT of the size of `itj`'s `partsmall` (3 particle-attention blocks, 1 class-attention block, embedding
(64, 256, 64), pair embedding (32, 32, 32), 4 heads) trained for 30 steps of 256 jets, then tested on the 30 000 test
jets. About 35 s on a 2016 six-core CPU.

```bash
weaver --data-train 't2:data/t2_split/train/t2_detector.parquet' --data-val data/t2_split/val/t2_detector.parquet --data-test 't2:data/t2_split/test/t2_detector.parquet' --data-config src/itj/configs/jets3_kin.yaml --network-config examples/weaver/networks/example_ParticleTransformer.py -o num_layers 3 -o num_cls_layers 1 -o embed_dims "[64,256,64]" -o pair_embed_dims "[32,32,32]" -o num_heads 4 --model-prefix runs/weaver_partsmall/net --gpus '' --num-workers 0 --batch-size 256 --start-lr 1e-3 --num-epochs 1 --steps-per-epoch 30 --in-memory --fetch-step 1 --predict-output pred.parquet
```

```bat
weaver --data-train t2:data/t2_split/train/t2_detector.parquet --data-val data/t2_split/val/t2_detector.parquet --data-test t2:data/t2_split/test/t2_detector.parquet --data-config src/itj/configs/jets3_kin.yaml --network-config examples/weaver/networks/example_ParticleTransformer.py -o num_layers 3 -o num_cls_layers 1 -o embed_dims "[64,256,64]" -o pair_embed_dims "[32,32,32]" -o num_heads 4 --model-prefix runs/weaver_partsmall/net --gpus "" --num-workers 0 --batch-size 256 --start-lr 1e-3 --num-epochs 1 --steps-per-epoch 30 --in-memory --fetch-step 1 --predict-output pred.parquet
```

What it writes:

| file | content |
|---|---|
| `runs/weaver_partsmall/net_epoch-0_state.pt`, `net_best_epoch_state.pt` | the state dict of the network wrapper (keys `mod.*`) |
| `runs/weaver_partsmall/predict_output/pred_t2.parquet` | `scores` (softmax, $N \times 3$), `_label_` and the observers `jet_pt` ... `jet_tau4`, one row per test jet in file order |
| `src/itj/configs/jets3_kin.<hash>.auto.yaml` | the data configuration as weaver completed it (ignored by git) |

After 30 steps the tagger is far from trained: the run checks the chain, not the physics. Load the checkpoint into
`itj` and compare the scores:

```python
import numpy as np
import pyarrow.parquet as pq

from itj.io import load
from itj.models import load_model
from itj.train import predict

model = load_model("runs/weaver_partsmall/net_best_epoch_state.pt", name="partsmall")  # a bare weaver state dict
test = load("data/t2_split/test/t2_detector.parquet")
s_itj = predict(model, test)
s_weaver = np.asarray(
    pq.read_table("runs/weaver_partsmall/predict_output/pred_t2.parquet").column("scores").to_pylist()
)
print("max |delta s| =", np.abs(s_itj - s_weaver).max())
```

## The same on JetClass

The identical command on the split JetClass files, with one named group per class (about 80 s: the JetClass files are
larger):

```bash
weaver --data-train 'QCD:data/jetclass/split/train/ZJetsToNuNu_120.parquet' 'Wqq:data/jetclass/split/train/WToQQ_120.parquet' 'Tbqq:data/jetclass/split/train/TTBar_120.parquet' --data-val 'QCD:data/jetclass/split/val/ZJetsToNuNu_120.parquet' 'Wqq:data/jetclass/split/val/WToQQ_120.parquet' 'Tbqq:data/jetclass/split/val/TTBar_120.parquet' --data-test 'QCD:data/jetclass/split/test/ZJetsToNuNu_120.parquet' 'Wqq:data/jetclass/split/test/WToQQ_120.parquet' 'Tbqq:data/jetclass/split/test/TTBar_120.parquet' --data-config src/itj/configs/jets3_kin.yaml --network-config examples/weaver/networks/example_ParticleTransformer.py -o num_layers 3 -o num_cls_layers 1 -o embed_dims "[64,256,64]" -o pair_embed_dims "[32,32,32]" -o num_heads 4 --model-prefix runs/weaver_partsmall_jc/net --gpus '' --num-workers 0 --batch-size 256 --start-lr 1e-3 --num-epochs 1 --steps-per-epoch 30 --in-memory --fetch-step 1 --predict-output pred.parquet
```

```bat
weaver --data-train QCD:data/jetclass/split/train/ZJetsToNuNu_120.parquet Wqq:data/jetclass/split/train/WToQQ_120.parquet Tbqq:data/jetclass/split/train/TTBar_120.parquet --data-val QCD:data/jetclass/split/val/ZJetsToNuNu_120.parquet Wqq:data/jetclass/split/val/WToQQ_120.parquet Tbqq:data/jetclass/split/val/TTBar_120.parquet --data-test QCD:data/jetclass/split/test/ZJetsToNuNu_120.parquet Wqq:data/jetclass/split/test/WToQQ_120.parquet Tbqq:data/jetclass/split/test/TTBar_120.parquet --data-config src/itj/configs/jets3_kin.yaml --network-config examples/weaver/networks/example_ParticleTransformer.py -o num_layers 3 -o num_cls_layers 1 -o embed_dims "[64,256,64]" -o pair_embed_dims "[32,32,32]" -o num_heads 4 --model-prefix runs/weaver_partsmall_jc/net --gpus "" --num-workers 0 --batch-size 256 --start-lr 1e-3 --num-epochs 1 --steps-per-epoch 30 --in-memory --fetch-step 1 --predict-output pred.parquet
```

The predictions go to `runs/weaver_partsmall_jc/predict_output/pred_QCD.parquet`, `pred_Wqq.parquet` and
`pred_Tbqq.parquet`, one file per named test group.

## The pretrained ParT_kin

The model of the JetClass paper with the kinematic inputs, ten classes, on the first 5 % of each split test file
(1 000 jets per class); about 20 s on a CPU, start-up included.

```bash
weaver --predict --network-config examples/weaver/networks/example_ParticleTransformer.py --model-prefix models/pretrained/ParT_kin.pt --data-config src/itj/configs/JetClass_kin.yaml --data-test 'QCD:data/jetclass/split/test/ZJetsToNuNu_120.parquet' 'Wqq:data/jetclass/split/test/WToQQ_120.parquet' 'Tbqq:data/jetclass/split/test/TTBar_120.parquet' --data-fraction 0.05 --fetch-step 1 --gpus '' --num-workers 0 --batch-size 256 --predict-output pred_ParT_kin.parquet
```

```bat
weaver --predict --network-config examples/weaver/networks/example_ParticleTransformer.py --model-prefix models/pretrained/ParT_kin.pt --data-config src/itj/configs/JetClass_kin.yaml --data-test QCD:data/jetclass/split/test/ZJetsToNuNu_120.parquet Wqq:data/jetclass/split/test/WToQQ_120.parquet Tbqq:data/jetclass/split/test/TTBar_120.parquet --data-fraction 0.05 --fetch-step 1 --gpus "" --num-workers 0 --batch-size 256 --predict-output pred_ParT_kin.parquet
```

A bare `--predict-output` name goes to `predict_output/` next to the model: `models/pretrained/predict_output/
pred_ParT_kin_QCD.parquet`, `_Wqq` and `_Tbqq`, each with `scores` ($N \times 10$, in the JetClass class order QCD 0,
..., Wqq 7, Tbqq 8), `_label_` and the observers. In test mode weaver reads the first `--data-fraction` of every file
in order, so the same jets are `load(path, max_jets=1000)` of each split test file:

```python
import numpy as np
import pyarrow.parquet as pq

from itj.io import load
from itj.models import load_pretrained
from itj.train import predict

model = load_pretrained("ParT_kin")
for name, stem in (("QCD", "ZJetsToNuNu_120"), ("Wqq", "WToQQ_120"), ("Tbqq", "TTBar_120")):
    jets = load(f"data/jetclass/split/test/{stem}.parquet", max_jets=1000)
    table = pq.read_table(f"models/pretrained/predict_output/pred_ParT_kin_{name}.parquet")
    s_weaver = np.asarray(table.column("scores").to_pylist())
    print(name, "max |delta s| =", np.abs(predict(model, jets) - s_weaver).max())
```

The differences should be at the level of float32 rounding; anything larger means that the inputs differ (a feature
set, the padding or the order of the jets). `weaver` also prints the confusion matrix and the accuracy of every test
group; with three of the ten classes, the accuracy is that of the argmax over ten outputs (task T4.3 explains why it is
lower than the three-class accuracy).

## GPU (optional): full trainings on the three-class subset

On a GPU (a cluster job, see [examples/cluster](../cluster/README.md#a-gpu-training-with-weaver), or Colab), train the
full ParT (2.1 M parameters) or ParticleNet on QCD, W and top. With the training files of JetClass
(`jetgen fetch jetclass --split train`, 100 000 jets per class, 374 MB) and the split validation files:

```bash
weaver --data-train 'QCD:data/jetclass/train/ZJetsToNuNu_000.parquet' 'Wqq:data/jetclass/train/WToQQ_000.parquet' 'Tbqq:data/jetclass/train/TTBar_000.parquet' --data-val 'QCD:data/jetclass/split/val/ZJetsToNuNu_120.parquet' 'Wqq:data/jetclass/split/val/WToQQ_120.parquet' 'Tbqq:data/jetclass/split/val/TTBar_120.parquet' --data-test 'QCD:data/jetclass/split/test/ZJetsToNuNu_120.parquet' 'Wqq:data/jetclass/split/test/WToQQ_120.parquet' 'Tbqq:data/jetclass/split/test/TTBar_120.parquet' --data-config src/itj/configs/jets3_kin.yaml --network-config examples/weaver/networks/example_ParticleTransformer.py --model-prefix runs/weaver_part/net --gpus 0 --num-workers 1 --batch-size 512 --start-lr 1e-3 --num-epochs 20 --optimizer ranger --use-amp --in-memory --fetch-step 1 --predict-output pred.parquet
```

For ParticleNet replace the network configuration by `examples/weaver/networks/example_ParticleNet.py` and the model
prefix by `runs/weaver_pn/net`; `jets3_kinpid.yaml` and `jets3_full.yaml` add the particle types and the impact
parameters. `--use-amp` (mixed precision, as in the ParT repository) is needed for a batch of 512 on an 11 GB GPU:
without it the full ParT runs out of memory on a GTX 1080 Ti. With 300 000 training jets an epoch of ParT then takes
about 6-7 minutes on a GTX 1080 Ti (850 jets/s); newer GPUs with tensor cores are several times faster.
`load_model("runs/weaver_part/net_best_epoch_state.pt", name="part")` loads the result into `itj` (pass `feature_set`
when the inputs are not `kin`).

**Colab.** On a GPU runtime, one cell installs everything, fetches and splits the JetClass files of week 2 and
fetches the training files:

```bash
!git clone https://github.com/hqucms/inside-the-jet.git && cd inside-the-jet && pip install -q -e ".[analysis,weaver]" && python examples/make_task_data.py --week 2 --only jetclass && jetgen fetch jetclass --split train
```

then `%cd inside-the-jet` and run the commands above with `--gpus 0`.

## Expected times and outputs

| run | where | time | output |
|---|---|---|---|
| toy, 30 steps | CPU, 6 threads | about 35 s | `runs/weaver_partsmall/`, `pred_t2.parquet` (30 000 jets) |
| JetClass, 30 steps | CPU | about 80 s | `runs/weaver_partsmall_jc/`, three prediction files (20 000 jets each) |
| pretrained `ParT_kin`, 3 x 1 000 jets | CPU | about 20 s | `models/pretrained/predict_output/pred_ParT_kin_*.parquet` |
| full ParT, 300 000 jets x 20 epochs | GPU | about 2-2.5 h on a GTX 1080 Ti | `runs/weaver_part/` |

## Common problems

- **"CUDA" or "no GPU" errors:** pass `--gpus ''` (bash) or `--gpus ""` (Command Prompt); the default `'0'` asks for a
  GPU.
- **`--num-workers`** must not exceed the number of files in a named group; `0` loads in the main process and is not
  slower on a CPU (and avoids problems with the spawned workers on Windows and macOS).
- **Apple Silicon:** weaver selects MPS by itself when it is asked for the CPU; set
  `PYTORCH_ENABLE_MPS_FALLBACK=1` (or `export PYTORCH_ENABLE_MPS_FALLBACK=1` once per terminal).
- **Python 3.14 or an Intel Mac:** the extra does not install; do T4.6(a) with `itj` alone and say so.
- **"could not broadcast input array from shape ...":** weaver 0.5.4 fails when it reads only part of a Parquet
  file, which the default `--fetch-step 0.01` does; every command here therefore has `--fetch-step 1` (each file is
  read in one piece).
- **Lists in `-o`:** the values are read as Python literals, so quote lists: `-o embed_dims "[64,256,64]"`.
