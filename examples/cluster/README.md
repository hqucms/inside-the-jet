# Generating data and training on a cluster

Large toy data sets are generated as **shards**: one job per shard, each with its own seed and its own range of jet
ids, combined at the end with `jetgen merge`. The shards run in parallel, each is reproducible from its seed, and a
failed job is simply run again. The last section is a GPU job that trains the full ParT with weaver. The snippets
below are generic: **adapt them to your cluster** (partitions, accounts, modules, file systems and limits differ from
one cluster to the next).

## Before the first job

1. Install the package in a virtual environment on the cluster, as in the [README](../../README.md#installation) and
   [docs/install.md](../../docs/install.md): the CPU build of PyTorch for CPU nodes, a CUDA build (and the `weaver`
   extra) for GPU nodes. The jobs must be able to read the environment.
2. Put the data on a work or scratch file system with enough space, not in a small home directory (about 2.5 kB per
   jet with the truth, 2.1 kB without, `--no-truth`).
3. Try one small shard in an interactive session (not on the login node), from the repository root:

   ```bash
   jetgen generate -n 1000 --seed 1000 --first-event-id 0 -o data/shards/test_shard0000.parquet
   jetgen info data/shards/test_shard0000.parquet --provenance
   ```

## One shard

Shard `i` of a data set gets the seed `BASE_SEED + i` and the jet ids from `i * JETS` on:

```bash
#!/bin/bash
# run_shard.sh I [extra jetgen options]: generate shard I (adapt to your cluster)
set -euo pipefail
I=$1; shift
JETS=${JETS:-100000}                # jets per shard
BASE_SEED=${BASE_SEED:-1000}        # use separate seed ranges for independent data sets
OUT_DIR=${OUT_DIR:-data/shards}
OUT=$(printf "%s/shard%04d.parquet" "$OUT_DIR" "$I")
[ -f "$OUT" ] && { echo "exists: $OUT"; exit 0; }   # resubmitting only makes the missing shards
mkdir -p "$OUT_DIR"
jetgen generate -n "$JETS" --seed $((BASE_SEED + I)) --first-event-id $((I * JETS)) \
  --workers "${WORKERS:-4}" --no-progress -o "$OUT" "$@"
```

Every option of `jetgen generate` can follow the shard index, e.g. `bash run_shard.sh 3 --preset hadrons` or
`bash run_shard.sh 3 --process q,g --no-ue`. A file is written as `<name>.part` and renamed when it is complete, so a
job that is killed never leaves a truncated shard under its final name. At about 850 jets per second and core at
`detector` (a 2016 CPU), a shard of 100 000 jets takes about a minute on 4 cores.

## SLURM array (adapt to your cluster)

```bash
#!/bin/bash
#SBATCH --job-name=jetgen
#SBATCH --array=0-9                 # ten shards; --array=0-99%10 runs at most ten at a time
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
##SBATCH --partition=...            # your partition, account, ...
source .venv/bin/activate
WORKERS=$SLURM_CPUS_PER_TASK bash run_shard.sh "$SLURM_ARRAY_TASK_ID"
```

Submit from the repository root with `mkdir -p logs && sbatch slurm_array.sh`.

## HTCondor (adapt to your cluster)

```text
# htcondor.sub
executable     = /bin/bash
arguments      = -c "source .venv/bin/activate && WORKERS=4 bash run_shard.sh $(Process)"
initialdir     = /path/to/inside-the-jet
request_cpus   = 4
request_memory = 4 GB
output         = logs/jetgen_$(Cluster)_$(Process).out
error          = logs/jetgen_$(Cluster)_$(Process).err
log            = logs/jetgen_$(Cluster).log
queue 10
```

Submit with `mkdir -p logs && condor_submit htcondor.sub`; `max_materialize` limits the jobs in the queue if your
cluster asks for it.

## Combining the shards

```bash
ls data/shards/shard*.parquet | wc -l                          # all shards there?
jetgen merge data/shards/shard*.parquet -o data/toy_n1M.parquet
jetgen info data/toy_n1M.parquet
```

`jetgen merge` refuses shards with the same seed (their jets would repeat the same random numbers) and shards whose
`truth_event_id` ranges overlap (`--renumber` numbers the jets 0 to N - 1 instead); `--shuffle --seed S` mixes the
jets of all shards. Shards with different configurations merge too: the configuration of the result is then `None`
and every input is listed in `provenance["merged"]`. It does not know how many shards you meant to make, so count them
first. For weaver, split the merged file (or every shard) with `jetgen split ... --out-dir DIR`
([examples/weaver](../weaver/README.md)).

## A GPU training with weaver

One SLURM job that trains the full ParT on the three-class JetClass subset (the command of
[examples/weaver](../weaver/README.md#gpu-optional-full-trainings-on-the-three-class-subset)). It needs the `weaver`
extra and a CUDA build of PyTorch in the environment, the training files (`jetgen fetch jetclass --split train`) and
the split validation files (`python examples/make_task_data.py --week 2`), fetched beforehand on a node with internet
access.

```bash
#!/bin/bash
#SBATCH --job-name=weaver-part
#SBATCH --gres=gpu:1                # one GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
##SBATCH --partition=...            # your GPU partition, account, ...
set -euo pipefail
source .venv/bin/activate
nvidia-smi
weaver --data-train 'QCD:data/jetclass/train/ZJetsToNuNu_000.parquet' 'Wqq:data/jetclass/train/WToQQ_000.parquet' 'Tbqq:data/jetclass/train/TTBar_000.parquet' \
  --data-val 'QCD:data/jetclass/split/val/ZJetsToNuNu_120.parquet' 'Wqq:data/jetclass/split/val/WToQQ_120.parquet' 'Tbqq:data/jetclass/split/val/TTBar_120.parquet' \
  --data-test 'QCD:data/jetclass/split/test/ZJetsToNuNu_120.parquet' 'Wqq:data/jetclass/split/test/WToQQ_120.parquet' 'Tbqq:data/jetclass/split/test/TTBar_120.parquet' \
  --data-config src/itj/configs/jets3_kin.yaml --network-config examples/weaver/networks/example_ParticleTransformer.py \
  --model-prefix runs/weaver_part/net --gpus 0 --num-workers 1 --batch-size 512 --start-lr 1e-3 --num-epochs 20 \
  --steps-per-epoch 585 --steps-per-epoch-val 58 --optimizer ranger --use-amp --in-memory --fetch-step 1 \
  --predict-output pred.parquet --log-file logs/weaver_part_${SLURM_JOB_ID}.log
```

Submit with `mkdir -p logs && sbatch slurm_weaver.sh`. The checkpoints and the predictions go to `runs/weaver_part/`;
`itj.models.load_model("runs/weaver_part/net_best_epoch_state.pt", name="part")` loads the result for the analyses of
T4 and T5.

## Good practice

- Measure the time and memory of one shard first (task T0.3) and size the jobs from that.
- Do not run simulations or trainings on the login node; use jobs or an interactive session.
- Use separate seed ranges for data sets that must be independent (e.g. `BASE_SEED=1000` for training and
  `BASE_SEED=2000` for testing, with fewer than 1000 shards each). The same seed gives the same jets, and the same
  seed with another preset gives the same hard processes (paired samples).
- Keep the logs (`logs/` is ignored by git) until the data set is checked.
