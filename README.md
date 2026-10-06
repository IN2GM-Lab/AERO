# AERO — A Pluggable Congestion Control for QUIC

AERO is a learning-based, implementation-agnostic congestion controller for QUIC. A
PPO-trained actor observes four normalized network signals — throughput, delay, jitter and
loss — and emits a single scalar `alpha` that scales the congestion window
(`cwnd <- (base ** alpha) * cwnd`). Because the policy only consumes transport statistics and
only returns a window multiplier, it drops into any QUIC stack that can expose those signals;
`aero_aioquic.py` is a working integration with [aioquic](https://github.com/aiortc/aioquic).

This repository contains the full pipeline: dataset preparation, training, offline evaluation,
the final trained model, and the production integration.

**Project website:** `index.html` is a standalone GitHub Pages site presenting the motivation,
design and measured results. Enable it under *Settings → Pages → Deploy from a branch → / (root)*.
Both papers are in [`papers/`](papers/).

## Citation

If you use our work, please cite:

```bibtex
@inproceedings{sidhu2025aero,
  title={Aero: A pluggable congestion control for QUIC},
  author={Sidhu, Jashanjot Singh and Bentaleb, Abdelhak},
  booktitle={Proceedings of the 35th Workshop on Network and Operating System Support for Digital Audio and Video},
  pages={29--35},
  year={2025}
}
```

```bibtex
@article{sidhu2026towards,
  title={Towards a Pluggable and Implementation-Agnostic Learning-Based Congestion Control for QUIC},
  author={Sidhu, Jashanjot Singh and Bentaleb, Abdelhak},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  year={2026},
  publisher={ACM New York, NY}
}
```

## Contents

| File | Role |
|---|---|
| `get_data.py` | Stage 0 — walks the dataset and drives `parse_json1.py` |
| `parse_json1.py` | Stage 0 — turns each JSON observation into one row of `drl_cc.txt` |
| `train_aero.py` | Stage 1 — PPO training loop; reads `drl_cc.txt`, writes checkpoints |
| `ppo6.py` | The actor/critic networks. **This is the network AERO ships with** |
| `parse_json.py` | `get_data()` helper imported by `train_aero.py` |
| `test_aero.py` | Stage 2 — offline evaluation of a trained model; mirrors the production decision path |
| `aero_aioquic.py` | Reference integration: AERO as an aioquic congestion controller |
| `model/abr_ppo_38000.model` | **The final trained actor** (epoch 38000) used in the papers |
| `Results/sim/ppo4/` | Where `train_aero.py` writes logs and checkpoints |

## Requirements

Tested on Python 3.10.12 / Linux.

```bash
pip install -r requirements.txt
```

| Package | Tested version | Needed by |
|---|---|---|
| `torch` | 2.1.1 | `ppo6.py`, `train_aero.py`, `test_aero.py`, `aero_aioquic.py` |
| `numpy` | 1.26.1 | all data and training scripts |
| `tqdm` | 4.66.1 | imported by `train_aero.py` |
| `matplotlib` | any | imported by `aero_aioquic.py` |
| `aioquic` | — | **only** for the deployment step, not for training |

CUDA is optional. Every script falls back to CPU automatically
(`torch.cuda.is_available()`), and the shipped model loads with `map_location='cpu'`.

---

## Step 1 — Download the dataset

AERO trains on the emulated dataset from Microsoft's
[RL4BandwidthEstimationChallenge](https://github.com/microsoft/RL4BandwidthEstimationChallenge):

```bash
wget https://raw.githubusercontent.com/microsoft/RL4BandwidthEstimationChallenge/main/download-emulated-dataset.sh
chmod +x download-emulated-dataset.sh
./download-emulated-dataset.sh
```

That produces chunk directories of JSON traces, e.g.:

```
<dataset-root>/
└── Training/
    ├── emulated_dataset_chunk_0/*.json
    ├── emulated_dataset_chunk_1/*.json
    └── emulated_dataset_chunk_2/*.json
```

`get_data.py` walks the tree recursively looking for `*.json`, so the exact nesting does not
matter — only the root path you point it at.

Each JSON provides `observations` (a list of 150-value vectors: 15 features x 10 time slots)
and `true_capacity`. `parse_json1.py` reads features at slot offsets
`0` recv_rate, `4` delay, `5` min_seen_delay, `9` jitter, `11` lost_packets,
`12` video, `13` audio, `14` probing.

## Step 2 — Point the scripts at your paths

The scripts carry hardcoded paths from the original development machine. **Edit these before
running anything:**

| File:line | Current value | Set it to |
|---|---|---|
| `get_data.py:6` | `root_folder = '/home/ubuntu/Json/'` | **Absolute path to your dataset root** from Step 1 |
| `parse_json1.py:13` | `output_file_path = 'drl_cc.txt'` | Trace filename; resolved **relative to your working directory** |
| `train_aero.py:113` | `open('drl_cc.txt', 'r')` | Must match `parse_json1.py:13` |
| `train_aero.py:52` | `LOG_FILE = './Results/sim/ppo4/log'` | Log prefix — the directory must already exist |
| `train_aero.py:55` | `LOG_FILE_VALID = './Results/sim/ppo4/test_results/log_valid_ppo'` | Checkpoint directory — must already exist |
| `aero_aioquic.py:94` | `test_model='/aioquic/src/aioquic/quic/congestion/abr_ppo_38000.model'` | Path to the `.model` file **inside your aioquic deployment** (see Step 5) |
| `aero_aioquic.py:16` | `from aioquic.quic.congestion.ppo5 import Actor, Critic` | Must match the filename you copy the network in as (see Step 5) |

`test_aero.py` needs no editing — its defaults (`test_aero.py:45-47`) are all overridable on the
command line.

Neither of these directories is created automatically by `train_aero.py`, and it will crash on
startup if they are missing. They ship with the repo, but if you change the paths:

```bash
mkdir -p Results/sim/ppo4/test_results
```

**Lines you can safely ignore** — these look like paths but are dead code:

- `train_aero.py:53` — `root_folder` is declared and never used (training reads `drl_cc.txt`).
- `parse_json.py:82`, `parse_json1.py:88` — inside a `'''...'''` comment block.
- `aero_aioquic.py:52-59` — unused log-path constants left over from training.

## Step 3 — Generate the training trace

```bash
python3 get_data.py
```

This flattens every JSON observation into one whitespace-separated row of `drl_cc.txt`:

```
throughput  cwnd  max_cwnd  delay  min_delay  jitter  lost_packets  video  audio  probing
```

> **Note:** `parse_json1.py` opens the output in **append** mode. Delete `drl_cc.txt` before
> regenerating, or rows from the previous run will still be in the file.

## Step 4 — Train

```bash
python3 train_aero.py
```

PPO with an RMSprop actor/critic, 29 explorations x 25 steps per epoch, batch size 128,
reward `throughput - LOSS_PENALTY*loss - DELAY_PENALTY*delay - JITTER_PENALTY*jitter` over
normalized signals, plus a shaping term that rewards predicted windows approaching — but not
exceeding — `max_cwnd`. `LOSS_PENALTY` and `DELAY_PENALTY` adapt per step to whether the flow
is video-, audio- or probing-dominated.

Outputs, every `UPDATE_INTERVAL = 500` epochs:

```
Results/sim/ppo4/log_record                                 per-step trace
Results/sim/ppo4/log_central                                epoch losses
Results/sim/ppo4/test_results/log_valid_ppo/abr_ppo_<epoch>.model         actor
Results/sim/ppo4/test_results/log_valid_ppo/abr_critic_ppo_<epoch>.model  critic
```

> **Note:** training runs in an unbounded `while True` loop — there is no stopping criterion.
> Stop it yourself (Ctrl-C) once the logged losses plateau, then pick a checkpoint.
> The model shipped in `model/` is epoch 38000.

## Step 5 — Evaluate a trained model

`test_aero.py` replays a recorded trace through the policy using **the same decision path as
production** — the same normalization, the same reward, the same `alpha` sign-flip heuristic
and the same `(1.89 ** alpha) * cwnd` window update as `aero_aioquic.py`. It is evaluation
only: no optimizers, no backward pass, no checkpoint writes, and it terminates.

```bash
# evaluate the shipped final model on the full trace
python3 test_aero.py

# or point it anywhere
python3 test_aero.py --model model/abr_ppo_38000.model \
                     --data drl_cc.txt \
                     --steps 5000 \
                     --log Results/sim/aero_test/log_test_record
```

It prints reward statistics, the `alpha` distribution, the realized window multipliers, how
often the window exceeded the trace's `max_cwnd`, and how often the reward heuristic overrode
the model's sign — then writes a per-step log. Read this script first if you want to understand
how to drive the model from your own stack: it is the minimal correct way to call it.

## Step 6 — Deploy inside aioquic

`aero_aioquic.py` is the production integration, included as a reference. It subclasses
aioquic's `QuicCongestionControl`, registers itself with
`register_congestion_control("aero", AeroCongestionControl)`, and calls the policy from
`on_packet_acked` (slow start) and `on_packets_lost`. It derives the four state signals from
live transport state — `bytes_in_flight / latest_rtt` for throughput, `smoothed_rtt` for delay,
`|latest_rtt - smoothed_rtt|` for jitter, and a delta of the lost-packet counter — then clamps
the result to `[MIN_CWND, MAX_CWND]` bytes.

To install it into an aioquic checkout:

```bash
AIOQUIC=/path/to/aioquic/src/aioquic/quic/congestion

cp aero_aioquic.py            $AIOQUIC/aero.py
cp ppo6.py                    $AIOQUIC/ppo6.py
cp model/abr_ppo_38000.model  $AIOQUIC/
```

Then fix the two lines flagged in Step 2 to match where you put things:

- `aero.py:16` → `from aioquic.quic.congestion.ppo6 import Actor, Critic`
- `aero.py:94` → `test_model='<absolute path to abr_ppo_38000.model>'`

Select it at runtime with the congestion control name `"aero"`.

> ### Watch out: the `ppo5` / `ppo6` filename trap
> The published aioquic integration imports `ppo5`, but the file it ships under that name is
> **byte-identical to this repository's `ppo6.py`** — it was simply renamed when copied in. The
> two modules are *not* interchangeable in behavior:
>
> | | real `ppo5` | `ppo6` (what AERO uses) |
> |---|---|---|
> | `forward` noise default | `use_noise=True` | `use_noise=False` |
> | noise clamp | `±2.0` | `±1.0` |
>
> Both have an identical architecture, so a checkpoint loads into either without error — the
> difference is silent. Since `aero.py` calls `model(state)` without passing `use_noise`,
> wiring in a real `ppo5` would inject random noise into **every** congestion-window decision.
> Always deploy `ppo6.py`, under whatever name your import uses.

## The model

`model/abr_ppo_38000.model` is an actor `state_dict` for `ppo6.Actor(state_dim=4, action_dim=1)`:

```
fc1 (4 -> 256) -> BatchNorm1d -> LeakyReLU
fc2 (256 -> 256) -> BatchNorm1d -> LeakyReLU
fc3 (256 -> 1) -> tanh            # alpha, bounded to [-1, 1]
```

Load it as:

```python
import torch
from ppo6 import Actor

model = Actor(4, 1)
model.load_state_dict(torch.load('model/abr_ppo_38000.model', map_location='cpu'))
model.eval()                                  # keep BatchNorm on running statistics

with torch.no_grad():
    alpha = model(state.unsqueeze(0)).squeeze()   # state = [throughput, delay, jitter, loss]
```

Call `.eval()` and keep it there — the two BatchNorm layers must use running statistics, since
inference runs one sample at a time.

## Known issues

These are real quirks in the code, documented rather than silently patched so published results
stay reproducible:

- **Unguarded normalization in `aero_aioquic.py`.** Its running extremes start at `MAX_* = 0`
  and `MIN_* = 1000`, so for the first few calls `MAX - MIN` is zero or negative and the
  normalization can emit `inf`/`nan`. `test_aero.py` skips those rows instead of reproducing
  the behavior.
- **`drl_cc.txt` is append-only** (`parse_json1.py:81`). Delete it before regenerating.
- **Training never terminates** — stop `train_aero.py` manually.
- **`parse_json.py` injects noise.** `get_data()` adds `random.randint(1,99)` plus Gaussian
  noise to delay, and substitutes a random `0-5` when loss is zero. Traces are therefore not
  bit-reproducible between runs unless you seed it.
