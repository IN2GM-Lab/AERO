#!/usr/bin/env python3
"""
test_aero.py -- offline evaluation of the final AERO policy.

Mirrors the decision path of the production controller
(aero_aioquic.py :: get_congestion_window) but drives it from a recorded
trace instead of a live QUIC connection. This script only evaluates:
no optimizers, no backward pass, no checkpoint writes, and it terminates.

Network : ppo6.Actor      -- the module aioquic ships under the name "ppo5.py"
Model   : model/abr_ppo_38000.model
Data    : drl_cc.txt, produced by get_data.py -> parse_json1.py
          columns: throughput cwnd max_cwnd delay min_delay jitter
                   lost_packets video audio probing
"""

import argparse
import os
import statistics

import numpy as np
import torch

from ppo6 import Actor

S_DIM = 4
A_DIM = 1

# --- constants lifted from aero_aioquic.py so the reward and the action
# --- map match what the deployed controller computes.
LOSS_PENALTY = 5
DELAY_PENALTY = 3
ALPHA_BASE = 1.87          # avg learned base factor A (NOSSDAV Table 1 / §4.1.3)
SMALL_CWND = 100           # aero.py: windows below this get an extra x3
SMALL_CWND_BOOST = 3

# aero.py also clamps the result into [MIN_CWND, MAX_CWND] *bytes*. The trace
# carries cwnd in packets (parse_json1: mean(recv_rate) * 60 / 1000), so those
# byte bounds would swamp every row here. Left off by default; the meaningful
# offline ceiling is the trace's own max_cwnd column.
APPLY_BYTE_CLAMP = False
MIN_CWND = 40000
MAX_CWND = 4640000

DEFAULT_MODEL = os.path.join('model', 'abr_ppo_38000.model')
DEFAULT_DATA = 'drl_cc.txt'
DEFAULT_LOG = os.path.join('Results', 'sim', 'aero_test', 'log_test_record')

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def load_policy(model_path):
    model = Actor(S_DIM, A_DIM).type(dtype)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()               # BatchNorm1d uses running stats; never switched back
    return model


def evaluate(model, rows, log_file):
    # running extremes, as aero.py keeps them across calls
    max_thr = max_delay = max_lost = max_jitter = 0
    min_thr = min_delay = min_lost = min_jitter = 1000

    state = torch.from_numpy(np.zeros(S_DIM))
    rec = {k: [] for k in ('reward', 'alpha', 'mult', 'cwnd', 'flipped', 'over')}

    log_file.write(
        f"{'step':<8}\t{'alpha':<12}\t{'mult':<12}\t{'cwnd_in':<12}\t"
        f"{'cwnd_out':<12}\t{'max_cwnd':<12}\t{'reward':<12}\t{'flipped':<8}\n"
    )

    for step, row in enumerate(rows):
        throughput, cwnd, max_cwnd, delay, _min_delay, jitter, lost_packets = row[:7]

        max_thr, min_thr = max(max_thr, throughput), min(min_thr, throughput)
        max_delay, min_delay = max(max_delay, delay), min(min_delay, delay)
        max_lost, min_lost = max(max_lost, lost_packets), min(min_lost, lost_packets)
        max_jitter, min_jitter = max(max_jitter, jitter), min(min_jitter, jitter)

        hi = np.max([max_thr, max_jitter, max_delay, max_lost])
        lo = np.min([min_thr, min_delay, min_jitter, min_lost])
        span = hi - lo
        if span <= 0:
            # aero.py divides unguarded here and can emit inf/nan on the first
            # few calls, while its MAX_* start at 0 and MIN_* at 1000.
            continue

        n_thr = (throughput - lo) / span
        n_delay = (delay - lo) / span
        n_jitter = (jitter - lo) / span
        n_lost = (lost_packets - lo) / span

        state[0], state[1], state[2], state[3] = n_thr, n_delay, n_jitter, n_lost

        reward = n_thr - LOSS_PENALTY * n_lost - DELAY_PENALTY * n_delay

        with torch.no_grad():
            alpha = model(state.unsqueeze(0).type(dtype)).squeeze()
        alpha = float(alpha.item())

        # aero.py forces the action's sign to agree with the reward
        flipped = (reward > 0 and alpha < 0) or (reward < 0 and alpha > 0)
        if flipped:
            alpha = -alpha

        mult = ALPHA_BASE ** alpha
        cwnd_out = mult * cwnd
        if cwnd < SMALL_CWND:
            cwnd_out *= SMALL_CWND_BOOST
        cwnd_out = int(cwnd_out)
        if APPLY_BYTE_CLAMP:
            cwnd_out = min(max(cwnd_out, MIN_CWND), MAX_CWND)

        over = max_cwnd > 0 and cwnd_out > max_cwnd

        rec['reward'].append(reward)
        rec['alpha'].append(alpha)
        rec['mult'].append(mult)
        rec['cwnd'].append(cwnd_out)
        rec['flipped'].append(flipped)
        rec['over'].append(over)

        log_file.write(
            f"{step:<8}\t{alpha:<12.6f}\t{mult:<12.6f}\t{cwnd:<12.0f}\t"
            f"{cwnd_out:<12}\t{max_cwnd:<12.0f}\t{reward:<12.6f}\t{str(flipped):<8}\n"
        )

    return rec


def report(rec):
    n = len(rec['reward'])
    if n == 0:
        print("No rows evaluated.")
        return

    def line(label, value):
        print(f"  {label:<28} {value}")

    print("\n" + "=" * 60)
    print("AERO offline evaluation")
    print("=" * 60)
    line("steps evaluated", n)
    line("reward  mean", f"{statistics.fmean(rec['reward']):.6f}")
    line("reward  median", f"{statistics.median(rec['reward']):.6f}")
    line("reward  stdev", f"{statistics.pstdev(rec['reward']):.6f}" if n > 1 else "n/a")
    line("reward  min / max", f"{min(rec['reward']):.6f} / {max(rec['reward']):.6f}")
    line("reward  > 0", f"{100.0 * sum(r > 0 for r in rec['reward']) / n:.1f}%")
    print("-" * 60)
    line("alpha   mean", f"{statistics.fmean(rec['alpha']):.6f}")
    line("alpha   min / max", f"{min(rec['alpha']):.6f} / {max(rec['alpha']):.6f}")
    line("alpha   mean |a|", f"{statistics.fmean([abs(a) for a in rec['alpha']]):.6f}")
    line("sign flipped by heuristic", f"{100.0 * sum(rec['flipped']) / n:.1f}%")
    print("-" * 60)
    line("cwnd multiplier mean", f"{statistics.fmean(rec['mult']):.6f}")
    line("cwnd multiplier min / max",
         f"{min(rec['mult']):.6f} / {max(rec['mult']):.6f}")
    line("cwnd out mean", f"{statistics.fmean(rec['cwnd']):.1f}")
    line("exceeded trace max_cwnd", f"{100.0 * sum(rec['over']) / n:.1f}%")
    print("=" * 60)


def read_rows(path, limit):
    rows = []
    with open(path, 'r') as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) < 7:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue          # skip a header or a malformed line
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    parser = argparse.ArgumentParser(description='Offline evaluation of the AERO policy')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='actor checkpoint')
    parser.add_argument('--data', default=DEFAULT_DATA, help='trace produced by get_data.py')
    parser.add_argument('--steps', type=int, default=0, help='limit rows (0 = all)')
    parser.add_argument('--log', default=DEFAULT_LOG, help='per-step log file')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        parser.error(f"checkpoint not found: {args.model}")
    if not os.path.exists(args.data):
        parser.error(
            f"trace not found: {args.data}\n"
            "Generate it first:  python3 get_data.py   "
            "(needs .json files under /home/ubuntu/Json/)"
        )

    rows = read_rows(args.data, args.steps)
    if not rows:
        parser.error(f"no usable rows in {args.data}")

    os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)
    model = load_policy(args.model)
    print(f"model : {args.model}")
    print(f"data  : {args.data}  ({len(rows)} rows)")
    print(f"log   : {args.log}")

    with open(args.log, 'w') as log_file:
        rec = evaluate(model, rows, log_file)
    report(rec)


if __name__ == '__main__':
    main()
