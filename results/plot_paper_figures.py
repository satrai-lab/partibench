"""
Reproduce the five paper figures from benchmark data.

Usage (from any directory):
    python results/plot_paper_figures.py                          # use bundled paper data
    python results/plot_paper_figures.py --bench path/to/bench.json
    python results/plot_paper_figures.py --out path/to/output_dir

Figures produced
----------------
fig6_activation_sizes.pdf  – Intermediate activation sizes across VGG16 blocks
fig7_pipelined.pdf         – End-to-end inference time: pipelined vs centralised baselines
fig8_input_growth.pdf      – Input size growth for input-partitioned sequences
fig9_input_partitioning.pdf– Optimal input distribution and inference time (Scenario A/B)
fig10_filter_vs_input.pdf  – Execution time and memory: filter vs input partitioning (F=25)

Data sources
------------
Figures 6, 8 : read from benchmark.json (model_specifics section).
Figure 10    : read from benchmark.json if edge_A node data is present;
               falls back to the hardcoded EC2 measurements from the paper.
Figures 7, 9 : use hardcoded EC2 measurements and analytical network model.
               To reproduce from your own runs, update the PAPER_DATA dict below.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BENCH = os.path.join(_HERE, "..", "local_example", "benchmarks", "benchmark.json")
_DEFAULT_OUT = os.path.join(_HERE, "PLOTS", "reproduced")

# ---------------------------------------------------------------------------
# Hardcoded paper measurements (EC2 experiments)
# Update these if you run your own multi-node experiments.
# ---------------------------------------------------------------------------
PAPER_DATA = {
    # Fig 7 – pipelined model partitioning
    # comm and exec times in ms (measured on EC2, includes application-level
    # simulated network delays).
    "fig7": {
        "labels":    ["IoT-only\n(iot_A1)", "Edge-only\n(edge_A)",
                      "Cloud-only\n(cloud)",  "Pipeline\n(iot_A1+edge_A)"],
        "exec_ms":   [1054.8, 463.2, 256.2, 796.6],
        "comm_ms":   [0.0,    696.0, 1089.0, 54.0],
        "exec_std":  [45.9,   55.5,  42.4,   55.9],
    },

    # Fig 9 – input partitioning analytical model
    # Execution times for blocks 0–6 (ms), measured on EC2.
    "fig9_exec": {
        "iot_A1":  789.4,
        "iot_A2":  789.4,   # same hardware tier as iot_A1
        "iot_A3":  789.4,
        "edge_A":  290.7,
        "cloud":   196.1,
    },
    # EC2 network parameters (one-way latency in ms, bandwidth in Mbps).
    "fig9_net": {
        "IoT_IoT":   {"bw": 200, "lat": 8},
        "IoT_Edge":  {"bw": 150, "lat": 11},
        "Edge_Cloud":{"bw": 100, "lat": 50},
    },

    # Fig 10 – filter vs input partitioning on edge_A, F=25
    # exec times in ms (mean over 5 runs), memory in MB.
    "fig10": {
        "fs25_exec": [8.6, 47.1, 10.1, 24.8, 9.7, 19.1, 14.2, 5.2,
                      9.3, 12.2,  3.9,  3.6,  4.4, 13.1, 2.2, 0.4],
        "fs25_std":  [2.1,  2.7,  0.1,  1.6,  1.5,  4.6,  1.2, 0.2,
                      0.1,  0.8,  0.9,  0.3,  0.9,  2.7,  0.9, 0.0],
        "is25_exec": [1.6, 18.6,  5.1,  9.5,  7.0, 12.4, 10.4, 5.3,
                      10.6, 15.0,  5.1,  6.7,  6.3, 11.6, 2.1, 0.4],
        "is25_std":  [0.1,  2.9,  0.0,  0.2,  0.7,  2.7,  2.3, 0.1,
                      0.7,  2.6,  0.5,  0.8,  1.9,  0.8,  0.4, 0.0],
        "fs25_mem":  [14.4, 26.4, 14.1, 17.6, 13.5, 16.0, 16.0, 19.5,
                      31.4, 31.4, 27.3, 27.2, 27.3, 204.5, 40.5, 31.8],
        "is25_mem":  [12.0, 17.0, 14.3, 15.1, 18.7, 29.4, 29.3, 44.9,
                      81.7, 82.0, 80.5, 80.5, 80.6, 204.6, 40.6, 31.8],
    },
}

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,   # embed fonts for IEEE submission
    "ps.fonttype": 42,
})

COLORS = {
    "iot":     "#e05c5c",
    "edge":    "#f5a623",
    "cloud":   "#4a90d9",
    "pipe":    "#27ae60",
    "filter":  "#8e44ad",
    "input":   "#2980b9",
    "mem_fs":  "#c0392b",
    "mem_is":  "#2471a3",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bytes_to_mb(b):
    return b / (1024 ** 2)


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def _load_bench(path):
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        print(f"  [warn] benchmark file not found: {expanded}")
        return None
    with open(expanded) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 6 – Activation sizes between consecutive VGG16 blocks
# ---------------------------------------------------------------------------
def plot_fig6(bench, outdir):
    print("Plotting Fig 6 – activation sizes...")
    data = bench.get("vgg16", {}).get("model_specifics", {})
    out_sizes = data.get("out_sizes", {}).get("fs_1", {})
    if isinstance(out_sizes, dict):
        # Nested structure: use is_1 (full-input, no-split output sizes)
        out_sizes = out_sizes.get("is_1", [])
    if not out_sizes:
        print("  [skip] output size data not found in benchmark.json")
        return

    # out_sizes[i] = bytes transferred from block i to block i+1
    # (last entry is block 15 → PRODUCER, kept for completeness)
    mb = [_bytes_to_mb(s) for s in out_sizes]
    n = len(mb)
    x = list(range(n))
    labels = [f"{i}→{i+1}" if i < n - 1 else f"{i}→P" for i in range(n)]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(x, mb, color=COLORS["cloud"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Transfer between consecutive blocks")
    ax.set_ylabel("Activation size (MB)")
    ax.set_title("Intermediate feature map sizes transferred between VGG16 blocks")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # annotate the attractive split point (block 6 → block 7)
    split = 6
    ax.axvline(x=split + 0.5, color="red", linestyle="--", linewidth=1.2,
               label=f"Split 6→7 ({mb[split]:.2f} MB)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    _save(fig, outdir, "fig6_activation_sizes.pdf")


# ---------------------------------------------------------------------------
# Figure 7 – Pipelined model partitioning vs centralised baselines
# ---------------------------------------------------------------------------
def plot_fig7(outdir):
    print("Plotting Fig 7 – pipelined vs baselines...")
    d = PAPER_DATA["fig7"]
    labels    = d["labels"]
    exec_ms   = np.array(d["exec_ms"])
    comm_ms   = np.array(d["comm_ms"])
    exec_std  = np.array(d["exec_std"])
    totals    = exec_ms + comm_ms

    x = np.arange(len(labels))
    width = 0.55
    bar_colors = [COLORS["iot"], COLORS["edge"], COLORS["cloud"], COLORS["pipe"]]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars_exec = ax.bar(x, exec_ms, width, label="Execution", color=bar_colors, alpha=0.85)
    bars_comm = ax.bar(x, comm_ms, width, bottom=exec_ms, label="Communication",
                       color=bar_colors, alpha=0.4, hatch="//")

    # error bars on total
    ax.errorbar(x, totals, yerr=exec_std, fmt="none", color="black",
                capsize=4, linewidth=1.2)

    # annotate totals
    for xi, total in zip(x, totals):
        ax.text(xi, total + 25, f"{total:.0f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_ylabel("Inference time (ms)")
    ax.set_title("End-to-end VGG16 inference time by deployment strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(totals) * 1.22)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    _save(fig, outdir, "fig7_pipelined.pdf")


# ---------------------------------------------------------------------------
# Figure 8 – Input size growth for input-partitioned sequences
# ---------------------------------------------------------------------------
def plot_fig8(bench, outdir):
    print("Plotting Fig 8 – input size growth...")
    data = bench.get("vgg16", {}).get("model_specifics", {})
    inp = data.get("inp_sizes", {}).get("fs_1", {})
    full_inp = data.get("inp_sizes", {}).get("fs_1", {})

    # full (no-split) input size for block 0
    is1 = full_inp.get("is_1")
    if not is1:
        print("  [skip] input size data not found")
        return
    full_size_mb = _bytes_to_mb(is1[0])  # block 0 full input

    curves = {}
    for key, label, color in [
        ("is_0.25", "F = 25%", "#2ecc71"),
        ("is_0.5",  "F = 50%", "#f39c12"),
        ("is_0.75", "F = 75%", "#e74c3c"),
    ]:
        block0 = inp.get(key)
        if block0 and isinstance(block0[0], list):
            # sizes for starting block 0 at increasing depths
            sizes_mb = [_bytes_to_mb(s) for s in block0[0]]
            curves[label] = (color, sizes_mb)

    if not curves:
        print("  [skip] partitioned input size data not found")
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, (color, sizes) in curves.items():
        depths = list(range(1, len(sizes) + 1))
        ax.plot(depths, sizes, marker="o", markersize=4, label=label, color=color)

    ax.axhline(full_size_mb, color="black", linestyle="--", linewidth=1,
               label=f"Full input ({full_size_mb:.2f} MB)")
    ax.set_xlabel("Sequence depth (number of blocks starting from block 0)")
    ax.set_ylabel("Input size per node (MB)")
    ax.set_title("Input size growth due to convolutional overlap amplification")
    ax.legend()
    ax.grid(linestyle=":", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    _save(fig, outdir, "fig8_input_growth.pdf")


# ---------------------------------------------------------------------------
# Figure 9 – Theoretical optimal input partitioning (Scenario A and B)
# ---------------------------------------------------------------------------
def _solve_partition(nodes):
    """
    Given a list of dicts with keys {name, E, CommVol, lat}, compute the
    optimal total inference time T and per-node partition fractions f_i (%).

    Formula (from paper §VI-B2):
        τ(n_i) = (E_i + CommVol_i) * f_i/100 + lat_i  (all equal to T)
        Σ f_i = 100

    Returns (T, {name: f_i}).  Nodes with f_i ≤ 0 should be excluded.
    """
    # A = Σ 1/(E_i + CommVol_i),  B = Σ lat_i / (E_i + CommVol_i)
    A = sum(1.0 / (n["E"] + n["CommVol"]) for n in nodes)
    B = sum(n["lat"] / (n["E"] + n["CommVol"]) for n in nodes)
    T = (1.0 + B) / A
    fracs = {n["name"]: 100.0 * (T - n["lat"]) / (n["E"] + n["CommVol"])
             for n in nodes}
    return T, fracs


def _build_nodes_scenario_a(exec_times, net):
    """
    Scenario A: source = iot_A1, merger = cloud, nominal conditions.
    Returns list of node dicts for _solve_partition.
    """
    inp_bytes  = 602136   # block 0 input (bytes)
    out_bytes  = 802840   # block 6 output (bytes)

    def bw_time(size_bytes, bw_mbps):
        return size_bytes * 8 / (bw_mbps * 1e6) * 1e3  # ms

    bw_ii  = net["IoT_IoT"]["bw"]
    lat_ii = net["IoT_IoT"]["lat"]
    bw_ie  = net["IoT_Edge"]["bw"]
    lat_ie = net["IoT_Edge"]["lat"]
    bw_ec  = net["Edge_Cloud"]["bw"]
    lat_ec = net["Edge_Cloud"]["lat"]

    # bottleneck bandwidth from IoT to cloud
    bw_ic = min(bw_ie, bw_ec)
    lat_ic = lat_ie + lat_ec

    nodes = [
        # iot_A1: source → input is free; output goes to cloud (bottleneck bw_ic)
        dict(name="iot_A1",
             E=exec_times["iot_A1"],
             CommVol=bw_time(out_bytes, bw_ic),
             lat=lat_ic),
        # iot_A2: receives input from iot_A1 via IoT-IoT; output to cloud via bw_ic
        dict(name="iot_A2",
             E=exec_times["iot_A2"],
             CommVol=bw_time(inp_bytes, bw_ii) + bw_time(out_bytes, bw_ic),
             lat=lat_ii + lat_ic),
        # iot_A3: same as iot_A2
        dict(name="iot_A3",
             E=exec_times["iot_A3"],
             CommVol=bw_time(inp_bytes, bw_ii) + bw_time(out_bytes, bw_ic),
             lat=lat_ii + lat_ic),
        # edge_A: input from iot_A1 via IoT-Edge; output to cloud via Edge-Cloud
        dict(name="edge_A",
             E=exec_times["edge_A"],
             CommVol=bw_time(inp_bytes, bw_ie) + bw_time(out_bytes, bw_ec),
             lat=lat_ie + lat_ec),
        # cloud: input from iot_A1 via bw_ic; already at merger → no output transfer
        dict(name="cloud",
             E=exec_times["cloud"],
             CommVol=bw_time(inp_bytes, bw_ic),
             lat=lat_ic),
    ]
    return nodes


def _build_nodes_scenario_b(exec_times, net):
    """
    Scenario B: +80ms edge-cloud latency, iot_A1 +200ms exec delay,
    merger = edge_A.
    """
    inp_bytes = 602136
    out_bytes = 802840

    def bw_time(size_bytes, bw_mbps):
        return size_bytes * 8 / (bw_mbps * 1e6) * 1e3

    bw_ii  = net["IoT_IoT"]["bw"]
    lat_ii = net["IoT_IoT"]["lat"]
    bw_ie  = net["IoT_Edge"]["bw"]
    lat_ie = net["IoT_Edge"]["lat"]
    bw_ec  = net["Edge_Cloud"]["bw"]
    lat_ec = net["Edge_Cloud"]["lat"] + 80     # degraded

    exec_iot_a1_b = exec_times["iot_A1"] + 200  # +200ms penalty

    nodes = [
        # iot_A1: source, output to edge_A (merger) via IoT-Edge
        dict(name="iot_A1",
             E=exec_iot_a1_b,
             CommVol=bw_time(out_bytes, bw_ie),
             lat=lat_ie),
        # iot_A2: input from iot_A1 via IoT-IoT; output to edge_A via IoT-Edge
        dict(name="iot_A2",
             E=exec_times["iot_A2"],
             CommVol=bw_time(inp_bytes, bw_ii) + bw_time(out_bytes, bw_ie),
             lat=lat_ii + lat_ie),
        # iot_A3: same as iot_A2
        dict(name="iot_A3",
             E=exec_times["iot_A3"],
             CommVol=bw_time(inp_bytes, bw_ii) + bw_time(out_bytes, bw_ie),
             lat=lat_ii + lat_ie),
        # edge_A: input from iot_A1 via IoT-Edge; already at merger
        dict(name="edge_A",
             E=exec_times["edge_A"],
             CommVol=bw_time(inp_bytes, bw_ie),
             lat=lat_ie),
        # cloud: input via bw_ec path; output back to edge_A via Edge-Cloud (degraded)
        dict(name="cloud",
             E=exec_times["cloud"],
             CommVol=bw_time(inp_bytes, min(bw_ie, bw_ec)) + bw_time(out_bytes, bw_ec),
             lat=(lat_ie + lat_ec) + lat_ec),
    ]
    return nodes


def _iterative_removal(all_nodes):
    """
    Starting from all_nodes, iteratively remove the node with the smallest
    (or negative) partition fraction, recording T at each step.
    Returns list of (num_nodes, T, fracs_dict) from max nodes down to 1.
    """
    active = list(all_nodes)
    results = []
    while active:
        # filter out any node whose partition factor would be ≤ 0
        T, fracs = _solve_partition(active)
        valid = [n for n in active if fracs[n["name"]] > 0]
        if len(valid) < len(active):
            # some nodes are infeasible at current T; recompute
            active = valid
            if not active:
                break
            T, fracs = _solve_partition(active)
        results.append((len(active), T, dict(fracs)))
        if len(active) == 1:
            break
        # remove the node contributing least (smallest fraction)
        min_node = min(active, key=lambda n: fracs[n["name"]])
        active = [n for n in active if n["name"] != min_node["name"]]
    return list(reversed(results))  # ascending order (1 node → all nodes)


def plot_fig9(outdir):
    print("Plotting Fig 9 – optimal input partitioning...")
    exec_times = PAPER_DATA["fig9_exec"]
    net        = PAPER_DATA["fig9_net"]

    nodes_a = _build_nodes_scenario_a(exec_times, net)
    nodes_b = _build_nodes_scenario_b(exec_times, net)

    results_a = _iterative_removal(nodes_a)
    results_b = _iterative_removal(nodes_b)

    # --- line plot: T vs number of nodes ---
    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(10, 4))

    n_a = [r[0] for r in results_a]
    T_a = [r[1] for r in results_a]
    n_b = [r[0] for r in results_b]
    T_b = [r[1] for r in results_b]

    ax_t.plot(n_a, T_a, "o-", color=COLORS["cloud"],  label="Scenario A (baseline)",
              markersize=6, linewidth=1.8)
    ax_t.plot(n_b, T_b, "s--", color=COLORS["iot"],   label="Scenario B (degraded)",
              markersize=6, linewidth=1.8)

    # annotate the best 5-node and 4-node times
    ax_t.annotate(f"{T_a[-1]:.1f} ms",
                  xy=(n_a[-1], T_a[-1]), xytext=(n_a[-1] - 0.6, T_a[-1] + 20),
                  fontsize=8, color=COLORS["cloud"])
    ax_t.annotate(f"{T_b[-1]:.1f} ms",
                  xy=(n_b[-1], T_b[-1]), xytext=(n_b[-1] - 0.6, T_b[-1] - 30),
                  fontsize=8, color=COLORS["iot"])

    ax_t.set_xlabel("Number of participating nodes")
    ax_t.set_ylabel("Minimum inference time (ms)")
    ax_t.set_title("Theoretical inference time vs node count")
    ax_t.legend()
    ax_t.grid(linestyle=":", alpha=0.5)
    ax_t.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- bar chart: optimal partition fractions (max nodes scenario) ---
    best_a_fracs = results_a[-1][2]   # all-nodes fractions for Scenario A
    best_b_fracs = results_b[-1][2]   # all-nodes fractions for Scenario B
    node_names = [n["name"] for n in nodes_a
                  if best_a_fracs.get(n["name"], 0) > 0
                  or best_b_fracs.get(n["name"], 0) > 0]

    xa = np.arange(len(node_names))
    w = 0.38
    fa = [max(best_a_fracs.get(n, 0), 0) for n in node_names]
    fb = [max(best_b_fracs.get(n, 0), 0) for n in node_names]

    ax_f.bar(xa - w / 2, fa, w, color=COLORS["cloud"], alpha=0.85, label="Scenario A")
    ax_f.bar(xa + w / 2, fb, w, color=COLORS["iot"],   alpha=0.85, label="Scenario B")
    ax_f.set_xticks(xa)
    ax_f.set_xticklabels(node_names, rotation=20, ha="right")
    ax_f.set_ylabel("Input fraction (%)")
    ax_f.set_title("Optimal input distribution (maximum node count)")
    ax_f.legend()
    ax_f.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    _save(fig, outdir, "fig9_input_partitioning.pdf")


# ---------------------------------------------------------------------------
# Figure 10 – Filter vs input partitioning (F=25, edge_A)
# ---------------------------------------------------------------------------
def plot_fig10(bench, outdir):
    print("Plotting Fig 10 – filter vs input partitioning (F=25)...")
    d = PAPER_DATA["fig10"]

    # Try to read from benchmark.json if edge_A data is present
    node_data = bench.get("vgg16", {}).get("node_specifics", {}).get("edge_A")
    if node_data:
        et = node_data.get("exec_times", {})
        mem = bench["vgg16"]["model_specifics"].get("mem_cons", {})

        def _extract_exec(key):
            entries = et.get(key, [])
            if not entries or not isinstance(entries[0], dict):
                return None, None
            means = [e.get("et_mean", 0) for e in entries]
            stds  = [e.get("et_std",  0) for e in entries]
            return means, stds

        fs_exec, fs_std = _extract_exec("fs_0.25")
        is_exec, is_std = _extract_exec("is_0.25")
        fs_mem = mem.get("fs_0.25")
        is_mem = mem.get("is_0.25")
        if fs_exec and is_exec and fs_mem and is_mem:
            print("  Using edge_A data from benchmark.json")
        else:
            fs_exec = is_exec = fs_mem = is_mem = None
    else:
        fs_exec = is_exec = fs_mem = is_mem = None

    if fs_exec is None:
        print("  Falling back to hardcoded paper data")
        fs_exec = d["fs25_exec"]
        fs_std  = d["fs25_std"]
        is_exec = d["is25_exec"]
        is_std  = d["is25_std"]
        fs_mem  = d["fs25_mem"]
        is_mem  = d["is25_mem"]

    blocks = list(range(len(fs_exec)))
    x = np.arange(len(blocks))
    width = 0.4

    fig, (ax_e, ax_m) = plt.subplots(1, 2, figsize=(10, 4))

    # Execution time
    ax_e.bar(x - width / 2, fs_exec, width, yerr=fs_std, capsize=2,
             color=COLORS["filter"], alpha=0.85, label="Filter split (F=25%)")
    ax_e.bar(x + width / 2, is_exec, width, yerr=is_std, capsize=2,
             color=COLORS["input"],  alpha=0.85, label="Input split (F=25%)")
    ax_e.set_xlabel("VGG16 block index")
    ax_e.set_ylabel("Execution time (ms)")
    ax_e.set_title("Execution time: filter vs input partitioning (F=25%)")
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(blocks)
    ax_e.legend()
    ax_e.grid(axis="y", linestyle=":", alpha=0.5)

    # Memory consumption
    ax_m.bar(x - width / 2, fs_mem, width,
             color=COLORS["mem_fs"], alpha=0.85, label="Filter split (F=25%)")
    ax_m.bar(x + width / 2, is_mem, width,
             color=COLORS["mem_is"], alpha=0.85, label="Input split (F=25%)")
    ax_m.set_xlabel("VGG16 block index")
    ax_m.set_ylabel("Memory (MB)")
    ax_m.set_title("Memory consumption: filter vs input partitioning (F=25%)")
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(blocks)
    ax_m.legend()
    ax_m.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    _save(fig, outdir, "fig10_filter_vs_input.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Reproduce paper figures from PartiBench benchmark data.")
    parser.add_argument("--bench", default=_DEFAULT_BENCH,
                        help="Path to benchmark.json (default: local_example/benchmarks/benchmark.json)")
    parser.add_argument("--out",   default=_DEFAULT_OUT,
                        help="Output directory for PDF figures")
    parser.add_argument("--figs",  nargs="+", type=int,
                        choices=[6, 7, 8, 9, 10], default=[6, 7, 8, 9, 10],
                        help="Which figures to generate (default: all)")
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("numpy is required: pip install numpy")

    bench = _load_bench(args.bench)
    if bench is None:
        bench = {}
        print("  Continuing without benchmark.json — figures 6 and 8 will be skipped.")

    os.makedirs(args.out, exist_ok=True)
    print(f"Output directory: {args.out}\n")

    if 6 in args.figs:
        if bench:
            plot_fig6(bench, args.out)
        else:
            print("  [skip] Fig 6 requires benchmark.json")

    if 7 in args.figs:
        plot_fig7(args.out)

    if 8 in args.figs:
        if bench:
            plot_fig8(bench, args.out)
        else:
            print("  [skip] Fig 8 requires benchmark.json")

    if 9 in args.figs:
        plot_fig9(args.out)

    if 10 in args.figs:
        plot_fig10(bench if bench else {}, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
