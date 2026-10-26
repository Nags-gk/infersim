# Study guide: questions this project invites

Short answers you should be able to give without notes, and where each lives in the code.

## Roofline

**What is arithmetic intensity, and where is the ridge point?**
FLOPs divided by bytes moved to and from the memory that holds the operands. A kernel with
intensity I can reach at most min(peak, I x bandwidth). The ridge is peak / bandwidth: for H100
BF16, 989 TFLOP/s / 3.35 TB/s = about 295 FLOP/byte. (`hardware.ridge_point`)

**Why is decode memory-bound?**
A decode GEMM has M = batch. Its FLOPs are 2·M·N·K and its bytes are dominated by the K·N
weights (2 bytes each in BF16), so intensity is about M. Below batch ~295 the weights cannot be
streamed fast enough to keep the tensor cores busy. The model's crossover is batch 257, slightly
lower because achievable efficiencies are 0.70 (compute) and 0.85 (memory) and M is padded to
64-row tiles. (`bench/run_bench.py: decode_crossover_batch`)

**Why does FP8 not move the crossover batch?**
FP8 halves weight bytes (intensity doubles) but also doubles peak FLOPs (ridge doubles).
The ratio, which sets the crossover, is unchanged.

**Why is prefill compute-bound?**
M = prompt tokens (thousands), so intensity is in the thousands, well past the ridge.

## Attention and KV cache

**KV bytes per token for Llama-3.1-8B?** 32 layers x 2 (K and V) x 8 KV heads x 128 dims x 2 bytes
= 128 KiB. GQA shares each KV head across 4 query heads, so the cache is 4x smaller than MHA.

**How does MLA shrink the cache?** DeepSeek-V3 caches a 512-dim compressed latent plus a 64-dim
rotary key per layer (576 values), not per-head K and V. During decode the up-projection weights
are absorbed into the query and output projections, so attention runs directly on the latent.
(`ops._attention_ops`)

**Why does decode attention cost grow with batch even though weights do not?** Each sequence
reads its own cache: bytes scale with sum(context lengths). At batch 64 and 2k context,
Llama-3.1-8B reads more KV than weights.

## Parallelism and communication

**Ring vs one-shot all-reduce?** Ring sends 2(p-1)/p of the payload over each link in 2(p-1)
steps: bandwidth-optimal, latency-heavy. One-shot has each rank read all peers once: (p-1) times
the payload but only two synchronizations. Small decode messages favor one-shot; large prefill
messages favor ring. The model takes the faster. (`roofline.allreduce_time`)

**How many all-reduces per token under TP?** Two per layer (after attention and after the MLP):
160 for Llama-3.1-70B.

**Expert parallelism?** Experts are split across ranks instead of being sliced; tokens are sent to
their experts and back with two all-to-alls per MoE layer, replacing the FFN all-reduce.

**How many distinct experts does a batch touch?** E·(1 - (1 - k/E)^T) for T tokens choosing k of
E experts uniformly. One token touches k; large batches touch all E, so MoE decode reads more
weights as batch grows. (`ops.expected_active_experts`)

## LPU and heterogeneous systems

**Why do LPUs need so many chips?** Weights must live in on-chip SRAM: a 70B model in FP8 is about
70 GB, and a Groq 3 LPU has about 500 MB, so 157 chips before any KV cache.

**Why are they fast per user?** SRAM bandwidth per chip (150 TB/s assumed, top of the quoted
range) is far above HBM, so weight streaming is cheap; latency is dominated by chip-to-chip hops.

**Why less throughput per chip?** Little room for KV cache limits concurrent users, and many chips
share one model.

**What do the two heterogeneous layouts trade?** Disaggregation sends the KV cache once per
request (prompt tokens x KV bytes per token over the network). The attention/FFN split sends
activations twice per layer per step, so it only pays off if those transfers are cheap.

## Design space exploration

**What is on the Pareto chart?** x is tokens/s per user (1 / TPOT), y is tokens/s per device. A
layout is on the frontier if nothing else beats it on both. **Steady state:** for U users each
generating O tokens, the cycle takes O decode steps plus one prefill per user.

## Calibration

**What gets fitted?** eff_compute, eff_memory and launch_overhead, minimizing the mean squared
log error between predicted and measured step latency (Nelder-Mead). **Why log error?** Steps span
microseconds to seconds; log error weights relative error equally. **How do you know it is not
overfitting?** Every fourth measurement is held out; MAPE is reported on both. The test suite
checks the fitter recovers known parameters from synthetic data.

## Honest limitations to volunteer

No measured calibration yet; no compute/communication overlap; balanced experts; assumed LPU
compute and links. Absolute numbers are estimates; comparisons are the point.
