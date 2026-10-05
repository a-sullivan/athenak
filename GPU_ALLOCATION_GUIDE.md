# GPU Allocation and Job Sizing for AthenaK

A guide to understanding and configuring HPC GPU jobs for AthenaK simulations. This covers memory, GPU allocation, and when to scale to multiple GPUs.

## Key Concept: Two Separate Memories

When you run a GPU job, there are **two completely different memory pools** that have nothing to do with each other:

| | Where it lives | How you get it | How much you have |
|---|---|---|---|
| **Host RAM** | On the node's motherboard (attached to CPUs) | `--mem` in Slurm script | ~1 TB per Rusty GPU node |
| **GPU memory (VRAM)** | Soldered onto the GPU card | Cannot request; comes with card | 80 GB on `a100-sxm4-80gb` |

**Your simulation data lives in GPU memory.** The `--mem` flag does **not** control GPU memory. If you set `--mem=128GB`, you get 128 GB of CPU RAM that your job will mostly not touch. The whole GPU, including all 80 GB of VRAM, arrives when Slurm assigns you the card.

## GPU Memory Usage by Resolution

AthenaK allocates approximately **78 doubles per cell** for MHD simulations with your configuration (5 variables, no mesh refinement, `fofc = true`). This includes `u0`, `w0`, `u1`, face-centered B-fields, fluxes, EMFs, and reconstruction states.

The grid size (`<mesh> nx1/2/3`) determines total cells, but you must add ghost cells (set by `nghost`). With `nghost=4` and `<meshblock> = 64`:

| Grid | Cells (excl. ghosts) | Cells (incl. ghosts) | GPU memory |
|---|---|---|---|
| 128³ | 2.1M | 3.0M | ~1.9 GB |
| 256³ | 16.8M | 23.9M | ~15 GB |
| 512³ | 134M | 191M | ~119 GB* |

*Exceeds single A100 capacity; requires 2+ GPUs.

**Memory is a hard ceiling.** Exceed it and your job dies with OOM. Below it, unused memory doesn't help — having 15 GB used on an 80 GB card is not slower than 70 GB used.

## Host Memory (`--mem`)

Recommended: **32 GB** per node. This covers:
- CUDA runtime initialization (~hundreds of MB)
- Output staging (binary dumps convert to 32-bit floats in host RAM)
- MPI communication buffers and pinned memory for InfiniBand
- Executable, libraries, process overhead

32 GB is generous and keeps you in the safe zone without being greedy.

## How Many GPUs to Request?

GPU allocation is controlled by **two hard constraints** and one soft preference:

### Hard Constraint 1: MeshBlock Count

AthenaK cannot put fewer than one MeshBlock on an MPI rank, and you get one rank per GPU. The grid is decomposed into MeshBlocks of size `<meshblock> nx1/2/3`.

**Maximum usable GPUs = total number of MeshBlocks**

Example: `<mesh> = 128³` with `<meshblock> = 64` gives 2×2×2 = **8 MeshBlocks**, so max 8 GPUs. Asking for 16 GPUs will cause AthenaK to refuse to start.

### Hard Constraint 2: Memory

If your total grid memory exceeds 80 GB per GPU, you need more GPUs. With 2 GPUs, you need ≤160 GB total; with 4, ≤320 GB, etc.

### Soft Preference: Computational Efficiency

**A GPU needs enough work to be efficient.** An A100 has ~6,912 CUDA cores, and AthenaK assigns roughly one thread per cell. The more cells per GPU, the busier it is.

- **Too few cells:** Most cores sit idle. Splitting 64³ = 262K cells across 8 GPUs gives ~38K cells per GPU (~5 cells per core). You're GPU-starved.
- **One GPU per 2M+ cells** is a healthy target. At 64³ per block (3M cells with ghosts), one GPU is busy. Two 64³ blocks per GPU (6M cells) is better still.

### Cost of Splitting Work

When you use multiple GPUs on 2+ nodes, every timestep requires network communication for ghost cell halo exchange. This has a fixed overhead that doesn't scale down with problem size. For small problems that finish in seconds to minutes, this overhead dominates.

**Rule of thumb:** If your 1-GPU runtime is <5 minutes, don't use 2 GPUs. The queue wait typically exceeds the speedup gain. A100 nodes have only 4 GPUs per node, so 8 GPUs forces you onto 2+ nodes and over InfiniBand.

## Quick Reference: When to Use N GPUs

| Grid | tlim | Cells/GPU | 1 GPU time | Recommendation |
|---|---|---|---|---|
| 128³ | 100 | 3M | 1–2 min | **1 GPU** |
| 256³ | 100 | 24M | 4–20 min | **1 GPU** |
| 256³ | 600+ (10 rotations) | 24M | 30–120 min | **2 GPUs** |
| 512³ | any | 191M | >1 hour | **2–4 GPUs** (memory-bound) |

**Note:** Doubling resolution costs **16×** the work (8× more cells, 2× more timesteps due to CFL). Extending `tlim` is linear in cost.

## Actual Timestep Calculation

AthenaK computes `dt = cfl_number × min(dx₁/λ₁, dx₂/λ₂, dx₃/λ₃)` — a per-direction minimum, not a combined 3D limit. For RK2 in 3D, the stability-safe range is ~0.2–0.4. Do **not** rely on warnings; the code won't stop you from using an unstable CFL. For `split_monopole.athinput`, use `cfl_number ≤ 0.4`.

## Slurm Script Best Practices

### Memory limit fix (CRITICAL)

Your login shell's `ulimit -l` (locked memory) was 8 kB, and Slurm propagates it to compute nodes. UCX (the MPI layer) cannot pin enough memory, and all tasks fail before the first cycle.

**Fix:** Add to your script:
```bash
#SBATCH --propagate=NONE
ulimit -l unlimited
```

### Recommended configuration for single GPU

```bash
#!/bin/bash
#SBATCH --job-name=split_monopole
#SBATCH -p gpu
#SBATCH -C a100                  # constraint to a100 nodes if built with sm_80
#SBATCH --nodes=1                
#SBATCH --ntasks-per-node=1      # 1 MPI rank = 1 GPU
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8        # CPU host backend; CPUs for MPI + I/O
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --propagate=NONE

ulimit -l unlimited

module purge
module load cuda/12.8.0 openmpi/cuda-4.1.8

export LD_PRELOAD=/mnt/sw/fi/cephtweaks/lib/libcephtweaks.so
export CEPHTWEAKS_LAZYIO=1

srun --cpu-bind=cores ./athena -i split_monopole.athinput
```

### Recommended configuration for 2 GPUs (same node)

```bash
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# rest identical
```

Keep `--nodes=1` to stay on a single A100 node (which has 4 GPUs). This avoids InfiniBand overhead.

## Measuring Actual Performance

Once a job completes, check two things:

1. **Job efficiency:** `seff <jobid>` shows actual CPU/memory usage
2. **Computational throughput:** Look for the line `zone-cycles/cpu_second` at the end of the log. This is your actual measured throughput, which you can use to predict runtime for larger problems.

## Example: Scaling to Larger Problems

Starting from your 128³ baseline with measured `zone-cycles/cpu_second = X`:

For a 256³ run at the same `tlim`:
- **Work increase:** 16× (as discussed above)
- **Predicted time on 1 GPU:** 16 × (128³ time) = ~15–30 min
- **Decision:** Still 1 GPU (comfortable <1 hour)

For a 256³ run with `tlim = 600` (10 rotations):
- **Work increase:** 16 × 6 = 96×
- **Predicted time on 1 GPU:** 2–5 hours
- **Decision:** Use 2 GPUs (~1.7× speedup, wall time ≈1 hour)

For a 512³ run:
- **Memory:** ~119 GB (exceeds 80 GB) → **must use ≥2 GPUs**
- **Even with 2 GPUs:** 60 GB per card (tight with CUDA context). **Recommend 4 GPUs** (30 GB per card)

## Architecture Considerations

Your binary was built with `Kokkos_ARCH_AMPERE80` (NVIDIA sm_80, the A100 architecture). Use `#SBATCH -C a100` to request A100 nodes specifically, or rebuild with `-DKokkos_ARCH_HOPPER90=ON` to target H100 nodes (8 per node, so 8 ranks fit on one node without splitting).

## Related Configuration Options

- **Ghost cells (`nghost`):** Larger ghosts = more work per block. Default 4 is standard; increasing to 5+ is rarely worth it.
- **MeshBlock size (`<meshblock>`):** Larger blocks = fewer blocks and fewer MPI ranks possible. Smaller blocks = more overhead. For single-GPU runs, consider `<meshblock> = 128` to reduce ghost overhead to ~20%.
- **Output frequency (`dt` in `<output>` blocks):** Each binary dump is ~67 MB per variable and happens in host RAM. Frequent outputs can dominate runtime; see actual I/O cost with `seff` after a run.

---

**Last updated:** 2026-09-15  
**Relevant configuration:** split_monopole.athinput, 128³ grid, RK2 integrator, Kokkos_ARCH_AMPERE80
