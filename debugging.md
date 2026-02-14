# Debugging

---

## Nsight Introduction

NVIDIA Nsight is a suite of tools for debugging and profiling GPU applications. 

- **Nsight Systems (`nsys`):** It shows the system-wide timeline, CPU-GPU interactions, and memory transfers. You use this *first* to find bottlenecks (e.g., "Why is my GPU idle here?").
- **Nsight Compute (`ncu`):** Once `nsys` tells you *which* kernel is slow, you use `ncu` to inspect that specific kernel's instruction-level performance (e.g., "Why is this matrix multiplication stalling on memory?").

### Usage


```shell
# nsight memory usage diagnosis command
nsys profile --trace=cuda,nvtx --cuda-warp-state ./my_cuda_program
```

This command captures a timeline of your application's execution to diagnose system-level behavior.

- **`nsys profile`**: The base command to start a profiling session.
- **`--trace=cuda,nvtx`**: Specifies which APIs to track on the timeline.
  - **`cuda`**: Tracks CUDA API calls (e.g., `cudaMalloc`, `cudaMemcpy`, kernel launches). This reveals how your CPU schedules work for the GPU.
  - **`nvtx`**: Tracks **NVIDIA Tools Extension** markers. If you manually annotated your code (e.g., `nvtxRangePush("MyFunction")`), these custom labels will appear on the timeline, making it easy to correlate code blocks with performance data.
- **`--cuda-warp-state`**: It periodically samples the GPU's warp schedulers to record why warps (threads) are stalled.
  - *Significance:* It helps diagnose if your GPU is waiting on **Memory** (latency), **Instruction Fetch**, or **Synchronization** without the overhead of a full kernel profile.
- **`./my_cuda_program`**: The executable to run.

**What you get:** A `.nsys-rep` file that you view in the Nsight Systems GUI.

```shell
# nsight computation insight cfommands
ncu -o OUT_NAME --target-processes all --set full -c 4 -f EXEC_NAME
ncu --import OUT_NAME.ncu-rep > OUT_NAME.analysis
```

This command performs a deep-dive analysis on the kernels running inside your application.

- **`ncu`**: The Nsight Compute CLI tool.
- **`-o OUT_NAME`**: Sets the output filename for the report (creates `OUT_NAME.ncu-rep`).
- **`--target-processes all`**: Tells the profiler to attach to all processes spawned by the application. This is essential if your application uses multiple processes (e.g., MPI) to drive GPUs.
- **`--set full`**: This collects **every available metric**.
  - *Warning:* This adds massive overhead (the kernel is replayed dozens of times) and takes a long time. It provides data on occupancy, cache hit rates, memory throughput, and instruction stalls.
- **`-c 4` (Launch Count):** Limits profiling to the **first 4 kernel launches** only.
  - *Why use this?* Since `--set full` is so slow, you don't want to profile a kernel that runs 10,000 times.
- **`-f`**: Forces overwrite if `OUT_NAME.ncu-rep` already exists.

### Metrics

These metrics provide the **context for the performance** but do not indicate efficiency by themselves:

- **DRAM / SM Frequency:** The clock speeds of your memory (VRAM) and Streaming Multiprocessors (cores) during execution.
- **Duration / Elapsed Cycles:** How long the kernel took to run.

When optimizing CUDA, you should prioritize the metrics in this order:

1. **DRAM Throughput:** The bandwidth used between the GPU chip and the global VRAM.

   If this is high, you are "Memory Bound." You need to reduce global memory accesses - improve memory coalescing or use Shared Memory.

2. **Compute (SM) Throughput:** 

   If this is high, you are "Compute Bound." You are doing heavy math. To go faster, you need to use faster instructions (e.g., Tensor Cores, FP16) or reduce the math required.

3. **L1/TEX Cache Throughput:** 

   If this is high, it means the ALUs are waiting on data from the L1 cache or Shared Memory. You need to improve your memory access.

---

## CUDA-GDB

```shell
# Shows your current focus (e.g., Block 0, Thread 10).
(cuda-gdb) cuda thread
# Switches focus to a specific thread in a specific block.
(cuda-gdb) cuda thread (0,1,0) block (2,0,0)

# List all kernels
(cuda-gdb) info cuda kernels
# List all blocks
(cuda-gdb) info cuda blocks
# List all threads
(cuda-gdb) info cuda threads

# To inspect blocks and threads, you need to stop the debugger inside the kernel code itself.
(cuda-gdb) break KernelFunction
(cuda-gdb) run
```

