# Fundamentals of CUDA Programming

## The GPU Philosophy: Bandwidth over Latency

Think of a CPU as a Ferrari: it gets one or two people to a destination incredibly fast. A GPU is a massive fleet of buses: each bus is slower than the Ferrari, but together they can move an entire city's population at once.

## The Hierarchy of Computation

To manage these thousands of threads, CUDA organizes them into a clear hierarchy. You must memorize this structure to write any kernel:

- **Thread:** The smallest unit of work. Executes your "Kernel" function.
- **Block:** A group of threads. Threads in the same block can "talk" to each other via shared memory and synchronize.
- **Grid:** A collection of blocks that make up a single "Kernel" launch. Blocks within a grid are independent—they cannot reliably communicate.

![hierarchy](assets/hierarchy.jpg)

## The Programmer’s Workflow

1. **Allocate** memory on the CPU.
2. **Initialize** data on the CPU.
3. **Allocate** memory on the GPU (using `cudaMalloc`).
4. **Copy** data from CPU to GPU (`cudaMemcpy` with `HostToDevice`).
5. **Define** the execution configuration (how many blocks? how many threads?).
6. **Launch** the Kernel (the `<<<...>>>` syntax).
7. **Synchronize** (wait for the GPU to finish).
8. **Copy** results back to the CPU (`DeviceToHost`).
9. **Free** memory on both sides.

![hello_world](assets/hello_world.jpg)

*Note for step 5:*

A standard kernel launch looks like this:

```C++
KernelName<<<gridDim, blockDim, sharedMem, stream>>>(param1, param2, ...);
```

Calls to a kernel function are asynchronous.

While there are four possible arguments inside the brackets, you will almost always use the first two.

### 1. `gridDim` (Blocks per Grid)

The first parameter specifies the number of **Blocks** in the grid.

### 2. `blockDim` (Threads per Block)

The second parameter specifies the number of **Threads** within each block.

*Note for step 7:*

The `cudaDeviceSynchronize()` function is a **blocking call** that ensures all previously issued CUDA operations on the current GPU device are completed before the host (CPU) thread continues execution. 

*Note for GPU function in the example:*

For a simple 1D array, the formula for a thread's global index $i$ is:

$$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

- **`blockIdx.x`**: Which block am I in?
- **`blockDim.x`**: How many threads are in each block?
- **`threadIdx.x`**: Which thread am I within my specific block?