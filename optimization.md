# Optimization

---

## Shared Memory Addressing

This part shows how **Shared Memory** is physically organized to allow multiple threads to access data simultaneously without crashing into each other.

### The Core Mechanics: Address Interleaving

Shared memory doesn't store data in one long line. Instead, it "interleaves" data across the 32 banks at 32-bit (4-byte) granularity.

- **Bank 0** holds bytes 0-3.
- **Bank 1** holds bytes 4-7.
- **Bank 31** holds bytes 124-127.
- **Bank 0 (next row)** then holds bytes 128-131.

![](assets/bank.jpg)

### Example A: 4-Byte Bank Width

In this mode, each bank is exactly 4 bytes wide. Look at the binary addresses in the image:

- **Addr1 (Byte 6):** The bits show it belongs to **Bank 1** because it falls in the 4-7 byte range.
- **Addr2 (Byte 133):** The bits show it also belongs to **Bank 1**, but in the *second row* (bytes 132-135).

*Note: If Thread A wants Byte 6 and Thread B wants Byte 133, they are hitting the **same bank** (Bank 1). This causes a **bank conflict**, and the hardware must process them sequentially.*

---

## Optimization Strategy 1: Shared Memory

Move data in shared memory, and assign each thread to its own bank

*Example:*

Original code:

<img src="assets/origin.jpg" style="zoom:50%;" />

Updated code:

<img src="assets/updated.jpg" style="zoom:50%;" />

Why updated code is better:

1. The updated code access global memory in contiguous chunks.
2. When modifying value, we do it in shared memory, which is way faster.

<img src="assets/exp.jpg" style="zoom:50%;" />

---

## Optimization Strategy 2: Memory Coalescing

When 32 threads in a warp access a contiguous **128-byte** block of memory, the GPU can fulfill that with a **single memory transaction**.

In practice, here is how you achieve and verify memory coalescing.

1. Ensure your global index maps directly to your array index without gaps.

   ```c++
   // Coalesced Access
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   float val = data[tid];
   
   // Non-coalesced Access (uses stride 4 in this case)
   int tid = blockIdx.x * blockDim.x + threadIdx.x * 4;
   float val = data[tid];
   ```

2. Use Structure of Arrays instead of Array of Structures

   **AoS (Bad for Coalescing):** `struct Particle { float x, y, z; } particles[N];`

   - Thread 0 reads `particles[0].x`, Thread 1 reads `particles[1].x`.
   - The addresses are separated by the size of the struct (3 floats), creating a stride.

   **SoA (Good for Coalescing):** `struct Particles { float x[N], y[N], z[N]; }`

   - Thread 0 reads `x[0]`, Thread 1 reads `x[1]`.
   - The addresses are perfectly contiguous.

---

## Optimization Strategy 3: Control Flow

Avoid `if/else` statements where threads in the same warp take different paths. This "divergence" forces the warp to execute both paths sequentially, idling half the threads.

*Example of why divergence is not desirable:*

![](assets/bad_example.jpg)

---

## Optimization Strategy 4: Managing Occupancy

**Occupancy** is the ratio of active warps on a Streaming Multiprocessor (SM) to the maximum number of warps that SM can theoretically support. For example, if an SM can hold 64 warps but only 32 are active, occupancy is 50%.

**If your arithmetic intensity is low** enough that stalls happens often, try to have more active wraps on a SM.

---

## Optimization Strategy 5: Avoiding Bank Conflicts

Each thread in a wrap should **only access one address**.
