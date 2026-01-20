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

### Example B: 8-Byte Bank Width

Some newer architectures allow you to configure banks to be 8 bytes wide to better support `double` precision or larger data types.

- **Addr1 (Byte 6):** Now fits into **Bank 0**, which now covers bytes 0-7.
- **Addr2 (Byte 133):** Now fits into **Bank 16**.
- **The Result:** Because they are now in different banks, these two requests can be served **simultaneously** with zero conflict.

---

## Optimization Strategy 1: Shared Memory

Move data in shared memory, and assign each thread to its own bank

---

## Optimization Strategy 2: Memory Coalescing

When 32 threads in a warp access a contiguous 128-byte block of memory, the GPU can fulfill that with a **single memory transaction**.

---

## Optimization Strategy 3: Managing Occupancy (TLP vs. ILP)

**Thread-Level Parallelism (TLP):** If your kernel uses fewer registers per thread, you can fit more blocks on the SM. This "hides" memory latency by having more threads ready to work.

**Instruction-Level Parallelism (ILP):** Sometimes, using *more* registers per thread allows the compiler to break dependencies and execute more instructions in parallel within a single thread.

Example:

Assume we have a very long calculation ($1000 \times 999 \times 998 \times \dots \times 1$), but we only have a few registers to store our value. We will need to wait for the registers to be free before storing the intermediate results of the calculations.

---

## Optimization Strategy 4: Control Flow & Precision

Avoid `if/else` statements where threads in the same warp take different paths. This "divergence" forces the warp to execute both paths sequentially, idling half the threads.

Only use **Double Precision (FP64)** if absolutely necessary. On the RTX 2060, FP32 throughput is **32x faster** than FP64.
