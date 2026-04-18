// CUDA kernel for Walsh-Hadamard Transform on the last dimension.
// Used for QTIP activation pre/post-processing.

#include "common.cuh"

// One thread block per row. Threads cooperatively perform the butterfly.
// Supports dimensions up to 4096 (shared memory limit: 4096 * 4 = 16 KB).
// For larger dimensions, a multi-pass approach would be needed.

static __global__ void hadamard_f32_kernel(
    const float * __restrict__ src,
    float * __restrict__ dst,
    const int64_t ne0,     // transform dimension (power of 2)
    const int64_t n_rows
) {
    const int64_t row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float smem[];

    const float * src_row = src + row * ne0;
    float * dst_row = dst + row * ne0;

    // Load row into shared memory
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) {
        smem[i] = src_row[i];
    }
    __syncthreads();

    // Butterfly stages: log2(ne0) stages
    for (int64_t h = 1; h < ne0; h <<= 1) {
        for (int64_t i = threadIdx.x; i < ne0 / 2; i += blockDim.x) {
            // Map thread i to a butterfly pair
            const int64_t block_idx = i / h;
            const int64_t within = i % h;
            const int64_t j = block_idx * (h << 1) + within;

            float a = smem[j];
            float b = smem[j + h];
            smem[j]     = a + b;
            smem[j + h] = a - b;
        }
        __syncthreads();
    }

    // Scale by 1/sqrt(n) and write to output
    const float scale = rsqrtf((float)ne0);
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) {
        dst_row[i] = smem[i] * scale;
    }
}

void ggml_cuda_op_hadamard(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne0 = src0->ne[0];
    const int64_t n_rows = ggml_nrows(src0);

    GGML_ASSERT(ne0 > 0 && (ne0 & (ne0 - 1)) == 0); // power of 2
    GGML_ASSERT(ne0 <= 4096); // shared memory limit

    const float * src_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    // Use enough threads to cover ne0/2 butterflies, capped at 1024
    const int threads = (int)std::min((int64_t)1024, ne0 / 2);
    const int blocks = (int)n_rows;
    const size_t smem_size = ne0 * sizeof(float);

    hadamard_f32_kernel<<<blocks, threads, smem_size, stream>>>(src_d, dst_d, ne0, n_rows);
}
