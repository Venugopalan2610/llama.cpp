// CUDA dequantization kernel for QTIP 2-bit trellis-coded quantization.
//
// Each QTIP block encodes a 16×16 tile (256 floats) using a 16-bit trellis
// state walk + HYB codebook lookup (Klimov-Shamir hash → 512-entry LUT).
//
// Block layout (68 bytes):
//   - d:             fp16 scale factor (2 bytes)
//   - trellis_data:  [0..1] = initial 16-bit state (little-endian)
//                    [2..65] = 127 × 4-bit codes, bit-packed (508 bits, 64 bytes)
//
// Tile layout: blocks are stored in row_tile-major order.
// Block ib corresponds to tile (row_tile, col_tile) where:
//   row_tile = ib / n_col_tiles
//   col_tile = ib % n_col_tiles
// The 256 decoded values are in row-major order within the 16×16 tile.

#include "common.cuh"
#include "convert.cuh"
#include "qtip-dequant.cuh"

// QTIP HYB LUT: 512 entries × 2 floats, from K-means on 2D Gaussian.
// Stored in constant memory for fast broadcast across warps.
#include "../qtip_tlut.h"

static __constant__ float d_qtip_tlut[512][2];

static bool qtip_lut_initialized = false;

static void ensure_qtip_lut(cudaStream_t stream) {
    if (qtip_lut_initialized) return;
    CUDA_CHECK(cudaMemcpyToSymbol(d_qtip_tlut, qtip_tlut, sizeof(qtip_tlut)));
    qtip_lut_initialized = true;
}

// Helper: run trellis walk for one block and write decoded values to an output
// buffer at a given stride pattern. Used by both flat and matrix-aware kernels.
template<typename dst_t>
static __device__ void qtip_decode_block(
    const block_qtip_2b * block,
    dst_t * __restrict__ out,
    const int64_t stride  // distance between consecutive decoded values in output
) {
    const float scale = __half2float(block->d);
    const uint8_t * td = block->trellis_data;

    // Step 0: read initial 16-bit state (little-endian, first code pre-applied)
    uint32_t state = (uint32_t)td[0] | ((uint32_t)td[1] << 8);

    // Decode first 2 values from initial state
    uint32_t x = state * (state + 1);
    int idx = (x >> 6) & 511;
    int sflp = 1 - ((x >> 15) & 1) * 2;
    out[0 * stride] = ggml_cuda_cast<dst_t>(d_qtip_tlut[idx][0] * scale);
    out[1 * stride] = ggml_cuda_cast<dst_t>(d_qtip_tlut[idx][1] * sflp * scale);

    // Decode remaining 127 steps
    for (int j = 1; j < 128; j++) {
        const int bit_pos = 16 + (j - 1) * 4;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;

        uint16_t two_bytes;
        if (byte_idx + 1 < 66) {
            two_bytes = (uint16_t)td[byte_idx] | ((uint16_t)td[byte_idx + 1] << 8);
        } else {
            two_bytes = (uint16_t)td[byte_idx];
        }
        uint32_t new_bits = (two_bytes >> bit_off) & 0xF;

        state = ((state << 4) | new_bits) & 0xFFFF;

        x = state * (state + 1);
        idx = (x >> 6) & 511;
        sflp = 1 - ((x >> 15) & 1) * 2;

        out[(j * 2)     * stride] = ggml_cuda_cast<dst_t>(d_qtip_tlut[idx][0] * scale);
        out[(j * 2 + 1) * stride] = ggml_cuda_cast<dst_t>(d_qtip_tlut[idx][1] * sflp * scale);
    }
}

// Flat dequant: one thread per block, output sequential.
template<typename dst_t>
static __global__ void dequantize_block_qtip_2b_kernel(
    const void * __restrict__ vx,
    dst_t * __restrict__ y,
    const int64_t num_blocks
) {
    const int64_t ib = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= num_blocks) return;

    const block_qtip_2b * block = ((const block_qtip_2b *) vx) + ib;
    dst_t * out = y + ib * QTIP_BLOCK_SIZE;
    qtip_decode_block(block, out, (int64_t)1);
}

// Matrix-aware dequant: one thread per tile, writes to correct matrix positions.
// Tiles are in row_tile-major order. Output is row-major matrix [nrows × ncols].
template<typename dst_t>
static __global__ void dequantize_matrix_qtip_2b_kernel(
    const void * __restrict__ vx,
    dst_t * __restrict__ y,
    const int64_t num_blocks,
    const int64_t ncols,        // = input_dim = n
    const int64_t n_col_tiles   // = ncols / 16
) {
    const int64_t ib = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= num_blocks) return;

    const block_qtip_2b * block = ((const block_qtip_2b *) vx) + ib;

    const int64_t row_tile = ib / n_col_tiles;
    const int64_t col_tile = ib % n_col_tiles;

    const float scale = __half2float(block->d);
    const uint8_t * td = block->trellis_data;

    // Step 0: initial state
    uint32_t state = (uint32_t)td[0] | ((uint32_t)td[1] << 8);

    uint32_t x = state * (state + 1);
    int idx = (x >> 6) & 511;
    int sflp = 1 - ((x >> 15) & 1) * 2;

    // Value 0: tile position (0, 0) → matrix (row_tile*16 + 0, col_tile*16 + 0)
    int64_t mat_pos0 = (row_tile * 16 + 0) * ncols + col_tile * 16 + 0;
    y[mat_pos0] = ggml_cuda_cast<dst_t>(d_qtip_tlut[idx][0] * scale);
    // Value 1: tile position (0, 1) → matrix (row_tile*16 + 0, col_tile*16 + 1)
    int64_t mat_pos1 = (row_tile * 16 + 0) * ncols + col_tile * 16 + 1;
    y[mat_pos1] = ggml_cuda_cast<dst_t>(d_qtip_tlut[idx][1] * sflp * scale);

    // Steps 1..127
    for (int j = 1; j < 128; j++) {
        const int bit_pos = 16 + (j - 1) * 4;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;

        uint16_t two_bytes;
        if (byte_idx + 1 < 66) {
            two_bytes = (uint16_t)td[byte_idx] | ((uint16_t)td[byte_idx + 1] << 8);
        } else {
            two_bytes = (uint16_t)td[byte_idx];
        }
        uint32_t new_bits = (two_bytes >> bit_off) & 0xF;
        state = ((state << 4) | new_bits) & 0xFFFF;

        x = state * (state + 1);
        idx = (x >> 6) & 511;
        sflp = 1 - ((x >> 15) & 1) * 2;

        // Tile position of value j*2: row_in_tile = (j*2)/16, col_in_tile = (j*2)%16
        const int val0_idx = j * 2;
        const int val1_idx = j * 2 + 1;
        const int64_t r0 = row_tile * 16 + val0_idx / 16;
        const int64_t c0 = col_tile * 16 + val0_idx % 16;
        const int64_t r1 = row_tile * 16 + val1_idx / 16;
        const int64_t c1 = col_tile * 16 + val1_idx % 16;

        y[r0 * ncols + c0] = ggml_cuda_cast<dst_t>(d_qtip_tlut[idx][0] * scale);
        y[r1 * ncols + c1] = ggml_cuda_cast<dst_t>(d_qtip_tlut[idx][1] * sflp * scale);
    }
}

// Flat dequant (sequential output)
template<typename dst_t>
void dequantize_row_qtip_2b_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QTIP_BLOCK_SIZE == 0);
    ensure_qtip_lut(stream);

    const int64_t num_blocks = k / QTIP_BLOCK_SIZE;
    const int threads = 128;
    const int blocks = (int)((num_blocks + threads - 1) / threads);

    dequantize_block_qtip_2b_kernel<dst_t><<<blocks, threads, 0, stream>>>(vx, y, num_blocks);
}

// Matrix-aware dequant (correct row-major output)
template<typename dst_t>
void dequantize_matrix_qtip_2b_cuda(const void * vx, dst_t * y,
                                     const int64_t nrows, const int64_t ncols,
                                     cudaStream_t stream) {
    GGML_ASSERT(ncols % 16 == 0);
    GGML_ASSERT(nrows % 16 == 0);
    ensure_qtip_lut(stream);

    const int64_t n_col_tiles = ncols / 16;
    const int64_t num_blocks = (nrows / 16) * n_col_tiles;
    const int threads = 128;
    const int blocks = (int)((num_blocks + threads - 1) / threads);

    dequantize_matrix_qtip_2b_kernel<dst_t><<<blocks, threads, 0, stream>>>(
        vx, y, num_blocks, ncols, n_col_tiles);
}

// ========================================================================
// Inverse RHT kernels for full QTIP dequant
// ========================================================================

// Row-wise in-place FHT: one thread block per row, butterfly in shared memory.
// Applies orthonormal FHT (scaled by 1/sqrt(ne0)).
static __global__ void fht_rows_kernel(
    float * __restrict__ W,
    const int64_t nrows,
    const int64_t ncols
) {
    const int64_t row = blockIdx.x;
    if (row >= nrows) return;

    extern __shared__ float smem[];

    float * row_ptr = W + row * ncols;

    // Load row into shared memory
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        smem[i] = row_ptr[i];
    }
    __syncthreads();

    // Butterfly stages
    for (int64_t h = 1; h < ncols; h <<= 1) {
        for (int64_t i = threadIdx.x; i < ncols / 2; i += blockDim.x) {
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

    // Write back (no scaling — raw FHT)
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        row_ptr[i] = smem[i];
    }
}

// Column-wise in-place FHT: one thread block per column, strided access.
static __global__ void fht_cols_kernel(
    float * __restrict__ W,
    const int64_t nrows,
    const int64_t ncols
) {
    const int64_t col = blockIdx.x;
    if (col >= ncols) return;

    extern __shared__ float smem[];

    // Load column into shared memory (strided read)
    for (int i = threadIdx.x; i < nrows; i += blockDim.x) {
        smem[i] = W[i * ncols + col];
    }
    __syncthreads();

    // Butterfly stages
    for (int64_t h = 1; h < nrows; h <<= 1) {
        for (int64_t i = threadIdx.x; i < nrows / 2; i += blockDim.x) {
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

    // Write back (strided write)
    for (int i = threadIdx.x; i < nrows; i += blockDim.x) {
        W[i * ncols + col] = smem[i];
    }
}

// Element-wise multiply columns by sign_r and scale by 1/sqrt(m*n).
// Combined with sign_r application for efficiency.
static __global__ void apply_sign_r_and_scale_kernel(
    float * __restrict__ W,
    const float * __restrict__ sign_r,
    const int64_t nrows,
    const int64_t ncols,
    const float scale
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrows * ncols) return;
    const int64_t col = idx % ncols;
    W[idx] *= sign_r[col] * scale;
}

// Element-wise multiply rows by sign_l.
static __global__ void apply_sign_l_kernel(
    float * __restrict__ W,
    const float * __restrict__ sign_l,
    const int64_t nrows,
    const int64_t ncols
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrows * ncols) return;
    const int64_t row = idx / ncols;
    W[idx] *= sign_l[row];
}

// Full QTIP dequant with inverse RHT.
// W = sign_l * H_m * W_tilde * H_n * sign_r / sqrt(m*n)
// Steps:
//   1. Trellis decode W_tilde → row-major float [nrows × ncols]
//   2. FHT along rows (H_n on each row, unscaled)
//   3. Multiply columns by sign_r, scale by 1/sqrt(m*n)
//   4. FHT along columns (H_m on each column, unscaled)
//   5. Multiply rows by sign_l
void dequantize_qtip_2b_full_rht_cuda(const void * vx, float * W,
                                       const float * sign_r,
                                       const float * sign_l,
                                       const int64_t nrows, const int64_t ncols,
                                       cudaStream_t stream) {
    GGML_ASSERT(ncols % 16 == 0);
    GGML_ASSERT(nrows % 16 == 0);
    GGML_ASSERT(ncols <= 4096 && "FHT row dimension exceeds shared memory limit");
    GGML_ASSERT(nrows <= 4096 && "FHT col dimension exceeds shared memory limit");

    // Step 1: trellis decode to float matrix
    dequantize_matrix_qtip_2b_cuda<float>(vx, W, nrows, ncols, stream);

    // Step 2: FHT along rows (each row of length ncols)
    {
        const int threads = (int)std::min((int64_t)1024, ncols / 2);
        const size_t smem = ncols * sizeof(float);
        fht_rows_kernel<<<(int)nrows, threads, smem, stream>>>(W, nrows, ncols);
    }

    // Step 3: multiply columns by sign_r and scale by 1/sqrt(m*n)
    if (sign_r) {
        const float scale = 1.0f / sqrtf((float)(nrows * ncols));
        const int64_t total = nrows * ncols;
        const int threads = 256;
        const int blocks = (int)((total + threads - 1) / threads);
        apply_sign_r_and_scale_kernel<<<blocks, threads, 0, stream>>>(
            W, sign_r, nrows, ncols, scale);
    }

    // Step 4: FHT along columns (each column of length nrows)
    {
        const int threads = (int)std::min((int64_t)1024, nrows / 2);
        const size_t smem = nrows * sizeof(float);
        fht_cols_kernel<<<(int)ncols, threads, smem, stream>>>(W, nrows, ncols);
    }

    // Step 5: multiply rows by sign_l
    if (sign_l) {
        const int64_t total = nrows * ncols;
        const int threads = 256;
        const int blocks = (int)((total + threads - 1) / threads);
        apply_sign_l_kernel<<<blocks, threads, 0, stream>>>(
            W, sign_l, nrows, ncols);
    }
}

// ========================================================================
// Vector Hadamard + sign kernels for activation-transform MoE path
// ========================================================================

// In-place Hadamard on each column of a column-major matrix.
// data layout: data[row + col * n], n = vector length (power of 2).
// Applies 1/sqrt(n) scaling. One thread block per column.
static __global__ void hadamard_vector_kernel(
    float * __restrict__ data,
    const int n,       // vector length (rows), must be power of 2
    const int ncols    // number of columns
) {
    const int col = blockIdx.x;
    if (col >= ncols) return;

    extern __shared__ float smem[];

    float * col_ptr = data + (int64_t)col * n;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        smem[i] = col_ptr[i];
    }
    __syncthreads();

    for (int h = 1; h < n; h <<= 1) {
        for (int i = threadIdx.x; i < n / 2; i += blockDim.x) {
            const int block_idx = i / h;
            const int within    = i % h;
            const int j = block_idx * (h << 1) + within;
            float a = smem[j];
            float b = smem[j + h];
            smem[j]     = a + b;
            smem[j + h] = a - b;
        }
        __syncthreads();
    }

    const float scale = rsqrtf((float)n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        col_ptr[i] = smem[i] * scale;
    }
}

// Element-wise multiply: data[i + col*n] *= sign[i] for all (i, col).
static __global__ void sign_multiply_kernel(
    float * __restrict__ data,
    const float * __restrict__ sign,
    const int n,
    const int ncols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * ncols;
    if (idx >= total) return;
    const int row = idx % n;
    data[idx] *= sign[row];
}

// Pre-transform for QTIP MoE activation path:
//   x' = (1/sqrt(n)) * H_n * diag(sign_r) * x
// x is column-major [n, ncols]. Modifies x in place.
void qtip_pre_transform_cuda(float * x, const float * sign_r,
                              const int n, const int ncols,
                              cudaStream_t stream) {
    const int total = n * ncols;

    // Step 1: x *= sign_r
    if (sign_r) {
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        sign_multiply_kernel<<<blocks, threads, 0, stream>>>(x, sign_r, n, ncols);
    }

    // Step 2: Hadamard on each column with 1/sqrt(n) scaling
    {
        const int threads = (int)std::min(1024, n / 2);
        const size_t smem = n * sizeof(float);
        hadamard_vector_kernel<<<ncols, threads, smem, stream>>>(x, n, ncols);
    }
}

// Post-transform for QTIP MoE activation path:
//   y = diag(sign_l) * (1/sqrt(m)) * H_m * y'
// y is column-major [m, ncols]. Modifies y in place.
void qtip_post_transform_cuda(float * y, const float * sign_l,
                               const int m, const int ncols,
                               cudaStream_t stream) {
    // Step 1: Hadamard on each column with 1/sqrt(m) scaling
    {
        const int threads = (int)std::min(1024, m / 2);
        const size_t smem = m * sizeof(float);
        hadamard_vector_kernel<<<ncols, threads, smem, stream>>>(y, m, ncols);
    }

    // Step 2: y *= sign_l
    if (sign_l) {
        const int total = m * ncols;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        sign_multiply_kernel<<<blocks, threads, 0, stream>>>(y, sign_l, m, ncols);
    }
}

// Explicit instantiations
template void dequantize_row_qtip_2b_cuda<float>(const void * vx, float * y, const int64_t k, cudaStream_t stream);
template void dequantize_row_qtip_2b_cuda<half>(const void * vx, half * y, const int64_t k, cudaStream_t stream);

template void dequantize_matrix_qtip_2b_cuda<float>(const void * vx, float * y, int64_t nrows, int64_t ncols, cudaStream_t stream);
template void dequantize_matrix_qtip_2b_cuda<half>(const void * vx, half * y, int64_t nrows, int64_t ncols, cudaStream_t stream);
