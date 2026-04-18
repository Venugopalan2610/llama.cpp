#pragma once
#include "common.cuh"

// Flat dequant: outputs blocks sequentially (block 0 → positions 0-255, etc.)
// Use for row-by-row dequant where tiles happen to match row blocks.
template<typename dst_t>
void dequantize_row_qtip_2b_cuda(const void * vx, dst_t * y, int64_t k, cudaStream_t stream);

// Matrix-aware dequant: outputs to correct (row, col) positions in a row-major matrix.
// Tiles are stored in row_tile-major order; this kernel writes each tile's decoded
// values to their correct matrix positions: matrix[rt*16 + lr][ct*16 + lc].
// nrows = output_dim (m), ncols = input_dim (n).
template<typename dst_t>
void dequantize_matrix_qtip_2b_cuda(const void * vx, dst_t * y,
                                     int64_t nrows, int64_t ncols,
                                     cudaStream_t stream);

// Full QTIP dequant with inverse RHT: trellis decode + 2D FHT + sign vectors.
// Produces the actual weight matrix W from QTIP-compressed data.
// W = sign_l * H_m * W_tilde * H_n * sign_r / sqrt(m*n)
// where H_m, H_n are unscaled Walsh-Hadamard transforms.
// The function uses orthonormal FHT (scaled by 1/sqrt(dim)) internally,
// so two transforms give the 1/sqrt(m*n) factor automatically.
// sign_r: [ncols] float, sign_l: [nrows] float (can be nullptr to skip)
// Output W is row-major [nrows x ncols] float.
void dequantize_qtip_2b_full_rht_cuda(const void * vx, float * W,
                                       const float * sign_r,
                                       const float * sign_l,
                                       int64_t nrows, int64_t ncols,
                                       cudaStream_t stream);

// Activation-transform helpers for QTIP MoE path.
// Instead of applying 2D FHT to the full weight matrix, apply cheap 1D Hadamard
// transforms to the input/output activation vectors. This reduces the FHT work
// from O(mn log(mn)) to O(n log n + m log m) per token.
//
// Pre-transform:  x' = (1/√n) H_n diag(s_r) x   (apply to input before matmul)
// Post-transform: y  = diag(s_l) (1/√m) H_m y'   (apply to output after matmul)
//
// Data is column-major [dim, ncols] where ncols = tokens for this expert.
void qtip_pre_transform_cuda(float * x, const float * sign_r,
                              int n, int ncols, cudaStream_t stream);
void qtip_post_transform_cuda(float * y, const float * sign_l,
                               int m, int ncols, cudaStream_t stream);
