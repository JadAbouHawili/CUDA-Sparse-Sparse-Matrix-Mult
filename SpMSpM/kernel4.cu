#include "common.h"

#define COL_OUTPUT_SIZE 64 
#define WARP_SIZE 32

__global__ void mul_kernel_opt_4(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2,
                                COOMatrix* cooMatrix3, unsigned int numColM2) {
    const int numLoops = (numColM2 + COL_OUTPUT_SIZE - 1) / COL_OUTPUT_SIZE;
    const int WARPS_PER_BLOCK = blockDim.x / WARP_SIZE;

    __shared__ float rowOutput_s[COL_OUTPUT_SIZE];
    int rowIdxM1 = blockIdx.x;

    // Initialize shared memory using all threads
    for(int i = threadIdx.x; i < COL_OUTPUT_SIZE; i += blockDim.x) {
        rowOutput_s[i] = 0.0f;
    }
    __syncthreads();

    // Warp and lane indices
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    // Process row from csrMatrix1
    const int rowStartM1 = csrMatrix1->rowPtrs[rowIdxM1];
    const int rowEndM1 = csrMatrix1->rowPtrs[rowIdxM1 + 1];

    for (int counterLoop = 0; counterLoop < numLoops; ++counterLoop) {
        const int start_col = counterLoop * COL_OUTPUT_SIZE;
        const int end_col = min(start_col + COL_OUTPUT_SIZE, numColM2);

        // Process elements in csrMatrix1's row using warps
        for (int elemIdxM1 = rowStartM1 + warpId; elemIdxM1 < rowEndM1; elemIdxM1 += WARPS_PER_BLOCK) {
            const int colIndexM1 = csrMatrix1->colIdxs[elemIdxM1];
            const float valM1 = csrMatrix1->values[elemIdxM1];

            // Process corresponding row in csrMatrix2
            const int rowStartM2 = csrMatrix2->rowPtrs[colIndexM1];
            const int rowEndM2 = csrMatrix2->rowPtrs[colIndexM1 + 1];

            // Coalesced access using warp-sized chunks
            for (int elemIdxM2_base = rowStartM2; elemIdxM2_base < rowEndM2; elemIdxM2_base += WARP_SIZE) {
                const int elemIdxM2 = elemIdxM2_base + laneId;
                if (elemIdxM2 < rowEndM2) {
                    const int colIndexM2 = csrMatrix2->colIdxs[elemIdxM2];
                    const float valM2 = csrMatrix2->values[elemIdxM2];

                    if (colIndexM2 >= start_col && colIndexM2 < end_col) {
                        const int local_col = colIndexM2 - start_col;
                        atomicAdd(&rowOutput_s[local_col], valM1 * valM2);
                    }
                }
            }
        }
        __syncthreads();

        // Write non-zero results using all warps in the block
        for (int base_col = 0; base_col < COL_OUTPUT_SIZE; base_col += WARP_SIZE * WARPS_PER_BLOCK) {
            const int local_col = base_col + warpId * WARP_SIZE + laneId;
            if (local_col >= COL_OUTPUT_SIZE) break;

            const int global_col = start_col + local_col;
            const float value = rowOutput_s[local_col];
            const bool is_nonzero = (global_col < numColM2) && (fabsf(value) > 1e-7f);

            // Warp-level coordination
            const unsigned active_mask = __ballot_sync(0xFFFFFFFF, is_nonzero);
            if (active_mask != 0) {
                const int thread_offset = __popc(active_mask & ((1 << laneId) - 1));
                int warp_offset;

                if (laneId == 0) {
                    warp_offset = atomicAdd(&cooMatrix3->numNonzeros, __popc(active_mask));
                }
                warp_offset = __shfl_sync(active_mask, warp_offset, 0);

                if (is_nonzero) {
                    const int write_pos = warp_offset + thread_offset;
                    cooMatrix3->rowIdxs[write_pos] = rowIdxM1;
                    cooMatrix3->colIdxs[write_pos] = global_col;
                    cooMatrix3->values[write_pos] = value;
                }
            }
        }

        // Reset shared memory for next iteration
        for(int i = threadIdx.x; i < COL_OUTPUT_SIZE; i += blockDim.x) {
            rowOutput_s[i] = 0.0f;
        }
        __syncthreads();
    }
}

void spmspm_gpu4(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1,
                COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2,
                COOMatrix* cooMatrix3, unsigned int numRows1, unsigned int numRows2,
                unsigned int numCols2, unsigned int numNonzeros1, unsigned int numNonzeros2) {
    const int numThreadsPerBlock = 64;
    cudaMemset(&cooMatrix3->numNonzeros, 0, sizeof(int));
    mul_kernel_opt_4<<<numRows1, numThreadsPerBlock>>>(csrMatrix1, csrMatrix2, cooMatrix3, numCols2);
}

