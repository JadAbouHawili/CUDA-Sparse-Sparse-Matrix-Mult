#include "common.h"

#define COL_OUTPUT_SIZE 32 
#define WARP_SIZE 32 

__global__ void mul_kernel_opt_3(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, COOMatrix* cooMatrix3, unsigned int numColM2) {
    int numLoops = (numColM2 + COL_OUTPUT_SIZE - 1) / COL_OUTPUT_SIZE;
    const int WARPS_PER_BLOCK = blockDim.x / WARP_SIZE;

    __shared__ float rowOutput_s[COL_OUTPUT_SIZE];

    int rowIdxM1 = blockIdx.x;
    int rowStartM1 = csrMatrix1->rowPtrs[rowIdxM1];
    int rowEndM1 = csrMatrix1->rowPtrs[rowIdxM1 + 1];

    // Warp and lane indices
    int warpId = threadIdx.x / WARP_SIZE;  // Which warp in the block
    int laneId = threadIdx.x % WARP_SIZE;  // Thread index within the warp

    for (int counterLoop = 0; counterLoop < numLoops; ++counterLoop) {
        int start_col = counterLoop * COL_OUTPUT_SIZE;
        int end_col = min(start_col + COL_OUTPUT_SIZE, numColM2);

        // Initialize shared memory
        if (threadIdx.x < COL_OUTPUT_SIZE){
                rowOutput_s[threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Process elements in csrMatrix1's row using warps
        for (int elemIdxM1 = rowStartM1 + warpId; elemIdxM1 < rowEndM1; elemIdxM1 += WARPS_PER_BLOCK) {
            int colIndexM1 = csrMatrix1->colIdxs[elemIdxM1];
            float valM1 = csrMatrix1->values[elemIdxM1];

            int rowStartM2 = csrMatrix2->rowPtrs[colIndexM1];
            int rowEndM2 = csrMatrix2->rowPtrs[colIndexM1 + 1];

            // Process csrMatrix2's row in coalesced chunks using the entire warp
            for (int elemIdxM2_base = rowStartM2; elemIdxM2_base < rowEndM2; elemIdxM2_base += WARP_SIZE) {
                int elemIdxM2 = elemIdxM2_base + laneId;
                if (elemIdxM2 < rowEndM2) {
                    int colIndexM2 = csrMatrix2->colIdxs[elemIdxM2];
                    float valM2 = csrMatrix2->values[elemIdxM2];

                    if (colIndexM2 >= start_col && colIndexM2 < end_col) {
                        int local_col = colIndexM2 - start_col;
                        atomicAdd(&rowOutput_s[local_col], valM1 * valM2);
                    }
                }
            }
        }
        __syncthreads();

        if (threadIdx.x < WARP_SIZE) {
                const int global_col = start_col + threadIdx.x;
                bool is_nonzero = (global_col < numColM2) &&
                     (fabsf(rowOutput_s[threadIdx.x]) > 1e-7f);

                // Warp-level ballot for active mask
                unsigned active_mask = __ballot_sync(0xFFFFFFFF, is_nonzero);
                const int lane_id = threadIdx.x;

                if (active_mask != 0) {
                        // Warp-level prefix sum
                        int thread_offset = __popc(active_mask & ((1 << lane_id) - 1));

                        // Atomic add only once per warp
                        int warp_offset;
                        if (lane_id == 0) {
                                warp_offset = atomicAdd(&cooMatrix3->numNonzeros, __popc(active_mask));
                        }
                        warp_offset = __shfl_sync(active_mask, warp_offset, 0);

                        // Write if thread has non-zero
                        if (is_nonzero) {
                                const int write_pos = warp_offset + thread_offset;
                                cooMatrix3->rowIdxs[write_pos] = rowIdxM1;
                                cooMatrix3->colIdxs[write_pos] = global_col;
                                cooMatrix3->values[write_pos] = rowOutput_s[threadIdx.x];
                        }
                }
        }
        __syncthreads();


    }
}




void spmspm_gpu3(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1,
                 COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2,
                 COOMatrix* cooMatrix3, unsigned int numRows1, unsigned int numRows2,
                 unsigned int numCols2, unsigned int numNonzeros1, unsigned int numNonzeros2) {
    int numThreadsPerBlock = 64; 
    cudaMemset(&cooMatrix3->numNonzeros, 0, sizeof(int));
    mul_kernel_opt_3<<<numRows1, numThreadsPerBlock>>>(csrMatrix1, csrMatrix2, cooMatrix3, numCols2);
}






