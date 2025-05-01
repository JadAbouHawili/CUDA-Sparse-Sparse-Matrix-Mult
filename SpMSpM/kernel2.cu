
#include "common.h"

#define COL_OUTPUT_SIZE 512
#define EMPTY_CELL std::numeric_limits<float>::max()


__global__ void mul_kernel_opt_2(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, COOMatrix* cooMatrix3, unsigned int numColM2) {
    int numLoops = (numColM2 + COL_OUTPUT_SIZE - 1) / COL_OUTPUT_SIZE;

    __shared__ float rowOutput_s[COL_OUTPUT_SIZE];

    int rowIdxM1 = blockIdx.x;
    int rowStartM1 = csrMatrix1->rowPtrs[rowIdxM1];
    int rowEndM1 = csrMatrix1->rowPtrs[rowIdxM1 + 1];

    for (int counterLoop = 0; counterLoop < numLoops; ++counterLoop) {
        int start_col = counterLoop * COL_OUTPUT_SIZE;
        int end_col = min(start_col + COL_OUTPUT_SIZE, numColM2);

        // Initialize shared memory
        for (int i = threadIdx.x; i < COL_OUTPUT_SIZE; i += blockDim.x) {
            rowOutput_s[i] = 0.0f;
        }
        __syncthreads();

        // Process each element in csrMatrix1's current row
        for (int elemIdxM1 = rowStartM1 + threadIdx.x; elemIdxM1 < rowEndM1; elemIdxM1 += blockDim.x) {
            int colIndexM1 = csrMatrix1->colIdxs[elemIdxM1];
            float valM1 = csrMatrix1->values[elemIdxM1];

            int rowStartM2 = csrMatrix2->rowPtrs[colIndexM1];
            int rowEndM2 = csrMatrix2->rowPtrs[colIndexM1 + 1];

            for (int elemIdxM2 = rowStartM2; elemIdxM2 < rowEndM2; ++elemIdxM2) {
                int colIndexM2 = csrMatrix2->colIdxs[elemIdxM2];
                float valM2 = csrMatrix2->values[elemIdxM2];

                if (colIndexM2 >= start_col && colIndexM2 < end_col) {
                    int local_col = colIndexM2 - start_col;
                    atomicAdd(&rowOutput_s[local_col], valM1 * valM2);
                }
            }
        }
        __syncthreads();

        // Write non-zero values to COO
        for (int i = threadIdx.x; i < COL_OUTPUT_SIZE; i += blockDim.x) {
		if (fabsf(rowOutput_s[i]) > 1e-7f)  {
                int global_col = start_col + i;
                if (global_col < numColM2) { // Ensure within matrix bounds
                    int index = atomicAdd(&cooMatrix3->numNonzeros, 1);
                    cooMatrix3->rowIdxs[index] = rowIdxM1;
                    cooMatrix3->colIdxs[index] = global_col;
                    cooMatrix3->values[index] = rowOutput_s[i];
                }
            }
        }
        __syncthreads();
    }
}



// function that launches the kernel
void spmspm_gpu2(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3, unsigned int numRows1, unsigned int numRows2, unsigned int numCols2, unsigned int numNonzeros1, unsigned int numNonzeros2) {

    int numThreadsPerBlock = 512;
    cudaMemset(&cooMatrix3->numNonzeros, 0, sizeof(int));


    // num rows of first matrix
    int numBlocks = numRows1; 
    mul_kernel_opt_2<<<numBlocks,numThreadsPerBlock>>>(csrMatrix1, csrMatrix2, cooMatrix3, numCols2); 

}

