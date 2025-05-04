#include "common.h"

#define COL_OUTPUT_SIZE 2
#define EMPTY_CELL std::numeric_limits<float>::max()

__global__ void mul_kernel(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                           COOMatrix *cooMatrix3, unsigned int numColM2) {
  int numLoops = (numColM2 + COL_OUTPUT_SIZE - 1) / COL_OUTPUT_SIZE;

  float rowOutput_s[COL_OUTPUT_SIZE];

  int global_index = blockDim.x * blockIdx.x + threadIdx.x;
  int rowIdxM1 = global_index;
  if (global_index < csrMatrix1->numRows) {
    int rowStartM1 = csrMatrix1->rowPtrs[rowIdxM1];
    int rowEndM1 = csrMatrix1->rowPtrs[rowIdxM1 + 1];

    for (int counterLoop = 0; counterLoop < numLoops; ++counterLoop) {
      int start_col = counterLoop * COL_OUTPUT_SIZE;
      int end_col = min(start_col + COL_OUTPUT_SIZE, numColM2);

      for (int i = 0; i < COL_OUTPUT_SIZE; ++i) {
        rowOutput_s[i] = 0.0f;
      }

      // Process each element in csrMatrix1's current row
      for (int elemIdxM1 = rowStartM1; elemIdxM1 < rowEndM1; ++elemIdxM1) {
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

      // Write non-zero values to COO
      for (int i = 0; i < COL_OUTPUT_SIZE; ++i) {
        if (fabsf(rowOutput_s[i]) > 1e-7f) {
          int global_col = start_col + i;
          if (global_col < numColM2) { // Ensure within matrix bounds
            int index = atomicAdd(&cooMatrix3->numNonzeros, 1);
            cooMatrix3->rowIdxs[index] = rowIdxM1;
            cooMatrix3->colIdxs[index] = global_col;
            cooMatrix3->values[index] = rowOutput_s[i];
          }
        }
      }
    }
  }
}

// function that launches the kernel
void spmspm_gpu0(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3, unsigned int numRows1,
                 unsigned int numRows2, unsigned int numCols2,
                 unsigned int numNonzeros1, unsigned int numNonzeros2) {

  int numThreadsPerBlock = 32;
  cudaMemset(&cooMatrix3->numNonzeros, 0, sizeof(int));

  // num rows of first matrix
  int numBlocks = (numRows1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
  mul_kernel<<<numBlocks, numThreadsPerBlock>>>(csrMatrix1, csrMatrix2,
                                                cooMatrix3, numCols2);
}
