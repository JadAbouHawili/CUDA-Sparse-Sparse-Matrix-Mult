#include "common.h"

#define COL_OUTPUT_SIZE 64

__global__ void mul_kernel(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                           COOMatrix *cooMatrix3, unsigned int numColM2) {
  int rowIdxM1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (rowIdxM1 >= csrMatrix1->numRows)
    return; // Guard clause

  int rowStartM1 = csrMatrix1->rowPtrs[rowIdxM1];
  int rowEndM1 = csrMatrix1->rowPtrs[rowIdxM1 + 1];

  // Each thread processes its own row entirely
  float rowOutput[COL_OUTPUT_SIZE];

  // Loop over COL_OUTPUT_SIZE-sized chunks of columns
  for (int start_col = 0; start_col < numColM2; start_col += COL_OUTPUT_SIZE) {
    int end_col = min(start_col + COL_OUTPUT_SIZE, numColM2);

    // Reset local accumulator
    for (int i = 0; i < COL_OUTPUT_SIZE; ++i) {
      rowOutput[i] = 0.0f;
    }

    // Compute partial products for this column chunk
    for (int elemIdxM1 = rowStartM1; elemIdxM1 < rowEndM1; ++elemIdxM1) {
      int colIndexM1 = csrMatrix1->colIdxs[elemIdxM1];
      float valM1 = csrMatrix1->values[elemIdxM1];

      int rowStartM2 = csrMatrix2->rowPtrs[colIndexM1];
      int rowEndM2 = csrMatrix2->rowPtrs[colIndexM1 + 1];

      for (int elemIdxM2 = rowStartM2; elemIdxM2 < rowEndM2; ++elemIdxM2) {
        int colIndexM2 = csrMatrix2->colIdxs[elemIdxM2];
        float valM2 = csrMatrix2->values[elemIdxM2];

        if (colIndexM2 >= start_col && colIndexM2 < end_col) {
          rowOutput[colIndexM2 - start_col] += valM1 * valM2;
        }
      }
    }

    // Write non-zero results to COO matrix
    for (int i = 0; i < end_col - start_col; ++i) {
      if (fabsf(rowOutput[i]) > 1e-7f) {
        int global_col = start_col + i;
        int index = atomicAdd(&cooMatrix3->numNonzeros, 1);
        cooMatrix3->rowIdxs[index] = rowIdxM1;
        cooMatrix3->colIdxs[index] = global_col;
        cooMatrix3->values[index] = rowOutput[i];
      }
    }
  }
}

void spmspm_gpu0(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3, unsigned int numRows1,
                 unsigned int numRows2, unsigned int numCols2,
                 unsigned int numNonzeros1, unsigned int numNonzeros2) {

  int numThreadsPerBlock = 64;
  cudaMemset(&cooMatrix3->numNonzeros, 0, sizeof(int));
  int numBlocks = (numRows1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
  mul_kernel<<<numBlocks, numThreadsPerBlock>>>(csrMatrix1, csrMatrix2,
                                                cooMatrix3, numCols2);
}


