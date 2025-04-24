
#include "common.h"

// assigning a threadblock per row
#include "common.h"
#include <iostream>
// #include <unordered_map>

#define BLOCK_DIM 256
#define temp_size 750

using namespace std;

__global__ void mul_kernel_opt(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                           COOMatrix *cooMatrix3) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int row = i / BLOCK_DIM;
  int num_nonzeros_row =
      csrMatrix1->rowPtrs[row + 1] - csrMatrix1->rowPtrs[row];

  __shared__ float temp[temp_size];
  if (threadIdx.x == 0) {
    for (int k = 0; k < temp_size; k++) {
      temp[k] = 0.0;
    }
  }
  __syncthreads();
  if (threadIdx.x < num_nonzeros_row) {

    // iterate over every row in matrix 2

    int col = csrMatrix1->colIdxs[csrMatrix1->rowPtrs[row] + threadIdx.x];
    float val = csrMatrix1->values[csrMatrix1->rowPtrs[row] + threadIdx.x];

    int row_start_2 = csrMatrix2->rowPtrs[col];
    int row_end_2 = csrMatrix2->rowPtrs[col + 1];

    for (int k = row_start_2; k < row_end_2; k++) {
      int col2 = csrMatrix2->colIdxs[k];
      float val2 = csrMatrix2->values[k];

      float store = val * val2;

      atomicAdd(&temp[col2], store);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 0; k < temp_size; k++) {
      if (temp[k] != 0) {
        int index = atomicAdd(&cooMatrix3->numNonzeros, 1);
        cooMatrix3->rowIdxs[index] = row;
        cooMatrix3->colIdxs[index] = k;
        cooMatrix3->values[index] = temp[k];
      }
    }
  }
}

void spmspm_gpu1(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3, unsigned int numRows1,
                 unsigned int numRows2, unsigned int numCols2,
                 unsigned int numNonzeros1, unsigned int numNonzeros2) {

  // CSR CSR

  int threadsPerBlock = BLOCK_DIM;
  int num_Blocks = numRows1;

  mul_kernel_opt<<<num_Blocks, threadsPerBlock>>>(csrMatrix1, csrMatrix2,
                                              cooMatrix3);
}
