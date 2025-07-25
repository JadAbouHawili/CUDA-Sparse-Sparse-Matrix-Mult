
//__global__ void calculate_kernel(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1,
//                 CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2,
//                 CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2,
//                 COOMatrix *cooMatrix3, unsigned int numRows1,
//                 unsigned int numRows2, unsigned int numCols2,
//                 unsigned int numNonzeros1, unsigned int numNonzeros2,
//                 unordered_map<int, float> result) {
//
//  int numColumnsOutputMatrix = cooMatrix3->numCols;
//  int i = blockDim.x * blockIdx.x + threadIdx.x;
//  if (i < cooMatrix1->numNonzeros) {
//    int rowNumber = cooMatrix1->rowIdxs[i];
//    int colNumber = cooMatrix1->colIdxs[i];
//    float val = cooMatrix1->values[i];
//    int csrStart = csrMatrix2->rowPtrs[colNumber];
//    int csrEnd = csrMatrix2->rowPtrs[colNumber + 1];
//    for (int j = csrStart; j < csrEnd; ++j) {
//      int colNumber2 = csrMatrix2->colIdxs[j];
//      float val2 = csrMatrix2->values[j];
//      int oneToOneIndex = rowNumber * numColumnsOutputMatrix + colNumber2;
//      result[oneToOneIndex] += val * val2;
//    }
//  }
//
//}
//
//__global__ void populate_output(unordered_map<int, float> result, COOMatrix
//*cooMatrix3){
//
//  int numColumnsOutputMatrix = cooMatrix3->numCols;
//  // synchronize across all blocks
//  for (auto &p : result) {
//    int index = cooMatrix3->numNonzeros++;
//    cooMatrix3->rowIdxs[index] = p.first / numColumnsOutputMatrix;
//    cooMatrix3->colIdxs[index] = p.first % numColumnsOutputMatrix;
//    cooMatrix3->values[index] = p.second;
//  }
//}
void spmspm_gpu0(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3, unsigned int numRows1,
                 unsigned int numRows2, unsigned int numCols2,
                 unsigned int numNonzeros1, unsigned int numNonzeros2) {
  //// call kernels here and nothing else
  //  // matrix 1 as COO cooMatrix1
  //  // matrix 2 as CSR csrMatrix2
  //  unordered_map<int, float> result;
  //	int numThreadsPerBlock = 128;
  //	int numBlocks = (cooMatrix1->numNonzeros + numThreadsPerBlock - 1)/
  // numThreadsPerBlock;
  //	calculate_kernel<<<numBlocks,numThreadsPerBlock>>>(cooMatrix1,
  // csrMatrix1,
  //                 cscMatrix1, cooMatrix2,
  //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// parallel version, false assumption that cols of matrix 2 is bounded i.e
// fixed number on shared memory that would be enough to store the
// contributions when computing the output row
// usage of dynamic memory based on cols of second matrix, eventually run out...
/*
#include "common.h"

// assigning a threadblock per row
#include "common.h"
#include <iostream>
  // #include <unordered_map>

#define BLOCK_DIM 256
  // #define temp_size (750 / 2)

  using namespace std;

  __global__ void mul_kernel_opt(CSRMatrix * csrMatrix1, CSRMatrix *
csrMatrix2, COOMatrix * cooMatrix3) { int i = blockDim.x * blockIdx.x +
threadIdx.x; int row = i / BLOCK_DIM; int num_nonzeros_row =
        csrMatrix1->rowPtrs[row + 1] - csrMatrix1->rowPtrs[row];

    extern __shared__ float temp[];
    //__shared__ float temp[temp_size - 1];
    //__shared__ int temp_indices[temp_size - 1];
    //__shared__ int incr = 0;
    if (threadIdx.x == 0) {
      for (int k = 0; k < csrMatrix2->numCols; k++) {
        temp[k] = 0.0;
      }
    }
    __syncthreads();
    // if (threadIdx.x < num_nonzeros_row) {
    for (int iter = threadIdx.x; iter < num_nonzeros_row; iter += BLOCK_DIM) {

      // iterate over every row in matrix 2

      int col = csrMatrix1->colIdxs[csrMatrix1->rowPtrs[row] + iter];
      float val = csrMatrix1->values[csrMatrix1->rowPtrs[row] + iter];

      int row_start_2 = csrMatrix2->rowPtrs[col];
      int row_end_2 = csrMatrix2->rowPtrs[col + 1];

      for (int k = row_start_2; k < row_end_2; k++) {
        int col2 = csrMatrix2->colIdxs[k];
        float val2 = csrMatrix2->values[k];

        float store = val * val2;
        // col 2 should be indexed on consecutive locations and then when
        // looping over it accessing another array with the col indices (more
        // shared memory?)
        //
        // might lead to illegal memory access
        // int index = atomicAdd(&incr, 1);
        // temp_indices[index] = col2;
        atomicAdd(&temp[col2], store);
        //
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      for (int k = 0; k < csrMatrix2->numCols; k++) {
        if (temp[k] != 0) {
          int index = atomicAdd(&cooMatrix3->numNonzeros, 1);
          cooMatrix3->rowIdxs[index] = row;
          // must change this
          cooMatrix3->colIdxs[index] = k;
          cooMatrix3->values[index] = temp[k];
        }
      }
    }
  }

  void spmspm_gpu1(COOMatrix * cooMatrix1, CSRMatrix * csrMatrix1,
                   CSCMatrix * cscMatrix1, COOMatrix * cooMatrix2,
                   CSRMatrix * csrMatrix2, CSCMatrix * cscMatrix2,
                   COOMatrix * cooMatrix3, unsigned int numRows1,
                   unsigned int numRows2, unsigned int numCols2,
                   unsigned int numNonzeros1, unsigned int numNonzeros2) {

    // CSR CSR

    int threadsPerBlock = BLOCK_DIM;
    int num_Blocks = numRows1;

    mul_kernel_opt<<<num_Blocks, threadsPerBlock, numCols2>>>(
        csrMatrix1, csrMatrix2, cooMatrix3);
  }
*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
// assigning a thread per row , fixed shared memory size to 32. doesnt work on
// bigger matrices.
#include "common.h"
#include <iostream>
// #include <unordered_map>

#define BLOCK_DIM 256
#define temp_size 32

using namespace std;

__global__ void mul_kernel(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2,
                           COOMatrix *cooMatrix3) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // thread per row
  if (i < csrMatrix1->numRows) {
    int row_start = csrMatrix1->rowPtrs[i];
    int row_end = csrMatrix1->rowPtrs[i + 1];

    // temporary storage
    float temp[temp_size];
    for (int k = 0; k < temp_size; k++) {
      temp[k] = 0;
    }

    // iterate over every row in matrix 2

    for (int j = row_start; j < row_end; ++j) {
      int col = csrMatrix1->colIdxs[j];
      float val = csrMatrix1->values[j];

      int row_start_2 = csrMatrix2->rowPtrs[col];
      int row_end_2 = csrMatrix2->rowPtrs[col + 1];

      for (int k = row_start_2; k < row_end_2; k++) {
        int col2 = csrMatrix2->colIdxs[k];
        float val2 = csrMatrix2->values[k];

        float store = val * val2;

        temp[col2] += store;
      }
    }
    for (int k = 0; k < temp_size; k++) {
      if (temp[k] != 0) {
        int index = atomicAdd(&cooMatrix3->numNonzeros, 1);
        cooMatrix3->rowIdxs[index] = i;
        cooMatrix3->colIdxs[index] = k;
        cooMatrix3->values[index] = temp[k];
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
  // CSR CSR

  int threadsPerBlock = BLOCK_DIM;
  int num_Blocks = (numRows1 + threadsPerBlock - 1) / threadsPerBlock;

  mul_kernel<<<num_Blocks, threadsPerBlock>>>(csrMatrix1, csrMatrix2,
                                              cooMatrix3);
}
*/
