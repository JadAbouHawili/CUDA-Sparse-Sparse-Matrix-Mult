
#include "common.h"
#include <unordered_map>
using namespace std;

__global__ void calculate_kernel(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3, unsigned int numRows1,
                 unsigned int numRows2, unsigned int numCols2,
                 unsigned int numNonzeros1, unsigned int numNonzeros2, unordered_map<int, float> result) {
	
  int numColumnsOutputMatrix = cooMatrix3->numCols;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < cooMatrix1->numNonzeros) {
    int rowNumber = cooMatrix1->rowIdxs[i];
    int colNumber = cooMatrix1->colIdxs[i];
    float val = cooMatrix1->values[i];
    int csrStart = csrMatrix2->rowPtrs[colNumber];
    int csrEnd = csrMatrix2->rowPtrs[colNumber + 1];
    for (int j = csrStart; j < csrEnd; ++j) {
      int colNumber2 = csrMatrix2->colIdxs[j];
      float val2 = csrMatrix2->values[j];
      int oneToOneIndex = rowNumber * numColumnsOutputMatrix + colNumber2;
      result[oneToOneIndex] += val * val2;
    }
  }

}

__global__ void populate_output(unordered_map<int, float> result, COOMatrix *cooMatrix3){

  int numColumnsOutputMatrix = cooMatrix3->numCols;
  // synchronize across all blocks
  for (auto &p : result) {
    int index = cooMatrix3->numNonzeros++;
    cooMatrix3->rowIdxs[index] = p.first / numColumnsOutputMatrix;
    cooMatrix3->colIdxs[index] = p.first % numColumnsOutputMatrix;
    cooMatrix3->values[index] = p.second;
  }
}
void spmspm_gpu0(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3, unsigned int numRows1,
                 unsigned int numRows2, unsigned int numCols2,
                 unsigned int numNonzeros1, unsigned int numNonzeros2) {
// call kernels here and nothing else
  // matrix 1 as COO cooMatrix1
  // matrix 2 as CSR csrMatrix2
  unordered_map<int, float> result;
	int numThreadsPerBlock = 128;
	int numBlocks = (cooMatrix1->numNonzeros + numThreadsPerBlock - 1)/ numThreadsPerBlock;
	calculate_kernel<<<numBlocks,numThreadsPerBlock>>>(cooMatrix1, csrMatrix1,
                 cscMatrix1, cooMatrix2,
                 csrMatrix2, cscMatrix2,
                 cooMatrix3, numRows1,
                 numRows2, numCols2,
                  numNonzeros1,  numNonzeros2, result);	

//	numThreadsPerBlock = 128;
//	numBlocks = (cooMatrix1->numNonzeros + numThreadsPerBlock - 1)/ numThreadsPerBlock

//populate_output(result, cooMatrix3)
	cudaDeviceSynchronize();
  // synchronize across all blocks
  int numColumnsOutputMatrix = cooMatrix3->numCols;
  for (auto &p : result) {
    int index = cooMatrix3->numNonzeros++;
    cooMatrix3->rowIdxs[index] = p.first / numColumnsOutputMatrix;
    cooMatrix3->colIdxs[index] = p.first % numColumnsOutputMatrix;
    cooMatrix3->values[index] = p.second;
  }

}
