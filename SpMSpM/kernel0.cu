//
#include "common.h"
//#include <unordered_map>

#define BLOCK_DIM 256 
#define temp_size 16 

using namespace std;

__global__ void mul_kernel(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2, COOMatrix *cooMatrix3, int num_rows){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int k = csrMatrix1->numCols;

	extern __shared__ float buffer[];	
	
	int myRow = i/k;
	int myCol = i%k;
	
	if(myRow < csrMatrix1->numRows){
		int numNonZeroEntries = csrMatrix1->rowPtrs[myRow + 1] - csrMatrix1->rowPtrs[myRow];
		
		if(myCol < numNonZeroEntries){
			int index = csrMatrix1->rowPtrs[myRow] + myCol;

			int col = csrMatrix1->colIdxs[index];
			int val = csrMatrix1->values[index];

			// go to matrix2
			int numNonZeroEntries2 = csrMatrix2->rowPtrs[col + 1] - csrMatrix2->rowPtrs[col];
			int index2 = csrMatrix2->rowPtrs[col];
			for(int j = 0; j < numNonZeroEntries2; j++){
				int col_2 = csrMatrix2->colIdxs[index2 + j];
				int val_2 = csrMatrix2->values[index2 + j];
				atomicAdd(&buffer[myRow * csrMatrix2->numCols + col_2], val*val_2);
			}
		}
	}
	__syncthreads();

	if(threadIdx.x == 0){
		for(int i = 0 ; i < csrMatrix2->numCols; i++){
			for(int j = 0; j < num_rows; j++){
				float x = buffer[j * csrMatrix2->numCols + i];
				if(x > 0){
					int index = atomicAdd(&cooMatrix3->numNonzeros,1);
  					cooMatrix3->rowIdxs[index] = j;
  					cooMatrix3->colIdxs[index] = i;
  					cooMatrix3->values[index] = x;
				}
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
	int k = csrMatrix1->numCols;
	int num_rows = BLOCK_DIM/k;

	int threadsPerBlock = num_rows*k;
	int num_Blocks = ((csrMatrix1->numRows)*k + threadsPerBlock - 1)/threadsPerBlock;

	size_t shared_mem_size = sizeof(float) * num_rows * csrMatrix2->numCols;

	mul_kernel<<<num_Blocks,threadsPerBlock,shared_mem_size >>>(csrMatrix1,csrMatrix2,cooMatrix3,num_rows);


}
