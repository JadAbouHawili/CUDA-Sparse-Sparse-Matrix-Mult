#include <iostream>
#include "common.h"
//#include <unordered_map>

#define BLOCK_DIM 256 
#define temp_size 32 

using namespace std;

__global__ void mul_kernel(CSRMatrix *csrMatrix1, CSRMatrix *csrMatrix2, COOMatrix *cooMatrix3){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	//thread per row
	if(i < csrMatrix1->numRows){
		int row_start = csrMatrix1->rowPtrs[i];
		int row_end = csrMatrix1->rowPtrs[i+1];

		//temporary storage
		float temp[temp_size];
		for(int k = 0; k < temp_size; k++){
			temp[k] = 0;
		}

		// iterate over every row in matrix 2

		for(int j = row_start; j < row_end; ++j){
			int col = csrMatrix1->colIdxs[j];
			float val = csrMatrix1->values[j];


			int row_start_2 = csrMatrix2->rowPtrs[col];
			int row_end_2 = csrMatrix2->rowPtrs[col+1];

			for(int k = row_start_2; k < row_end_2; k++){
				int col2 = csrMatrix2->colIdxs[k];
				float val2 = csrMatrix2->values[k];

				float store = val*val2;

				temp[col2] += store;
			}
		}
		for(int k = 0; k < temp_size;k++){
			if(temp[k] != 0){
				int index = atomicAdd(&cooMatrix3->numNonzeros,1);
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
	int num_Blocks = (numRows1 + threadsPerBlock - 1)/threadsPerBlock;

	mul_kernel<<<num_Blocks,threadsPerBlock>>>(csrMatrix1,csrMatrix2,cooMatrix3);


}
