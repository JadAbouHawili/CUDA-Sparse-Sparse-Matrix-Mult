
#include "common.h"


float add_row_col(unsigned int* rowPtrs, unsigned int* colPtrs, float* r_values, float* c_values, int num_row_elements, int num_col_elements){

	float sum = 0;

	for(int i = 0; i < num_row_elements; i++){
		for(int j = 0; j < num_col_elements; j++){
			int index_row = rowPtrs[i];
			int index_col = colPtrs[j];

			if(index_row == index_col){
				sum += r_values[i]*c_values[j];
			}
		}
	}

	return sum;
}



void spmspm_cpu0(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3) {

	// matrix 1 as CSR
	// matrix 2 as CSC

	for(int i = 0; i < csrMatrix1->numRows; i++){
		for(int j = 0; j < cscMatrix2->numCols; j++){
			int num_elements_row = csrMatrix1->rowPtrs[i + 1] - csrMatrix1->rowPtrs[i];
			int num_elements_col = cscMatrix2->colPtrs[j + 1] - cscMatrix2->colPtrs[j];
			float addant = add_row_col(&csrMatrix1->colIdxs[csrMatrix1->rowPtrs[i]], &cscMatrix2->rowIdxs[cscMatrix2->colPtrs[j]],&csrMatrix1->values[csrMatrix1->rowPtrs[i]],&cscMatrix2->values[cscMatrix2->colPtrs[j]], num_elements_row, num_elements_col);

			if(addant != 0){
				int index = cooMatrix3->numNonzeros;
				cooMatrix3->numNonzeros++;
				cooMatrix3->rowIdxs[index] = i;
				cooMatrix3->colIdxs[index] = j;
				cooMatrix3->values[index] = addant;
			}

		}
	}
	
}

