
#include "common.h"

void spmspm_cpu0(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3) {

	// matrix 1 as COO
	// matrix 2 as CSC

	for(int col_2 = 0; col_2 < cscMatrix2-> numCols; ++col_2){
		int start = cscMatrix2->colPtrs[col_2];
		int end = cscMatrix2->colPtrs[col_2 + 1];
		float toStore_mem[cooMatrix1->numRows];
		for(int i = 0; i < cooMatrix1->numRows; i++){
			toStore_mem[i] = 0;
		}
		for(int i = 0; i < cooMatrix1->numNonzeros; ++i){
			int inRow = cooMatrix1->rowIdxs[i];
			int inCol = cooMatrix1->colIdxs[i];
			float val = cooMatrix1->values[i];

			float val2 = 0;
			for(int j = start; j < end; j++){
				if(inCol == cscMatrix2->rowIdxs[j]){
					val2 = cscMatrix2->values[j];
					break;
				}
			}
			float toStore = val2*val;
			
			toStore_mem[inRow] += toStore;
		}
		
		for(int i = 0; i < cooMatrix1->numRows; i++){
			float val = toStore_mem[i];
			if(val != 0){
				int index = cooMatrix3->numNonzeros;
				cooMatrix3->rowIdxs[index] = i;
				cooMatrix3->colIdxs[index] = col_2;
				cooMatrix3->values[index] = val;
				cooMatrix3->numNonzeros++;
			}
		}


	}
	
}

