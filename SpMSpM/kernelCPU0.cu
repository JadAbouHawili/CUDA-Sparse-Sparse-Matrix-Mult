
#include "common.h"

void spmspm_cpu0(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3) {
	for(int row = 0 ; row < cooMatrix1->numRows ; ++row){
		for(int col = 0 ; col < cooMatrix2->numCols ; ++col){
		float sum=0;
			for(int i =0 ; i < cooMatrix1->numCols ; ++i){
				// [row][col]
				sum += cooMatrix1->values[row * cooMatrix1->numCols + i] * cooMatrix2->values[i * cooMatrix2->numCols + col];
			}
			cooMatrix3->values[row * cooMatrix3->numCols + col] = sum;
		}
	}









}

