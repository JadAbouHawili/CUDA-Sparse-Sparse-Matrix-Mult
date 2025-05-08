#include "common.h"
__global__ void mul_kernel_opt_5(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2,
                                COOMatrix* cooMatrix3, unsigned int numColM2,unsigned int numRowsM1) {
    
}

void spmspm_gpu5(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1,
                COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2,
                COOMatrix* cooMatrix3, unsigned int numRows1, unsigned int numRows2,
                unsigned int numCols2, unsigned int numNonzeros1, unsigned int numNonzeros2) {
}

