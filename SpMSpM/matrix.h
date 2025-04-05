
#ifndef __MATRIX_H_
#define __MATRIX_H_

struct COOMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int capacity;
    unsigned int* rowIdxs;
    unsigned int* colIdxs;
    float* values;
};

COOMatrix* createCOOMatrixFromFile(const char* fileName);
COOMatrix* createEmptyCOOMatrix(unsigned int numRows, unsigned int numCols, unsigned int capacity);
void freeCOOMatrix(COOMatrix* cooMatrix);

COOMatrix* createEmptyCOOMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int capacity);
void freeCOOMatrixOnGPU(COOMatrix* cooMatrix);

void clearCOOMatrix(COOMatrix* cooMatrix);
void clearCOOMatrixOnGPU(COOMatrix* cooMatrix);
void copyCOOMatrixToGPU(COOMatrix* cooMatrix_h, COOMatrix* cooMatrix_d);
void copyCOOMatrixFromGPU(COOMatrix* cooMatrix_d, COOMatrix* cooMatrix_h);

void sortCOOMatrix(COOMatrix* cooMatrix);
void writeCOOMatrixToFile(COOMatrix* cooMatrix, const char* fileName);

struct CSRMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int* rowPtrs;
    unsigned int* colIdxs;
    float* values;
};

CSRMatrix* createCSRMatrixFromCOOMatrix(COOMatrix* cooMatrix);
void freeCSRMatrix(CSRMatrix* csrMatrix);

CSRMatrix* createEmptyCSRMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int numNonzeros);
void freeCSRMatrixOnGPU(CSRMatrix* csrMatrix);

void copyCSRMatrixToGPU(CSRMatrix* csrMatrix_h, CSRMatrix* csrMatrix_d);

struct CSCMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int* colPtrs;
    unsigned int* rowIdxs;
    float* values;
};

CSCMatrix* createCSCMatrixFromCOOMatrix(COOMatrix* cooMatrix);
void freeCSCMatrix(CSCMatrix* cscMatrix);

CSCMatrix* createEmptyCSCMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int numNonzeros);
void freeCSCMatrixOnGPU(CSCMatrix* cscMatrix);

void copyCSCMatrixToGPU(CSCMatrix* cscMatrix_h, CSCMatrix* cscMatrix_d);

#endif

