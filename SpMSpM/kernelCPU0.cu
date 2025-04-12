
#include "common.h"
#include <unordered_map>
using namespace std;

void spmspm_cpu0(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3) {

  // matrix 1 as COO cooMatrix1
  // matrix 2 as CSR csrMatrix2
  unordered_map<int, float> result;
  int numColumnsOutputMatrix = cooMatrix3->numCols;
  for (int i = 0; i < cooMatrix1->numNonzeros; ++i) {
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
  for (auto &p : result) {
    int index = cooMatrix3->numNonzeros++;
    cooMatrix3->rowIdxs[index] = p.first / numColumnsOutputMatrix;
    cooMatrix3->colIdxs[index] = p.first % numColumnsOutputMatrix;
    cooMatrix3->values[index] = p.second;
  }
}
