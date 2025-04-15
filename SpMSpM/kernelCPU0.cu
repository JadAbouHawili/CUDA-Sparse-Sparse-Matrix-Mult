#include "common.h"
using namespace std;

void spmspm_cpu0(COOMatrix *cooMatrix1, CSRMatrix *csrMatrix1,
                 CSCMatrix *cscMatrix1, COOMatrix *cooMatrix2,
                 CSRMatrix *csrMatrix2, CSCMatrix *cscMatrix2,
                 COOMatrix *cooMatrix3) {

  // matrix 1 as COO cooMatrix1
  // matrix 2 as CSR csrMatrix2
  //unordered_map<int, float> result;
  //int numColumnsOutputMatrix = cooMatrix3->numCols;
  //for (int i = 0; i < cooMatrix1->numNonzeros; ++i) {
  //  int rowNumber = cooMatrix1->rowIdxs[i];
  //  int colNumber = cooMatrix1->colIdxs[i];
  //  float val = cooMatrix1->values[i];
  //  int csrStart = csrMatrix2->rowPtrs[colNumber];
  //  int csrEnd = csrMatrix2->rowPtrs[colNumber + 1];
  //  for (int j = csrStart; j < csrEnd; ++j) {
  //    int colNumber2 = csrMatrix2->colIdxs[j];
  //    float val2 = csrMatrix2->values[j];
  //    int oneToOneIndex = rowNumber * numColumnsOutputMatrix + colNumber2;
  //    result[oneToOneIndex] += val * val2;
  //  }
  //}
  //for (auto &p : result) {
  //  int index = cooMatrix3->numNonzeros++;
  //  cooMatrix3->rowIdxs[index] = p.first / numColumnsOutputMatrix;
  //  cooMatrix3->colIdxs[index] = p.first % numColumnsOutputMatrix;
  //  cooMatrix3->values[index] = p.second;
  //}
	// matrix 1 as CSR csrMatrix1
	// matrix 2 as CSR csrMatrix2
	
	for(int i = 0; i < csrMatrix1->numRows; ++i){
		int row_start = csrMatrix1->rowPtrs[i];
		int row_end = csrMatrix1->rowPtrs[i+1];

		//temporary storage
		float temp[csrMatrix2->numCols];
		for(int k = 0; k < csrMatrix2->numCols; k++){
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
		for(int k = 0; k < csrMatrix2->numCols;k++){
			if(temp[k] != 0){
				int index = cooMatrix3->numNonzeros++;
  				cooMatrix3->rowIdxs[index] = i;
  				cooMatrix3->colIdxs[index] = k;
  				cooMatrix3->values[index] = temp[k];
			}
		}
	}


}
