
#include "common.h"

#define COL_OUTPUT_SIZE 512
#define EMPTY_CELL std::numeric_limits<float>::max()

__global__ mul_kernel_opt_2(CSRMatrix* csrMatrix1,CSRMatrix* csrMatrix2,COOMatrix* cooMatrix3,unsigned int numColM2){

    int numLoops = (numColM2 + COL_OUTPUT_SIZE - 1)/COL_OUTPUT_SIZE;

    __shared__  float rowOutput_s[COL_OUTPUT_SIZE];

    for(int counterLoop = 1; counterLoop <= numLoops; ++counterLoop){

        for(int i = threadIdx.x; i < COL_OUTPUT_SIZE; i += blockDim.x){
            rowOutput_s[i] = 0;
        }

        int rowIdxM1 = blockIdx.x ; 
        int rowSizeM1 = csrMatrix1 -> rowPtrs[rowIdxM1+1] - csrMatrix1 -> rowPtrs[rowIdxM1];
        int startingIndexInRowM1 = csrMatrix1 -> rowPtrs[rowIdxM1] + threadIdx.x ; // Assign a thread to each entry in row
        for(int varStartingRowIndexM1 = startingIndexInRowM1 ; varStartingRowIndexM1 < rowSizeM1 ; varStartingRowIndexM1 += blockDim.x ){
            int colIndexM1 = csrMatrix1 -> colIdxs[varStartingRowIndexM1] ;
            float valM1 = csrMatrix1 -> values[varStartingRowIndexM1] ;
            int rowSizeM2 = csrMatrix2 -> rowPtrs[colIndexM1+1] - csrMatrix2 -> rowPtrs[colIndexM1];
            int startingIndexInRowM2 = csrMatrix2 -> rowPtrs[colIndexM1] ;
            int emptyCellIdx = -1 ; 
            int numEmptyElements = 0 ; 
            for(int varStartingRowIndexM2 = startingIndexInRowM2 ; varStartingRowIndexM2 < rowSizeM2 ; ++varStartingRowIndexM2){
                float valM2 = csrMatrix2 -> values[varStartingRowIndexM2] ;
                int colIndexM2 = csrMatrix2 -> colIdxs[varStartingRowIndexM2] ;
                if(colIndexM2 < COL_OUTPUT_SIZE*counterLoop){
                    rowOutput_s[colIndexM2] += valM1 * valM2 ;
                    if(numEmptyElements == 0){
                        emptyCellIdx = varStartingRowIndexM2 ;
                    }
                    numEmptyElements +=1 ; 
                }
                else{
                    if(emptyCellIdx != -1){
                        csrMatrix2 -> values[emptyCellIdx] = csrMatrix2 -> values[varStartingRowIndexM2] ;
                        csrMatrix2 -> colIdxs[emptyCellIdx] = csrMatrix2 -> values[varStartingRowIndexM2] ;
                        numEmptyElements -= 1 ;
                        emptyCellIdx +=1 ; 
                    }                
                }
            }
            __syncthreads();
            for(int outputIdx = threadIdx.x ; outputIdx < COL_OUTPUT_SIZE ; outputIdx+= blockDim.x){
                if(rowOutput_s[outputIdx] != 0){
                    int index = atomicAdd(&cooMatrix3->numNonzeros,1);
                    cooMatrix3->rowIdxs[index] = rowIdxM1;
                    cooMatrix3->colIdxs[index] = 1;// to be computed;
                    cooMatrix3->values[index] = rowOutput_s[outputIdx];

                }
            }
        }
    }
}



// function that launches the kernel
void spmspm_gpu2(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3, unsigned int numRows1, unsigned int numRows2, unsigned int numCols2, unsigned int numNonzeros1, unsigned int numNonzeros2) {

    int numThreadsPerBlock = 512;

    // num rows of first matrix
    int numBlocks = numRows1; 
    mul_kernel_opt_2<<<numBlocks,numThreadsPerBlock>>>(csrMatrix1, csrMatrix2, cooMatrix3, numCols2); 

}

