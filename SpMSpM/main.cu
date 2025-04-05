
#include <assert.h>
#include <stdio.h>
#include <string>
#include <unistd.h>

#include "common.h"
#include "matrix.h"
#include "timer.h"

void verify(COOMatrix* cooMatrix, COOMatrix* cooMatrix_ref, unsigned int quickVerify);

void (*spmspm_cpu[])(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3) = {
    spmspm_cpu0,
    spmspm_cpu1
};


void (*spmspm_gpu[])(COOMatrix* cooMatrix1, CSRMatrix* csrMatrix1, CSCMatrix* cscMatrix1, COOMatrix* cooMatrix2, CSRMatrix* csrMatrix2, CSCMatrix* cscMatrix2, COOMatrix* cooMatrix3, unsigned int numRows1, unsigned int numRows2, unsigned int numCols2, unsigned int numNonzeros1, unsigned int numNonzeros2) = {
    spmspm_gpu0,
    spmspm_gpu1,
    spmspm_gpu2,
    spmspm_gpu3,
    spmspm_gpu4,
    spmspm_gpu5,
    spmspm_gpu6,
    spmspm_gpu7,
    spmspm_gpu8,
    spmspm_gpu9
};

int main(int argc, char**argv) {

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    setbuf(stdout, NULL);

    // Parse arguments
    const char* dataset = "data/dataset1";
    unsigned int runCPUVersion[2]  = { 0 };
    unsigned int runGPUVersion[10] = { 0 };
    unsigned int useGPU = 0;
    unsigned int quickVerify = 1;
    unsigned int writeResult = 0;
    int opt;
    while((opt = getopt(argc, argv, "d:ab0123456789vw")) >= 0) {
        switch(opt) {
            case 'd': dataset = optarg;                 break;
            case 'a': runCPUVersion[0] = 1;             break;
            case 'b': runCPUVersion[1] = 1;             break;
            case '0': runGPUVersion[0] = 1; useGPU = 1; break;
            case '1': runGPUVersion[1] = 1; useGPU = 1; break;
            case '2': runGPUVersion[2] = 1; useGPU = 1; break;
            case '3': runGPUVersion[3] = 1; useGPU = 1; break;
            case '4': runGPUVersion[4] = 1; useGPU = 1; break;
            case '5': runGPUVersion[5] = 1; useGPU = 1; break;
            case '6': runGPUVersion[6] = 1; useGPU = 1; break;
            case '7': runGPUVersion[7] = 1; useGPU = 1; break;
            case '8': runGPUVersion[8] = 1; useGPU = 1; break;
            case '9': runGPUVersion[9] = 1; useGPU = 1; break;
            case 'v': quickVerify = 0;      break;
            case 'w': writeResult = 1;      break;
            default:  fprintf(stderr, "\nUnrecognized option!\n");
                      exit(0);
        }
    }

    // Allocate memory and initialize data
    std::string matrixFile1 = std::string(dataset) + "-matrix1.txt";
    printf("Reading first input matrix from file: %s\n", matrixFile1.c_str());
    COOMatrix* cooMatrix1 = createCOOMatrixFromFile(matrixFile1.c_str());
    CSRMatrix* csrMatrix1 = createCSRMatrixFromCOOMatrix(cooMatrix1);
    CSCMatrix* cscMatrix1 = createCSCMatrixFromCOOMatrix(cooMatrix1);
    std::string matrixFile2 = std::string(dataset) + "-matrix2.txt";
    printf("Reading second input matrix from file: %s\n", matrixFile2.c_str());
    COOMatrix* cooMatrix2 = createCOOMatrixFromFile(matrixFile2.c_str());
    CSRMatrix* csrMatrix2 = createCSRMatrixFromCOOMatrix(cooMatrix2);
    CSCMatrix* cscMatrix2 = createCSCMatrixFromCOOMatrix(cooMatrix2);
    assert(cooMatrix1->numCols == cooMatrix2->numRows);
    std::string matrixFile3 = std::string(dataset) + "-matrix3.txt";
    printf("Reading reference output matrix from file: %s\n", matrixFile3.c_str());
    COOMatrix* cooMatrix3_ref = createCOOMatrixFromFile(matrixFile3.c_str());
    printf("Allocating output matrices\n");
    COOMatrix* cooMatrix3_cpu = createEmptyCOOMatrix(csrMatrix1->numRows, csrMatrix2->numCols, csrMatrix1->numRows*100);
    COOMatrix* cooMatrix3_gpu = createEmptyCOOMatrix(csrMatrix1->numRows, csrMatrix2->numCols, csrMatrix1->numRows*100);

    // Compute on CPU
    for(unsigned int version = 0; version < 2; ++version) {
        if(runCPUVersion[version]) {

            printf("Running CPU version %d\n", version);

            // Reset
            clearCOOMatrix(cooMatrix3_cpu);

            // Compute on CPU
            Timer timer;
            startTime(&timer);
            spmspm_cpu[version](cooMatrix1, csrMatrix1, cscMatrix1, cooMatrix2, csrMatrix2, cscMatrix2, cooMatrix3_cpu);
            stopTime(&timer);
            printElapsedTime(timer, "    CPU time", CYAN);

            // Verify
            verify(cooMatrix3_cpu, cooMatrix3_ref, quickVerify);

            // Write result to file
            if(writeResult) {
                sortCOOMatrix(cooMatrix3_cpu);
                std::string matrixFile3_cpu = std::string(dataset) + "-matrix3-cpu.txt";
                writeCOOMatrixToFile(cooMatrix3_cpu, matrixFile3_cpu.c_str());
            }

        }
    }

    // Compute on GPU
    if(useGPU) {

        // Allocate GPU memory
        Timer timer;
        startTime(&timer);
        COOMatrix* cooMatrix1_d = createEmptyCOOMatrixOnGPU(cooMatrix1->numRows, cooMatrix1->numCols, cooMatrix1->numNonzeros);
        CSRMatrix* csrMatrix1_d = createEmptyCSRMatrixOnGPU(csrMatrix1->numRows, csrMatrix1->numCols, csrMatrix1->numNonzeros);
        CSCMatrix* cscMatrix1_d = createEmptyCSCMatrixOnGPU(cscMatrix1->numRows, cscMatrix1->numCols, cscMatrix1->numNonzeros);
        COOMatrix* cooMatrix2_d = createEmptyCOOMatrixOnGPU(cooMatrix2->numRows, cooMatrix2->numCols, cooMatrix2->numNonzeros);
        CSRMatrix* csrMatrix2_d = createEmptyCSRMatrixOnGPU(csrMatrix2->numRows, csrMatrix2->numCols, csrMatrix2->numNonzeros);
        CSCMatrix* cscMatrix2_d = createEmptyCSCMatrixOnGPU(cscMatrix2->numRows, cscMatrix2->numCols, cscMatrix2->numNonzeros);
        COOMatrix* cooMatrix3_d = createEmptyCOOMatrixOnGPU(cooMatrix3_gpu->numRows, cooMatrix3_gpu->numCols, cooMatrix3_gpu->capacity);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        stopTime(&timer);
        printElapsedTime(timer, "GPU allocation time");

        // Copy data to GPU
        startTime(&timer);
        copyCOOMatrixToGPU(cooMatrix1, cooMatrix1_d);
        copyCSRMatrixToGPU(csrMatrix1, csrMatrix1_d);
        copyCSCMatrixToGPU(cscMatrix1, cscMatrix1_d);
        copyCOOMatrixToGPU(cooMatrix2, cooMatrix2_d);
        copyCSRMatrixToGPU(csrMatrix2, csrMatrix2_d);
        copyCSCMatrixToGPU(cscMatrix2, cscMatrix2_d);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        stopTime(&timer);
        printElapsedTime(timer, "Copy to GPU time");

        for(unsigned int version = 0; version < 10; ++version) {
            if(runGPUVersion[version]) {

                printf("Running GPU version %d\n", version);

                // Reset
                clearCOOMatrixOnGPU(cooMatrix3_d);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());

                // Compute on GPU
                startTime(&timer);
                spmspm_gpu[version](cooMatrix1_d, csrMatrix1_d, cscMatrix1_d, cooMatrix2_d, csrMatrix2_d, cscMatrix2_d, cooMatrix3_d, cooMatrix1->numRows, cooMatrix2->numRows, cooMatrix2->numCols, cooMatrix1->numNonzeros, cooMatrix2->numNonzeros);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());
                stopTime(&timer);
                printElapsedTime(timer, "    GPU kernel time", GREEN);

                // Copy data from GPU
                startTime(&timer);
                copyCOOMatrixFromGPU(cooMatrix3_d, cooMatrix3_gpu);
                CUDA_ERROR_CHECK(cudaDeviceSynchronize());
                stopTime(&timer);
                printElapsedTime(timer, "    Copy from GPU time");

                // Verify
                verify(cooMatrix3_gpu, cooMatrix3_ref, quickVerify);

                // Write result to file
                if(writeResult) {
                    sortCOOMatrix(cooMatrix3_gpu);
                    std::string matrixFile3_gpu = std::string(dataset) + "-matrix3-gpu.txt";
                    writeCOOMatrixToFile(cooMatrix3_gpu, matrixFile3_gpu.c_str());
                }

            }
        }

        // Free GPU memory
        startTime(&timer);
        freeCOOMatrixOnGPU(cooMatrix1_d);
        freeCSRMatrixOnGPU(csrMatrix1_d);
        freeCSCMatrixOnGPU(cscMatrix1_d);
        freeCOOMatrixOnGPU(cooMatrix2_d);
        freeCSRMatrixOnGPU(csrMatrix2_d);
        freeCSCMatrixOnGPU(cscMatrix2_d);
        freeCOOMatrixOnGPU(cooMatrix3_d);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        stopTime(&timer);
        printElapsedTime(timer, "GPU deallocation time");

    }

    // Free memory
    freeCOOMatrix(cooMatrix1);
    freeCSRMatrix(csrMatrix1);
    freeCSCMatrix(cscMatrix1);
    freeCOOMatrix(cooMatrix2);
    freeCSRMatrix(csrMatrix2);
    freeCSCMatrix(cscMatrix2);
    freeCOOMatrix(cooMatrix3_ref);
    freeCOOMatrix(cooMatrix3_cpu);
    freeCOOMatrix(cooMatrix3_gpu);

    return 0;

}

void verify(COOMatrix* cooMatrix, COOMatrix* cooMatrix_ref, unsigned int quickVerify) {
    if(cooMatrix_ref->numNonzeros != cooMatrix->numNonzeros) {
        printf("    \033[1;31mMismatching number of non-zeros (reference result = %d, computed result = %d)\033[0m\n", cooMatrix_ref->numNonzeros, cooMatrix->numNonzeros);
        return;
    } else if(quickVerify) {
        printf("    Quick verification succeeded\n");
        printf("        This verification is not exact. For exact verification, pass the -v flag.\n");
    } else {
        printf("    Verifying result\n");
        sortCOOMatrix(cooMatrix);
        for(unsigned int i = 0; i < cooMatrix_ref->numNonzeros; ++i) {
            unsigned int row_ref = cooMatrix_ref->rowIdxs[i];
            unsigned int row_comp = cooMatrix->rowIdxs[i];
            unsigned int col_ref = cooMatrix_ref->colIdxs[i];
            unsigned int col_comp = cooMatrix->colIdxs[i];
            float val_ref = cooMatrix_ref->values[i];
            float val_comp = cooMatrix->values[i];
            if(row_ref != row_comp || col_ref != col_comp || abs(val_comp - val_ref)/val_ref > 1e-5) {
                printf("        \033[1;31mMismatch detected: Reference: (%d, %d, %f), Computed: (%d, %d, %f)\033[0m\n", row_ref, col_ref, val_ref, row_comp, col_comp, val_comp);
                return;
            }
        }
        printf("        Verification succeeded\n");
    }
}

