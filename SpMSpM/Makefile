
NVCC        = nvcc
NVCC_FLAGS  = -O3
OBJ         = main.o matrix.o kernelCPU0.o kernelCPU1.o kernel0.o kernel1.o kernel2.o kernel3.o kernel4.o kernel5.o kernel6.o kernel7.o kernel8.o kernel9.o
EXE         = spmspm


default: $(EXE)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

