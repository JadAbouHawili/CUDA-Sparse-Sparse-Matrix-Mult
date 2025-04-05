
# Overview

This code performs a sparse matrix sparse matrix multiplication (SpMSpM) specialized for the scenario when the second matrix is very thin.

# Instructions

To compile:

```
make
```

To run:

```
./spmspm [flags]

```

Optional flags:

```
  -d <dataset>      the prefix of the files containing the two matrices to be multiplied
                        the first matrix will be read from <dataset>-matrix1.txt
                        the second matrix will be read from <dataset>-matrix2.txt
                        the reference output matrix will be read from <dataset>-matrix3.txt

  -a                run CPU version 0
  -b                run CPU version 1

  -0                run GPU version 0
  -1                run GPU version 1
  -2                run GPU version 2
  -3                run GPU version 3
  -4                run GPU version 4
  -5                run GPU version 5
  -6                run GPU version 6
  -7                run GPU version 7
  -8                run GPU version 8
  -9                run GPU version 9
                    NOTE: It is okay to specify multiple versions in the same run

  -v                perform exact verification

  -w                write result to a file
                        the CPU result will be written to <dataset>-matrix3-cpu.txt
                        the GPU result will be written to <dataset>-matrix3-gpu.txt

```

