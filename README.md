# Batched Cholesky Decomposition

## Link to Dataset:
https://drive.google.com/file/d/1hCHJI6G5UZLa2_zZOrg9RIs2T7meQH9j/view?usp=sharing

## Link to Report:
https://docs.google.com/document/d/e/2PACX-1vTcik1DFqdVNNmPJan4CBWCr__KqvCSRVTknqgLQf_DLIxeNI4ZCiwO0VqZUk6otmmOczngD7MnKk8O/pub

# make command
For Right looking
```
make right_partA input_file=./Part\ A/Dataset/mat_256.txt output_file=output.txt
```

For Left looking
```
make left_partA input_file=./Part\ A/Dataset/mat_256.txt output_file=output.txt
```

For Top looking
```
make top_partA input_file=./Part\ A/Dataset/mat_256.txt output_file=output.txt
```

For Cleaning the excutables file
```
make clean
```

## Format of Input File for partA
First Line: Size of Matrix <br />
Rest all Lines: Elements of the Matrix in Row Major Format <br />
