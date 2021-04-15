# Batched Cholesky Decomposition

## Link to Dataset:
https://drive.google.com/file/d/1hCHJI6G5UZLa2_zZOrg9RIs2T7meQH9j/view?usp=sharing

## Link to Report:
https://docs.google.com/document/d/e/2PACX-1vTcik1DFqdVNNmPJan4CBWCr__KqvCSRVTknqgLQf_DLIxeNI4ZCiwO0VqZUk6otmmOczngD7MnKk8O/pub

# make command
make right_partA input_file.txt output_file.txt <br />
make left_partA input_file.txt output_file.txt <br />
make top_partA input_file.txt output_file.txt <br />

## Sample make command
make right_partA ./dataset/mat_256.txt output_file.txt <br />
make left_partA ./dataset/mat_256.txt output_file.txt <br />
make top_partA ./dataset/mat_256.txt output_file.txt <br />

## Format of Input File for partA
First Line: Size of Matrix <br />
Rest all Lines: Elements of the Matrix in Row Major Format <br />
