# Batched Cholesky Decomposition

## Link to Dataset:
https://drive.google.com/file/d/1hCHJI6G5UZLa2_zZOrg9RIs2T7meQH9j/view?usp=sharing

## Link to Report:
https://docs.google.com/document/d/e/2PACX-1vTcik1DFqdVNNmPJan4CBWCr__KqvCSRVTknqgLQf_DLIxeNI4ZCiwO0VqZUk6otmmOczngD7MnKk8O/pub

# Make commands for Part A
For Right looking
```
make right_partA input_file=mat_256.txt output_file=output.txt
```

For Left looking
```
make left_partA input_file=mat_256.txt output_file=output.txt
```

For Top looking
```
make top_partA input_file=mat_256.txt output_file=output.txt
```

# Make commands for Part B
For Right looking Chunked
```
make right_chunked input_file=dataset_part_B/num_1024_dim_20.txt output_file=output.txt
```

For Right looking Interleaved
```
make right_interleaved input_file=dataset_part_B/num_1024_dim_20.txt output_file=output.txt
```

For Left looking Chunked
```
make left_chunked input_file=dataset_part_B/num_1024_dim_20.txt output_file=output.txt
```

For Left looking Interleaved
```
make left_interleaved input_file=dataset_part_B/num_1024_dim_20.txt output_file=output.txt
```

For Top looking Chunked
```
make top_chunked input_file=dataset_part_B/num_1024_dim_20.txt output_file=output.txt
```

For Top looking Interleaved
```
make top_interleaved input_file=dataset_part_B/num_1024_dim_20.txt output_file=output.txt
```

For Cleaning all the excutables file
```
make clean
```

## Format of Input File for partA
First Line: Size of Matrix <br />
Rest all Lines: Elements of the Matrix in Row Major Format <br />

## Format of Input File for partB
First Line: Number of Matrices <br />
Second Line: Size of Matrix <br />
Rest all Lines: Elements of the Matrix in Row Major Format <br />