# make command
For Right looking
```
make right input_file=./Dataset/mat_256.txt output_file=output.txt
make clean_right
```

For Left looking
```
make left input_file=./Dataset/mat_256.txt output_file=output.txt
make clean_left
```

For Top looking
```
make top input_file=./Dataset/mat_256.txt output_file=output.txt
make clean_top
```


# Command Lines
Compile Command: nvcc main_right_looking.cu -o rl_A <br />
Run Command: ./rl_A input_file.txt output_file.txt <br />

# Format of Input File
First Line: Size of Matrix <br />
Rest all Lines: Elements of the Matrix in Row Major Format <br />