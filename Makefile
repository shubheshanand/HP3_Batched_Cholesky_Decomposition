# build an executable named rl_A and take input from input_file.txt and print the output to output_file.txt
  right_partA: input_file.txt output_file.txt
 	  nvcc "./Part A/Right Looking/main_right_looking.cu" -o rl_A
          ./rl_A input_file.txt output_file.txt

  clean: 
	  $(RM) rl_A