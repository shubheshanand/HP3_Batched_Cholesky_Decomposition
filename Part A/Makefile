# build an executable named rl_A and take input from input_file.txt and print the output to output_file.txt
  right: ./Right\ Looking/main_right_looking.cu
	nvcc ./Right\ Looking/main_right_looking.cu -o rl_A
		./rl_A $(input_file) $(output_file)

  clean_right: 
	$(RM) rl_A


  left: ./left_looking/driver_code.cu
	nvcc ./left_looking/driver_code.cu -o ll_A
		./ll_A $(input_file) $(output_file)


  clean_left: 
	$(RM) ll_A


  top: ./top_looking/top_looking_shared.cu
	nvcc ./top_looking/top_looking_shared.cu -o tl_A
		./tl_A $(input_file) $(output_file)


  clean_top: 
	$(RM) tl_A