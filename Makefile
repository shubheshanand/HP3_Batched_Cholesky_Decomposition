#bulid all run files
  right_partA: ./Part\ A/Right\ Looking/main_right_looking.cu
	nvcc ./Part\ A/Right\ Looking/main_right_looking.cu -o rl_A
		./rl_A $(input_file) $(output_file)

  left_partA: ./Part\ A/left_looking/driver_code.cu
	nvcc ./Part\ A/left_looking/driver_code.cu -o ll_A
		./ll_A $(input_file) $(output_file)

  top_partA: ./Part\ A/top_looking/top_looking_shared.cu
	nvcc ./Part\ A/top_looking/top_looking_shared.cu -o tl_A
		./tl_A $(input_file) $(output_file)
  clean: 
	  $(RM) rl_A
	  $(RM) ll_A
	  $(RM) tl_A
