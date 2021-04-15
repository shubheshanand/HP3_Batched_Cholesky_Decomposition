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

  right_chunked: ./right_looking/right_looking_chunked.cu
	nvcc ./right_looking/right_looking_chunked.cu -o rl_Chunked
		./rl_Chunked $(input_file) $(output_file)
		
  right_interleaved: ./right_looking/right_looking_interleaved.cu
	nvcc ./right_looking/right_looking_interleaved.cu -o rl_Interleaved
		./rl_Interleaved $(input_file) $(output_file)

  left_chunked: ./left_looking/left_looking_chunked_mb.cu
	nvcc ./left_looking/left_looking_chunked_mb.cu -o ll_Chunked
		./ll_Chunked $(input_file) $(output_file)
		
  left_interleaved: ./left_looking/left_looking_interleaved_mb.cu
	nvcc ./left_looking/left_looking_interleaved_mb.cu -o ll_Interleaved
		./ll_Interleaved $(input_file) $(output_file)

  top_chunked: ./top_looking/top_looking_chunked.cu
	nvcc ./top_looking/top_looking_chunked.cu -o tl_Chunked
		./tl_Chunked $(input_file) $(output_file)
		
  top_interleaved: ./top_looking/top_looking_interleaved.cu
	nvcc ./top_looking/top_looking_interleaved.cu -o tl_Interleaved
		./tl_Interleaved $(input_file) $(output_file)

  clean: 
	  $(RM) rl_A
	  $(RM) ll_A
	  $(RM) tl_A
	  $(RM) rl_Chunked
	  $(RM) rl_Interleaved
	  $(RM) ll_Chunked
	  $(RM) ll_Interleaved
	  $(RM) tl_Chunked
	  $(RM) tl_Interleaved
