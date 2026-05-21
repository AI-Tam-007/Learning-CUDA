# Learning-CUDA
从零开始学习CUDA





SoftMax FP32:
Shape	     DType	   CPU	        Naive	       Shared	       Warp Shuffle	  Register Cache	 Vectorized	                            PyTorch
128×256	   FP32	  0.54337 ms	 0.120909  ms	  0.009728  ms   0.0192512 ms	  0.0078784 ms	  0.0072704  ms	                         0.017738 ms
4096×64  	 FP32	  4.87485 ms	 0.0569344 ms	  0.07936   ms	 0.0102112 ms	  0.0076512	ms    0.0075776  ms (一次向量化读取2个float)	 0.078848 ms
1024×1000	 FP32	  17.83   ms	 0.662016  ms	  0.0428128 ms	 0.0457632 ms	  0.0290816 ms	  0.02816    ms	                         0.030925 ms
1024×1024	 FP32  	18.2633 ms	 0.893542  ms	  0.0312288 ms	 0.042496  ms	  0.0277504 ms	  0.027648   ms	                         0.032870 ms
8192×4096	 FP32	  599.802 ms	 6.27671   ms	  1.07889   ms	 1.52983   ms	  0.98537   ms	  0.962662   ms	                         1.474842 ms

