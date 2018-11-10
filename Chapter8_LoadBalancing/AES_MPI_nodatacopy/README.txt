- The block size used to process data on the GPU (1, 16, or 32 MB) is determined by a constant in rijndael_host.cu
- There are two different implementations of the partitioning functions
      partition.cpp_withNode0
      partition.cpp_withoutNode0
The chosen implementation should be copied or linked to partition.cpp before compilation.

