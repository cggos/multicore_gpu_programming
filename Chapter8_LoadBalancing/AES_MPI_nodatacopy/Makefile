NVCC = nvcc
CC = g++
CUDA_LINK_FLAGS =  -rdc=true -arch=compute_20 -code=sm_21
CUDA_COMPILE_FLAGS = -g --device-c -arch=compute_20 -code=sm_21
CC_COMPILE_FLAGS = -g -I/usr/include/openmpi
CC_LINK_FLAGS = -g -lm -lstdc++ -lmpi -L/usr/lib -lpthread -lmpi_cxx

all : aesMPI aesCUDA aesCUDAStreams aesMPICPUWorker

aesMPI : main.o rijndael_host.o rijndael_device.o
	${NVCC} ${CUDA_LINK_FLAGS} ${CC_LINK_FLAGS} $^ -o $@

aesMPICPUWorker : mainCPUWorker.o rijndael_host.o rijndael_device.o
	${NVCC} ${CUDA_LINK_FLAGS} ${CC_LINK_FLAGS} $^ -o $@

main.o : main.cpp rijndael.h partition.cpp
	${CC} ${CC_COMPILE_FLAGS} -c main.cpp 

mainCPUWorker.o : mainCPUWorker.cpp rijndael.h partition.cpp
	${CC} ${CC_COMPILE_FLAGS} -c mainCPUWorker.cpp 

rijndael_host.o : rijndael_host.cu rijndael.h rijndael_device.h
	${NVCC} ${CUDA_COMPILE_FLAGS} ${CC_COMPILE_FLAGS} -c rijndael_host.cu 

rijndael_device.o : rijndael_device.cu rijndael.h rijndael_device.h
	${NVCC} ${CUDA_COMPILE_FLAGS} ${CC_COMPILE_FLAGS} -c rijndael_device.cu 

aesCUDA : aesCUDA.o rijndael_host.o rijndael_device.o
	${NVCC} ${CUDA_LINK_FLAGS} ${CC_LINK_FLAGS} $^ -o $@

aesCUDA.o : aesCUDA.cu rijndael.h
	${NVCC} ${CUDA_COMPILE_FLAGS} ${CC_COMPILE_FLAGS} -c aesCUDA.cu 


aesCUDAStreams : aesCUDA.o rijndael_host_streams.o rijndael_device.o
	${NVCC} ${CUDA_LINK_FLAGS} ${CC_LINK_FLAGS} $^ -o $@

rijndael_host_streams.o : rijndael_host_streams.cu rijndael.h rijndael_device.h
	${NVCC} ${CUDA_COMPILE_FLAGS} ${CC_COMPILE_FLAGS} -c rijndael_host_streams.cu 

clean:
	rm *.o 
