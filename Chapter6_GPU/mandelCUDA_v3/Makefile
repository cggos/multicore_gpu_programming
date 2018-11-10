NVCC = nvcc
CC = g++
CUDA_LINK_FLAGS =  -rdc=true -arch=compute_20 -code=sm_21
CUDA_COMPILE_FLAGS = --device-c -arch=compute_20 -code=sm_21
QT_COMPILE_FLAGS = -I/usr/include/qt4/QtCore  -I/usr/include/qt4/QtGui -I/usr/include/qt4
QT_LINK_FLAGS = -L/usr/lib/x86_64-linux-gnu -lQtGui -lQtCore -lpthread

mandelbrotCUDA : main.o kernel.o
	${NVCC} ${CUDA_LINK_FLAGS} ${QT_LINK_FLAGS} $^ -o $@

main.o : main.cpp kernel.h
	${CC} ${QT_COMPILE_FLAGS} -c main.cpp 

kernel.o : kernel.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} ${QT_COMPILE_FLAGS} -c kernel.cu 

clean:
	rm *.o 
