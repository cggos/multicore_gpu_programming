NVCC = nvcc
CC = g++
CUDA_LINK_FLAGS =  -rdc=true -arch=compute_30 -code=sm_30 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -arch=compute_30 -code=sm_30 -Xcompiler '-fPIC' -g
QT_COMPILE_FLAGS = -I/usr/include/qt5/QtCore  -I/usr/include/qt5/QtGui -I/usr/include/qt5 -g
QT_LINK_FLAGS = -L/usr/lib/x86_64-linux-gnu -lQt5Gui -lQt5Core  -lpthread 

mandelbrotCUDA : main.o kernel.o
	${NVCC} ${CUDA_LINK_FLAGS} ${QT_LINK_FLAGS} $^ -o $@

main.o : main.cpp kernel.h
	${CC} ${QT_COMPILE_FLAGS} -c main.cpp -fPIC

kernel.o : kernel.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} ${QT_COMPILE_FLAGS} -c kernel.cu 

clean:
	rm *.o 
