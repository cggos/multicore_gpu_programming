NVCC = nvcc
CC = g++
CUDAINST=/opt/cuda
CUDA_LINK_FLAGS =  -rdc=true -arch=compute_20 -code=sm_21
CUDA_COMPILE_FLAGS = -O2 --device-c -arch=compute_20 -code=sm_21
CC_COMPILE_FLAGS = -O2 -I${CUDAINST}/include -I/usr/include/qt5/QtCore -I/usr/include/qt5/ -I/usr/include/qt5/QtGui -fPIC
CC_LINK_FLAGS = -lm -lstdc++ -L/usr/lib/x86_64-linux-gnu -lQt5Gui -lQt5Core -lpthread
HEADERS= kernel.h mandelframe.h mandelregion.h workqueue.h

all : mandelHybrid

mandelHybrid : main.o mandelframe.o mandelregion.o workqueue.o kernel.o
	${NVCC} ${CUDA_LINK_FLAGS} ${CC_LINK_FLAGS} $^ -o $@

main.o : main.cpp ${HEADERS}
	${CC} ${CC_COMPILE_FLAGS} -c main.cpp 

mandelframe.o : mandelframe.cpp ${HEADERS}
	${CC} ${CC_COMPILE_FLAGS} -c mandelframe.cpp 

mandelregion.o : mandelregion.cpp ${HEADERS}
	${CC} ${CC_COMPILE_FLAGS} -c mandelregion.cpp 

workqueue.o : workqueue.cpp ${HEADERS}
	${CC} ${CC_COMPILE_FLAGS} -c workqueue.cpp 

kernel.o : kernel.cu kernel.h
	${NVCC} -c ${CUDA_LINK_FLAGS} ${CC_LINK_FLAGS} kernel.cu -o $@

clean:
	rm *.o 