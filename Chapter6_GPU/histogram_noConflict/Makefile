all:	histogram_V2 histogram_atomic histogram_V1 histogram_V1_ILP

histogram_V2:	histogram_V2.cu pgm.o
	nvcc -arch=sm_30 histogram_V2.cu pgm.o -o histogram_V2

histogram_V1:	histogram_V1.cu pgm.o
	nvcc -arch=sm_30 histogram_V1.cu pgm.o -o histogram_V1

histogram_V1_ILP:	histogram_V1_ILP.cu pgm.o
	nvcc -arch=sm_30 histogram_V1_ILP.cu pgm.o -o histogram_V1_ILP

histogram_atomic:	histogram_atomic.cu pgm.o
	nvcc -arch=sm_30 histogram_atomic.cu pgm.o -o histogram_atomic

pgm.o:	../common/pgm.cpp 
	g++ -c ../common/pgm.cpp -o ./pgm.o
