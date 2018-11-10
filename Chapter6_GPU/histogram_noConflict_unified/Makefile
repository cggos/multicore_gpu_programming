all:	histogram

histogram:	histogram.cu pgm.o
	nvcc -arch=sm_30 histogram.cu pgm.o -o histogram

pgm.o:	../common/pgm.cpp
	g++ -c ../common/pgm.cpp -o ./pgm.o
