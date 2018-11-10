all : dla_sequential dla_sequential2 dla_mpi 

dla_sequential2 : dla_sequential2.c dla_core.o
	gcc -o dla_sequential2 dla_sequential2.c dla_core.o -lstdc++

dla_sequential : dla_sequential.c dla_core.o
	gcc -o dla_sequential dla_sequential.c dla_core.o -lstdc++

dla_mpi : dla_core.o dla_core.h dla_mpi.c
	mpicc -o dla_mpi dla_mpi.c dla_core.o

dla_core.o : dla_core.c dla_core.h
	gcc -c dla_core.c