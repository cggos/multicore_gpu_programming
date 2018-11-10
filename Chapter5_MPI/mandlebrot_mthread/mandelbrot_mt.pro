SOURCES+=mandelbrot_mt.cpp
HEADERS+=sharedQue.h
CONFIG+=qt 
TARGET=mandelbrot_mt
QMAKE_CXX=mpic++
QMAKE_CC=mpicc
QMAKE_LINK=mpic++
