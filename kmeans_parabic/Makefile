CC=mpicxx

CFLAGS=-O3 -Wall -Wno-sign-compare 

all: kmeanst

common.o: common.cc common.h
	$(CC) -o common.o -c -openmp $(CFLAGS) common.cc 

kmeanst.o: kmeanst.cc kmeanst.h
	$(CC) -o kmeanst.o -c -openmp $(CFLAGS) kmeanst.cc 

kmeanst: kmeanst.o common.o
	$(CC) -o kmeanst -openmp $(CFLAGS) common.o kmeanst.o

 
