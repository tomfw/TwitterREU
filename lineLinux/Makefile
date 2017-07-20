#!/bin/sh
CC = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result

all: line reconstruct normalize concatenate 

line : line.cpp
	$(CC) -I /usr/local/include -L /usr/local/lib line.cpp -o line -lgsl -lgslcblas $(CFLAGS)
reconstruct: reconstruct.cpp
	$(CC) reconstruct.cpp -o reconstruct $(CFLAGS)
normalize: normalize.cpp
	$(CC) normalize.cpp -o normalize $(CFLAGS)
concatenate: concatenate.cpp
	$(CC) concatenate.cpp -o concatenate

clean:
	rm -rf line reconstruct normalize concatenate
