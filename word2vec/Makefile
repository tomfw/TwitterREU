
# Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CC = gcc 
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -Wmaybe-uninitialized

# for Debugging purposes, please turn on the following version of CC and CFLAGS
# CC = gcc -g -o0 
# CFLAGS = -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec retrofit_word2vec_one 

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
retrofit_word2vec_one: retrofit_word2vec.c 
	$(CC) retrofit_word2vec.c -o retrofit_word2vec_one $(CFLAGS)

clean:
	rm -rf word2vec 
	rm -rf retrofit_word2vec_one
	
