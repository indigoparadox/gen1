
CC := gcc
CFLAGS := -g -pg -lpthread -lm
OBJECTS := gen1.o util.o

gen1cuda: CFLAGS += -lcudart -DUSE_CUDA
gen1cuda: CC := nvcc

all:

testgen: $(OBJECTS) ../tests/check.o ../tests/check_gen.o
	$(CC) -o $@ $^ $(CFLAGS) -lcheck_pic -lsubunit -lm -lrt -lpthread

gen1: $(OBJECTS) main.o
	$(CC) -o $@ $^ $(CFLAGS)

gen1cuda: $(OBJECTS) main.o gen1cuda.o
	$(CC) -o $@ $^ $(CFLAGS)

.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

gen1cuda.o: gen1cuda.cu
	$(CC) -c $< -o $@ $(CFLAGS)

.PHONY: clean

clean:
	rm *.o testgen gen1

