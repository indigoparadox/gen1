
CFLAGS := -g -pg -lpthread -lm -Wall -Werror
OBJECTS := gen1.o util.o

all: gen1 testgen

testgen: $(OBJECTS) ../tests/check.o ../tests/check_gen.o
	$(CC) -o $@ $^ $(CFLAGS) -lcheck_pic -lsubunit -lm -lrt -lpthread

gen1: $(OBJECTS) main.o
	$(CC) -o $@ $^ $(CFLAGS)

.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)
