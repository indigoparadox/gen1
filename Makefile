
CFLAGS := -g -pg -lpthread -lm -Wall -Werror

gen1: gen1.o main.o util.o
	$(CC) -o $@ $^ $(CFLAGS)

.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)
