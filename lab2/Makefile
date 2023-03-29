CC = clang
CFLAGS = -Wall -O2 -I/opt/homebrew/include/gsl
LIBS = -lgsl -lgslcblas -lm

SRCS = main.c
OBJS = $(SRCS:.c=.o)
TARGET = program

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)
