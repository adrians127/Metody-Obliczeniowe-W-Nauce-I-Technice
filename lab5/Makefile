CC = gcc

all:
	make main

main: main.c
	gcc -Wall -I/home/adrians/gsl/include -c main.c
	gcc -L/home/adrians/gsl/lib main.o -lgsl -lgslcblas -lm