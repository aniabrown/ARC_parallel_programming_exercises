#======================================================================#
#                                                                      #
#      name:     Makefile                                              #
#                                                                      #
#======================================================================#

# serial compiler
CC  = gcc

# MPI compiler wrapper
MPI_CC = mpicc

# flags
CFLAGS = -O3 -mavx -std=c99 -Wall -Wextra -pedantic
CFLAGS_OMP = -fopenmp

# libraries
LIBS = -lm

# rules
default:	all
all:    	heat_serial heat_mpi

heat_serial:	heat_serial.o
		$(CC) $(CFLAGS) -o heat_serial heat_serial.o $(LIBS)
heat_mpi:	heat_mpi.o
		$(MPI_CC) $(CFLAGS) -o heat_mpi heat_mpi.o $(LIBS)


heat_serial.o:	heat_serial.c
		$(CC) $(CFLAGS) -c heat_serial.c
heat_mpi.o:	heat_mpi.c
		$(MPI_CC) $(CFLAGS) -c heat_mpi.c

clean:
		/bin/rm -f *.o *.c~ *.h~ makefile~ heat_serial heat_mpi
