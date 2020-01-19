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
LIB        = -lm

# includes
INCS =

default: 	all

all: 		integral_serial integral_mpi

integral_serial:	integral.c
		$(CC) $(INCS) $(CFLAGS) -o integral integral.c $(LIB) 

integral_mpi:	integral_mpi.c
		$(MPI_CC) $(INCS) $(CFLAGS) -o integral_mpi integral_mpi.c $(LIB) 

clean:
		/bin/rm -f *.o *.c~ *.h~ makefile~ integral_serial integral_mpi
