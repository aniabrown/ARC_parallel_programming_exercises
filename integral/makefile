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

default:	integral_mpi.c
		$(MPI_CC) $(INCS) $(CFLAGS) -o integral_mpi integral_mpi.c $(LIB) 

clean:
		/bin/rm -f *.o *.c~ *.h~ makefile~ integral_mpi
