
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "mpi.h"

#define MAX( A, B ) ( (A) > (B) ? (A) : (B) )
#define MIN( A, B ) ( (A) < (B) ? (A) : (B) )

#define PI 3.14159265358979323846264338327950288419716939937510582

int main( void ) {

  /* Example program to solve the heat equation in 1D using MPI */

  MPI_Status status;

  double nu;
  double *u, *uo;
  double du;
  double rms, rms_global;

  /* For timing */
  double start_time, end_time;

  int n_tot, n_time_steps;
  int L;
  int n_av, n_left;
  int my_n, my_first;
  int nprocs, my_rank;
  int hi_rank, lo_rank;
  int j, t;

  /* Initialise the MPI environment */
  MPI_Init( NULL, NULL );

  /* Find out the number of processes in the job and the rank of this process */
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs  );
  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

  /* input size*/
  n_tot = 8000;
  n_time_steps = 7000;
  nu = 0.49;
  
  if( my_rank == 0 ) {
    if( nu > 0.5 || nu < 0.0 ) {
      printf( "Sorry, the scheme implemented is unstable for nu=%f\n", nu );
      printf( "Please try again with 0<=nu<=0.5\n" );
      MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }
  }

  /* Set the length of the box */
  L = n_tot + 1;

  /* Each process only owns a subset of the points that can be varied
     Work out how big the subset owned by this process is */
  /* First work out on average how many point each process will own,
     and how many points that would leave over */
  n_av   = n_tot / nprocs;
  n_left = n_tot - n_av * nprocs;
  /* Now give this process the average number of points */
  my_n = n_av;
  /* And give the first n_left processes of the the points left over */
  if( my_rank < n_left )
    my_n = my_n + 1;

  /* Now on each of the process we have to allow room for both the
     boundary points AND the halos required for the parallel computation
     Let's say we have P processe each with ranks 0, 1, 2, ... P-1

     Rank 0 requires the points it can very, the lower boundary point (point
     zero in the whole problem) and an extra point for the upper halo 

     Rank 1 requires 1 extra point for a lower halo, and 1 extra point for the
     upper halo

     Ranks 2, 3, 4 ... P - 2 are just the same as rank 1

     Rank P - 1 requires an extra point for a lower halo, and an extra
     point for the boundary point at the end 

     So ALL ranks require 2 extra points over and above the points they
     vary! */

  my_n = my_n + 2;

  /* Allocate memory for the arrays */
  u  = malloc( my_n * sizeof( *u  ) );
  if( !u ) {
    printf( "Failed to allocate the array u" );
    MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
  }
  uo = malloc( my_n * sizeof( *uo ) );
  if( !uo ) {
    printf( "Failed to allocate the array u" );
    MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
  }
  
  /* Now set the boundary conditions and initial values. These values are chosen 
     so we can compare with an exact solution of the equation to check our solution is 
     correct */

  /* Set the boundary conditions, i.e. the edges.
     As these are set by the physics these should not be changed 
     again by the program */
  /* Rank 0 holds the left hand boundary condition */
  if( my_rank == 0 ) {
    u [ 0     ] = 0.0;
    uo[ 0     ] = 0.0;
  }
  /* Rank nprocs - 1 hold the right hand boundary condition */
  if( my_rank == nprocs - 1 ) {
    u [ my_n - 1 ] = 0.0;
    uo[ my_n - 1 ] = 0.0;
  }

  /* Initial values to be solved on the grid */
  /* For this we need to work out what is the global index of the
     first point on the grid assigned to this process */
  /* First cope with the average case */
  my_first = n_av * my_rank;
  /* Now cope with the left over points. How does this work?
     Let's say we had 2 left over points. These will be assigned to 
     ranks 0, 1 and 2

     The start point for rank zero is still the first point

     The start point for rank 1 is 1 more than expected, due to 
     the extra point with rank 0

     And for rank 2 it is shifted over by 2, 1 for the extra point on 
     rank 0, one for the point on rank 1

     We've now "used up" all our left over points so all other ranks 
     are like rank 2, i.e. 2 points shifted over */
  my_first = my_first + MIN( my_rank, n_left );
  for( j = 1; j < my_n - 1; j++ )
    u[ j ] = sin( ( j + my_first ) * PI / L );

  /* Finally for the parallel case we want to work out which ranks are this
     processes' neighbour. Small problem here as ranks 0 and nprocs-1 only
     have 1 neighbour, while all the rest have 2. However we can simplify the
     code that follows by using a special value provided by MPI for a rank that indicates that
     the process doesn't really exist i.e. any messages sent to a rank with this
     value will be thrown away, and any recieved will be ignored */
  if( my_rank != 0 ) 
    lo_rank = my_rank - 1;
  else
    lo_rank = MPI_PROC_NULL;

  if( my_rank != nprocs - 1 ) 
    hi_rank = my_rank + 1;
  else
    hi_rank = MPI_PROC_NULL;

  /* All set up so now solve the equations at each time step */

  /* Start the timer */
  start_time = MPI_Wtime();

  /* Time loop */
  for (t=0; t<n_time_steps; t++) {

    /* Store old solution */
    for (j=1; j<my_n-1; j++) {
      uo[j] = u[j];
    }

    /* Set up the halos in uo, that is uo[ 0 ] and uo[ my_n - 1 ] by sending the 
       relevant data in u from the neighbouring processes.*/

    /* First if we send our last element to the high node that corresponds to what we will be 
       recving into our lower halo from the lower node */
    MPI_Sendrecv( &uo[ my_n - 2 ], 1, MPI_DOUBLE, hi_rank, 20, 
		  &uo[ 0 ]       , 1, MPI_DOUBLE, lo_rank, 20, MPI_COMM_WORLD,
		  &status );
    /* Now recv into our upper halo from the node above, and send out our first element */
    MPI_Sendrecv( &uo[ 1 ]       , 1, MPI_DOUBLE, lo_rank, 10, 
		  &uo[ my_n - 1 ], 1, MPI_DOUBLE, hi_rank, 10, MPI_COMM_WORLD,
		  &status );

    /* Now solve the equation */

    /* du is used to track the maximum change in u */
    du = 0.0;
    for (j=1; j<my_n-1; j++) {
      /* Finite difference scheme */
      u[j] = uo[j] + nu*(uo[j-1]-2.0*uo[j]+uo[j+1]);
      /* Calculate the LOCAL maximum change in u */
      du = MAX( du, fabs( u[ j ] - uo[ j ] ) );
    }

    /* Occasionally report the maximum change as the temperature distribution 
       relaxes */
//    if( t%10 == 0 || t == n_time_steps - 1 ) {
//      /* Use a reduction to find the GLOBAL maximum change from the local changes */
//      MPI_Allreduce( &du, &du_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
//      /* Make sure only rank zero prints something */
//      if( my_rank == 0 )
//	printf( "At timestep %5i the maxmimum change in the solution is %-#14.8g\n",
//		t, du_global );
//    }
  }

  /* Finish the time */
  end_time = MPI_Wtime();

  /* Check the solution against the exact, analytic answer */
  rms = 0.0;
  for (j=1; j<my_n-1; j++) {
    du = u[ j ] -  sin( ( my_first + j ) * PI / L ) * exp( - n_time_steps * nu * PI * PI / ( L * L ) );
    rms += du*du;
  }
  /* reduction on rms */
  MPI_Reduce( &rms, &rms_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
  /* print rms error */
  if( my_rank == 0 ) {
    printf( "The RMS error in the final solution is %-#14.8g\n", sqrt(rms_global/((double) n_tot)) );
    printf( "On %d processes the time taken was %f seconds\n", nprocs, end_time - start_time );
  }

  /* Finalize the MPI environment */
  MPI_Finalize();

  return EXIT_SUCCESS;

}
