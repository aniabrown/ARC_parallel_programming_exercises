
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX( A, B ) ( (A) > (B) ? (A) : (B) )
#define PI 3.14159265358979323846264338327950288419716939937510582

// -------------------------------------------------------------------- //
//                                                                      //
//     wall_clock_time -- wall clock time function                      //
//                                                                      //
// -------------------------------------------------------------------- //

double wall_clock_time (void) {

  # include <sys/time.h>
  # define MILLION 1000000.0

  double secs;
  struct timeval tp;

  gettimeofday (&tp,NULL);
  secs = (MILLION * (double) tp.tv_sec + (double) tp.tv_usec) / MILLION;
  return secs;

}


int main( void ) {

  /* Example program to solve the heat equation in 1D in serial */

  double nu;
  double *u, *uo;
  double du;
  double rms;

  /* For timing */
  double start_time, end_time;

  int n, n_time_steps;
  int L;
  int j, t;

  /* input size*/
  n = 8000;
  n_time_steps = 7000;
  nu = 0.49;

  /* Set the length of the box */
  L = n + 1;

  /* Increase n to allow space for the boundary points, one at the start and 
     one at the end */
  n = n + 2;

  /* Allocate memory for the arrays */
  u  = malloc( n * sizeof( *u  ) );
  if( !u ) {
    printf( "Failed to allocate the array u" );
    exit( EXIT_FAILURE );
  }
  uo = malloc( n * sizeof( *uo ) );
  if( !uo ) {
    printf( "Failed to allocate the array u" );
    exit( EXIT_FAILURE );
  }
  
  /* Now set the boundary conditions and initial values. These values are chosen 
     so we can compare with an exact solution of the equation to check our solution is 
     correct */

  /* Set the boundary conditions, i.e. the edges.
     As these are set by the physics these should not be changed 
     again by the program */
  u [ 0     ] = 0.0;
  uo[ 0     ] = 0.0;
  u [ n - 1 ] = 0.0;
  uo[ n - 1 ] = 0.0;

  /* Initial values to be solved on the grid */
  for( j = 1; j < n - 1; j++ )
    u[ j ] = sin( j * PI / L );

  /* All set up so now solve the equations at each time step*/

  /* Start the timer */
  start_time = wall_clock_time ( );

  /* Time loop */
  for (t=0; t<n_time_steps; t++) {

    /* Store old solution */
    for (j=1; j<n-1; j++) {
      uo[j] = u[j];
    }

    /* Now solve the equation */

    /* du is used to track the maximum change in u */
    du = 0.0;
    for (j=1; j<n-1; j++) {
      /* Finite difference scheme */
      u[j] = uo[j] + nu*(uo[j-1]-2.0*uo[j]+uo[j+1]);
      /* Calculate the maximum change in u */
      du = MAX( du, fabs( u[ j ] - uo[ j ] ) );
    }

    /* Occasionally report the maximum change as the temperature distribution 
       relaxes */
//    if( t%10 == 0 || t == n_time_steps - 1 )
//      printf( "At timestep %5i the maxmimum change in the solution is %-#14.8g\n",
//	      t, du );

  }

  /* Finish the timer */
  end_time = wall_clock_time ( );

  /* Check the solution against the exact, analytic answer, */
  /* by computing the root mean square error. */
  rms = 0.0;
  for (j=1; j<n-1; j++) {
    du = u[ j ] - sin( j * PI / L ) *  exp( - n_time_steps * nu * PI * PI / ( L * L ) );
    rms += du*du;
  }
  printf( "The RMS error in the final solution is %-#14.8g\n", sqrt(rms/((double) n)) );
  printf( "On 1 process the time taken was %f seconds\n", end_time - start_time );

  return EXIT_SUCCESS;

}
