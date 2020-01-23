
/*

  integral_mpi.c  --  trapezoidal rule function integration: MPI version

 */


# include <stdio.h>
# include <stdlib.h>
# include <mpi.h>
# include <math.h>


// ------------------------------------------------------------------- //
//                                                                     //
//     f -- function to integrate                                      //
//                                                                     //
// ------------------------------------------------------------------- //

double f (double x) {
  return 2.0 * sqrt( 1.0 - x * x );
}


// ------------------------------------------------------------------- //
//                                                                     //
//     trapInt_MPI -- function for trapezoidal integration             //
//                    (MPI parallel version)                           //
//                                                                     //
// ------------------------------------------------------------------- //

double trapInt_MPI (double a, double b, int N) {

  // global integral value
  double v;

  // number of processes, process rank
  int    iproc, nproc;

  // variables associated with local evaluation
  int    N1,N2;
  double a_proc,b_proc,v_proc;

  // ancillary variables
  int    i;
  double x,h;

  // complete the body of the function
  // ... //

  // interval length
  h = (b - a) / ((double) N);

  // Get the number of processes
  MPI_Comm_size (MPI_COMM_WORLD, &nproc);

  // Get the rank of the process
  MPI_Comm_rank (MPI_COMM_WORLD, &iproc);

  // Which trapezia the process will contribute to the integration
  N1 = (N* iproc)/ nproc;
  N2 = N*(iproc+1)/nproc;

  // The range of integration for this process
  a_proc = a + ((double) N1) * h;
  b_proc = a + ((double) N2) * h;

  // initial and final point only count with weight half
  v_proc = (f(a_proc) + f(b_proc)) / 2.0;

  // add the inner points
  for (i=N1+1; i<=N2-1; i++) {
    x = a + i*h;
    v_proc = v_proc + f(x);
  }

  // scale by the interval width
  v_proc *= h;

  MPI_Allreduce( &v_proc, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

  return v;
}


// ------------------------------------------------------------------- //
//                                                                     //
//                             M  A  I  N                              //
//                                                                     //
// ------------------------------------------------------------------- //

int main (int narg, char** varg) {

  // MPI variables
  int    iproc, nproc;

  // main variables
  int    N;                      // global:  number of intervals
  double a,b,v;                  // global:  ends of interval, integral value

  // timing variables
  double time_start, time_end, time_avg, time_standard_dev;
  const int num_trials=100;
  double trial_times[num_trials];


  //
  // ----- init MPI starts parallel independent processes
  //
  MPI_Init (&narg, &varg);


  //
  // ----- this process obtains total number of processes (nproc) and own number (proc)
  //
  MPI_Comm_size (MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank (MPI_COMM_WORLD, &iproc);


  //
  // ----- input
  //
  // Where should the integration start?
  a = -1.0;
  // Where should the integration end?
  b = 1.0;
  // How many intervals should the area be divided into?
  N = 100000;


  if (iproc==0){
	printf("Calculating integral from %.2f to %.2f using %d trapeziums.\n", a, b, N);
	printf("Using %d processes.\n", nproc);
  }



  for (int t=0; t<num_trials; t++){
    // start time
    time_start = MPI_Wtime ( );

    // evaluate the integral
    v = trapInt_MPI (a, b, N);

    // end time
    time_end = MPI_Wtime ( );
    trial_times[t] = time_end - time_start;
  }



  //
  // ----- print sum
  //
  if (iproc == 0) {
    // calculate timing stats
    double total_time=0;
    for (int t=0; t<num_trials; t++){
      total_time += trial_times[t];
    }
    time_avg = total_time/num_trials;
    total_time=0;
    for (int t=0; t<num_trials; t++){
      total_time += (trial_times[t]-time_avg)*(trial_times[t]-time_avg);
    }
    time_standard_dev = sqrt(total_time/num_trials);
    
    printf("Results:\n");
    printf(" Average process time is %fs with standard deviation %f over %d trials\n", time_avg, time_standard_dev, num_trials);
    printf(" Value of integral = %f\n", v);
  }


  //
  // ----- finalise MPI
  //
  MPI_Finalize ( );


  return 0;
}


/*
  end
 */

