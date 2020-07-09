/* File:     mpi_trapezoid_1.c
 * Purpose:  Use MPI to implement a parallel version of the trapezoidal 
 *           rule. 
 *
 * Input:    None.
 * Output:   Estimate of the integral from a to b of f(x)
 *           using the trapezoidal rule and n trapezoids.
 *
 * Compile:  mpicc -g -Wall -o mpi_trape1 mpi_trapezoid_1.c
 * Run:      mpiexec -n <number of processes> ./mpi_trape1 \
 *                <starting point> <end point> <no of trpezoids>
 *
 * Algorithm:
 *    1.  Each process calculates "its" interval of
 *        integration.
 *    2.  Each process estimates the integral of f(x)
 *        over its interval using the trapezoidal rule.
 *    3a. Each process != 0 sends its integral to 0.
 *    3b. Process 0 sums the calculations received from
 *        the individual processes and prints the result.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void Usage(char prog_name[]);
double f(double x);
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len);
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p);

int main(int argc, char* argv[]) {
	int my_rank, comm_sz, n, local_n;
	double a, b, h, local_a, local_b;
	double local_int, total_int;
	double start, end;
	int source;
	
	/* Let the system do what it needs to start up MPI */	
	MPI_Init(&argc, &argv);
	/* Get my process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* Find out how many processes are being used */
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	if(my_rank==0){
		if(argc < 4){
			Usage(argv[0]);
		} else {
			a = atof(argv[1]);
			b = atof(argv[2]);
			n = atoi(argv[3]);
		}
	}
	start = MPI_Wtime();
	Get_input(my_rank, comm_sz, &a, &b, &n);
	h = (b-a)/n; /* h is the same for all processes */
	local_n = n/comm_sz; /* So is the number of trapezoids */	
	/* Length of each process' interval of
  * integration = local_n*h.  So my interval
  * starts at: */
	local_a = a + my_rank*local_n*h;
	local_b = local_a + local_n*h;
	local_int = Trap(local_a, local_b, local_n, h);
	/* Add up the integrals calculated by each process */
	if(my_rank != 0) {
		MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	} else {
		total_int = local_int;
		for(source = 1; source < comm_sz; source++) {
			MPI_Recv(&local_int, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			total_int += local_int;
		}
	}
	end = MPI_Wtime();
	if(my_rank == 0) {
		printf("With n = %d trapezoids, our estimate of ", n);
		printf("the integral\n from %f to %f = %.15e ", a, b, total_int);
		printf("in %.6fs\n", (end-start));
	}
	/* Shut down MPI */
	MPI_Finalize();
	return 0;
}/* main */


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double f(double x){
	return 4/(1+x*x);
}


/*------------------------------------------------------------------
 * Function:     Trap
 * Purpose:      Serial function for estimating a definite integral 
 *               using the trapezoidal rule
 * Input args:   left_endpt
 *               right_endpt
 *               trap_count 
 *               base_len
 * Return val:   Trapezoidal rule estimate of integral from
 *               left_endpt to right_endpt using trap_count
 *               trapezoids
 */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len) {
	double estimate, x;
	int i;
	estimate = (f(left_endpt) + f(right_endpt))/2.0;
	
	for(i = 1; i <= trap_count - 1; i++) {
		x = left_endpt + i*base_len;
		estimate += f(x);
	}
	estimate = estimate*base_len;
	return estimate;
} /* Trap */

void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p) { 
	int dest;
	if(my_rank == 0) {
		for(dest = 1; dest < comm_sz; dest++){
			MPI_Send(a_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
			MPI_Send(b_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
			MPI_Send(n_p, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		}
	} else {/* my rank != 0 */
		MPI_Recv(a_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(b_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(n_p, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
} /* Get_input */

void Usage(char prog_name[]) {
   fprintf(stderr, "usage: %s <left point> <right point> <number of tapezoids>\n",
         prog_name);
   exit(0);
} /* Usage */





