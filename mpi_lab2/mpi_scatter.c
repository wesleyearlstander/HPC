#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#define SIZE 4

int main(int argc, char *argv[]){
	int numtasks, rank, sendcount, recvcount, source;
	float sendbuf[SIZE][SIZE];
	float recvbuf[SIZE];

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	if(rank==0){
		for(int i=0; i < SIZE; i++)
			for(int j=0; j< SIZE; j++){
				sendbuf[i][j] =i*SIZE+j;
			}				
	}
	if (numtasks == SIZE) {
  		source = 1;
  		sendcount = SIZE;
  		recvcount = SIZE;
  		MPI_Scatter(sendbuf,sendcount,MPI_FLOAT,recvbuf,recvcount,
             MPI_FLOAT,source,MPI_COMM_WORLD);

  		printf("rank= %d  Results: %f %f %f %f\n",rank,recvbuf[0],
         recvbuf[1],recvbuf[2],recvbuf[3]);
 	 }
	else
  		printf("Must specify %d processors. Terminating.\n",SIZE);

	MPI_Finalize();
}
