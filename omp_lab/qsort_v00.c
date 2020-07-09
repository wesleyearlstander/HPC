
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#define BILLION  1000000000L;


int partition(int p, int r, float *data){
	float x=data[p];
	int k=p;
	int l=r+1;
	float t;
	while(1){
		do{
			k++;
		}while((data[k] <= x) && (k<r));
		do{
			l--;
		}while(data[l] > x);
		while(k<l){
			t=data[k];
			data[k]=data[l];
			data[l]=t;
			do{
				k++;
			}while(data[k]<=x);
			do{
				l--;
			}while(data[l]>x);
		}
		t=data[p];
		data[p]=data[l];
		data[l]=t;
		return l;
	}
}

void seq_qsort(int p, int r, float *data){
	if(p < r){
		int q=partition(p, r, data);
		seq_qsort(p, q-1, data);
		seq_qsort(q+1, r, data);
	}
}

void q_sort(int p, int r, float *data, int low_limit){
	if(p<r){
		if((r-p)<low_limit){
			seq_qsort(p,r,data);
		}
		else{
			int q=partition(p,r,data);
			#pragma omp parallel
			{
				#pragma omp single
				{
					#pragma omp task
					q_sort(p,q-1,data,low_limit);
					#pragma omp task
					q_sort(q+1,r,data,low_limit);
				}
			}
		}
	}
}

void validate_sort(int n, float *data){
	int i;
	for(i=0;i<n-1;i++){
		if(data[i] > data[i+1]){
			printf("Validate failed. \n");
		}
	}
	printf("Validate passed.\n");
}

int main(int argc, char *argv[]){
	int i, n, low_limit;
	float *data;
	double accum;
	struct timespec start, end;
	if(argc != 3){
		printf("a.out num_elems low_limit\n");
		return 1;
	}
	n=atoi(argv[1]);
	low_limit=atoi(argv[2]);
	/*generate the array*/
	data=(float*)malloc(sizeof(float)*n);
	for(i=0; i<n; i++){
		data[i]=rand();
	}
	printf("\nSorting %d numbers sequentially...\n\n", n);
	if(clock_gettime(CLOCK_REALTIME, &start) == -1){
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
    }
	q_sort(0,n-1,&data[0],low_limit);
	double start_time = omp_get_wtime();
	seq_qsort(0,n-1,&data[0]);
	double end_time = omp_get_wtime();
	if( clock_gettime(CLOCK_REALTIME, &end) == -1){
      perror( "clock gettime" );
      exit( EXIT_FAILURE );
    }
	accum=(end.tv_sec-start.tv_sec)+(double)(end.tv_nsec-start.tv_nsec)/BILLION;
	printf("Time: %lf s\n", accum/(end_time-start_time));
	printf("Done\n");
	validate_sort(n, &data[0]);
	free(data);
	return 0;
}
