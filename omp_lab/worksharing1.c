#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define TRUE  1
#define FALSE 0


int main()
{
   int i, n = 9;

   int a[n], b[n];

  omp_set_num_threads(4);

#pragma omp parallel default(none) shared(n,a,b) private(i)
{
   #pragma omp single
      printf("First for-loop: number of threads is %d\n",
             omp_get_num_threads());

   #pragma omp for schedule(runtime)
   for (i=0; i<n; i++)
   {
      printf("Thread %d executes loop iteration %d\n",
             omp_get_thread_num(),i);
      a[i] = i;
   }

   #pragma omp single
      printf("Second for-loop: number of threads is %d\n",
             omp_get_num_threads());

   #pragma omp for schedule(runtime)
   for (i=0; i<n; i++)
   {
      printf("Thread %d executes loop iteration %d\n",
             omp_get_thread_num(),i);
      b[i] = 2 * a[i];
   }
} /*-- End of parallel region --*/

   return(0);
}
