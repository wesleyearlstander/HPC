#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define TRUE  1
#define FALSE 0

int main()
{
   omp_set_dynamic(FALSE);
   if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
   omp_set_num_threads(4);

#pragma omp parallel 
   {
     printf("The parallel region is executed by thread %d\n",
        omp_get_thread_num());

     if ( omp_get_thread_num() == 2 ) {
        printf("  Thread %d does things differently\n",
               omp_get_thread_num());
     }
   }  /*-- End of parallel region --*/

   return(0);
}
