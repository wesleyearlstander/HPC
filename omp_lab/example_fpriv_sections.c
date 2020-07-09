
#include <omp.h>
#include <stdio.h>
#define NT 4
int main( ) {
    int section_count = 0, id;
    omp_set_dynamic(0);
    omp_set_num_threads(NT);
#pragma omp parallel
#pragma omp sections firstprivate( section_count )
{
	//id=omp_get_thread_num();
#pragma omp section
    {
        section_count++;
        /* may print the number one or two */
        printf( "section_count %d from %d\n", section_count, omp_get_thread_num() );
    }
#pragma omp section
    {
        section_count++;
        /* may print the number one or two */
        printf( "section_count %d from %d\n", section_count, omp_get_thread_num() );
    }
#pragma omp section
    {
        section_count++;
        /* may print the number one or two */
        printf( "section_count %d from %d\n", section_count, omp_get_thread_num() );
    }
#pragma omp section
    {
        section_count++;
        /* may print the number one or two */
        printf( "section_count %d from %d\n", section_count, omp_get_thread_num() );
    }
}
    return 0;
}
