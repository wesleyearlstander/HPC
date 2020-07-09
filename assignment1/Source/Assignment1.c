#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#define SWAP(x,y) t = x; x = y; y = t;

typedef struct Point {
    int val;
    float distance;
    float * dimensions;
} Point;

//ref = reference point array, m = size of ref, query = query point, d = dimensions
double par_EuclidianDistance (Point * ref, int m, Point query, int d) {
    double time = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic)
        for (size_t j = 0; j < m; j++) {
            float distance = 0;
            for (size_t i = 0; i < d; i++) {
                distance += pow(ref[j].dimensions[i] - query.dimensions[i], 2);
            }
            ref[j].distance = sqrt(distance);
        }
    return omp_get_wtime() - time;
}

//ref = reference point array, m = size of ref, query = query point, d = dimensions
double seq_EuclidianDistance (Point * ref, int m, Point query, int d) {
    double time = omp_get_wtime();
    for (size_t j = 0; j < m; j++) {
        float distance = 0;
        for (size_t i = 0; i < d; i++) {
            distance += pow(ref[j].dimensions[i] - query.dimensions[i], 2);
        }
        ref[j].distance = sqrt(distance);
    }
    return omp_get_wtime() - time; //time taken for calculation
}

//ref = reference point array, m = size of ref, query = query point, d = dimensions
double seq_ManhattanDistance (Point * ref, int m, Point query, int d) {
    double time = omp_get_wtime();
    for (size_t j = 0; j < m; j++) {
        float distance = 0;
        for (size_t i = 0; i < d; i++) {
            distance += abs(ref[j].dimensions[i] - query.dimensions[i]);
        }
        ref[j].distance = distance;
    }
    return omp_get_wtime() - time;
}

//ref = reference point array, m = size of ref, query = query point, d = dimensions
double par_ManhattanDistance (Point * ref, int m, Point query, int d) {
    double time = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic)
        for (size_t j = 0; j < m; j++) {
            float distance = 0;
            for (size_t i = 0; i < d; i++) {
                distance += abs(ref[j].dimensions[i] - query.dimensions[i]);
            }
            ref[j].distance = distance;
        }
    return omp_get_wtime() - time;
}

void validate_sort(int n, Point *data){
	int i;
	for(i=0;i<n-1;i++){
		if(data[i].distance > data[i+1].distance){
			printf("Validate failed. \n");
            return;
		}
	}
	printf("Validate passed.\n");
}

void validate_distance(int n, Point * data, Point * data2){
	int i;
	for(i=0;i<n-1;i++){
		if(data[i].distance != data2[i].distance){
			printf("Validate failed. \n");
            return;
		}
	}
	printf("Validate passed.\n");
}

void seq_qsortHelper(Point * data, int low, int high ) {
    int i = low, j = high;
	Point t;
	Point pivot = data[(i + j) / 2];

	while (i <= j) {
		while (data[i].distance < pivot.distance) i++;
		while (data[j].distance > pivot.distance) j--;
		if (i <= j) {
			SWAP(data[i], data[j])
			i++;
			j--;
		}
	}

	if (low < j){ seq_qsortHelper(data, low, j);  }
	if (i< high){ seq_qsortHelper(data, i, high); }
}

void parS_qsortHelper(Point * data, int low, int high, int low_limit ) {
    int i = low, j = high;
    Point t;
    Point pivot = data[(i + j) / 2];
    if (i < j)
    {
        while (i <= j) {
            while (data[i].distance < pivot.distance) i++;
            while (data[j].distance > pivot.distance) j--;
            if (i <= j) {
                SWAP(data[i], data[j]);
                i++;
                j--;
            }
        }
    }

    if ( ((high-low)<low_limit) ){
        if (low < j){ parS_qsortHelper(data, low, j, low_limit); }
        if (i < high){ parS_qsortHelper(data, i, high, low_limit); }

    }else{
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
                parS_qsortHelper(data, low, j, low_limit);
            #pragma omp section
                parS_qsortHelper(data, i, high, low_limit);
        }
    }
}

void parT_qsortHelper(Point * data, int low, int high, int low_limit ) {
    int i = low, j = high;
    Point t;
    Point pivot = data[(i + j) / 2];
    if (i < j)
    {
        while (i <= j) {
            while (data[i].distance < pivot.distance) i++;
            while (data[j].distance > pivot.distance) j--;
            if (i <= j) {
                SWAP(data[i], data[j]);
                i++;
                j--;
            }
        }
    }

    if ( ((high-low)<low_limit) ){
        if (low < j){ parT_qsortHelper(data, low, j, low_limit); }
        if (i < high){ parT_qsortHelper(data, i, high, low_limit); }

    }else{
        #pragma omp task
            parT_qsortHelper(data, low, j, low_limit);
        #pragma omp task
            parT_qsortHelper(data, i, high, low_limit);
    }
}

double parS_qsort(Point * data, int size, int low_limit ) {
    double time = omp_get_wtime();
    parS_qsortHelper(data, 0, size-1, low_limit);
    return omp_get_wtime() - time; //time for calculation
}


double parT_qsort(Point * data, int size, int low_limit) {
    double time = omp_get_wtime();
    #pragma omp parallel num_threads(2)
	{
		#pragma omp single nowait
		{
			parT_qsortHelper(data, 0, size-1, low_limit);
		}
	}
    return omp_get_wtime() - time; //time for calculation
}

//serial quick sort caller
double seq_qsort(Point * data, int size ) {
    double time = omp_get_wtime();
    seq_qsortHelper(data, 0, size-1);
    return omp_get_wtime() - time; //time for calculation
}

//bitonic merge function
void bitonicMerge (Point * data, int low, int count, int dir) {
    if (count>1)
    {
        int k = count/2;
        for (int i=low; i<low+k; i++){
            Point t; //required to use SWAP macro
            if (dir==(data[i].distance>data[i+k].distance)){
                SWAP(data[i], data[i+k]);
            }
        }
        bitonicMerge(data, low, k, dir);
        bitonicMerge(data, low+k, k, dir);
    }
}

//serial bitonic sort algorithm
void seq_bitonicSortHelper( Point * data,int low, int count, int dir)
{
    if (count>1)
    {
        int k = count/2;
        seq_bitonicSortHelper(data, low, k, 1);
        seq_bitonicSortHelper(data, low+k, k, 0);
        bitonicMerge(data, low, count, dir);
    }
}

//bitonic sort tasks layer recursion
void parT_bitonicSortHelper( Point * data,int low, int count, int dir, int low_limit)
{
    if (count>1)
    {
        int k = count/2;
        if (count-low <= low_limit) {
            seq_bitonicSortHelper(data, low, k, 1);
            seq_bitonicSortHelper(data, low+k, k, 0);
        } else {
            #pragma omp parallel num_threads(2)
            {
                #pragma omp single
                {
                    #pragma omp task
                        parT_bitonicSortHelper(data, low, k, 1, low_limit);
                    #pragma omp task
                        parT_bitonicSortHelper(data, low+k, k, 0, low_limit);
                }
            }
        }
        bitonicMerge(data, low, count, dir);
    }
}

//bitonic sort sections layer recursion
void parS_bitonicSortHelper( Point * data,int low, int count, int dir, int low_limit)
{
    if (count>1)
    {
        int k = count/2;
        if ((count-low) <= low_limit) {
            seq_bitonicSortHelper(data, low, k, 1);
            seq_bitonicSortHelper(data, low+k, k, 0);
        } else {
            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                    parS_bitonicSortHelper(data, low, k, 1, low_limit);
                #pragma omp section
                    parS_bitonicSortHelper(data, low+k, k, 0, low_limit);
            }
        }
        bitonicMerge(data, low, count, dir);
    }
}

//serial bitonic sort caller
double seq_bitonicSort( Point * data, int size)
{
    double time = omp_get_wtime();
    seq_bitonicSortHelper(data, 0, size, 1);
    return omp_get_wtime() - time; //time for calculation
}

//Bitonic sort sections caller
double parS_bitonicSort( Point * data, int size, int low_limit)
{
    double time = omp_get_wtime();
    int k = size/2;
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
            seq_bitonicSortHelper(data, 0, k, 1);
        #pragma omp section
            seq_bitonicSortHelper(data, k, k, 0);
    }
    bitonicMerge(data, 0, size, 1);
    return omp_get_wtime() - time;
}

//Bitonic sort tasks caller
double parT_bitonicSort( Point * data, int size, int low_limit)
{
    double time = omp_get_wtime();
    if (size>1)
    {
        int k = size/2;
        #pragma omp parallel num_threads(2) shared(data) firstprivate(k)
        {
            #pragma omp single
            {
                #pragma omp task
                    parT_bitonicSortHelper(data, 0, k, 1, low_limit);
                #pragma omp task
                    parT_bitonicSortHelper(data, 0+k, k, 0, low_limit);
            }
        }
        bitonicMerge(data, 0, size, 1);
    }
    return omp_get_wtime() - time;
}

double seq_insertionSort( Point * data, int size) {
    double time = omp_get_wtime();
    int j;
    Point key;
    for (size_t i = 1; i < size; i++) {
        key = data[i];
        j = i - 1;
        while (j >= 0 && data[j].distance > key.distance) {
            data[j+1] = data[j];
            j = j-1;
        }
        data[j+1] = key;
    }
    return omp_get_wtime() - time;
}

void seq_merge(Point * data, int left, int middle, int right)
{
    int i, j, k;
    int n1 = middle - left + 1;
    int n2 = right - middle;
    Point L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = data[left + i];
    for (j = 0; j < n2; j++)
        R[j] = data[middle + 1+ j];

    i = 0;
    j = 0;
    k = left;

    while (i < n1 && j < n2)
    {
        if (L[i].distance <= R[j].distance)
        {
            data[k] = L[i];
            i++;
        }
        else
        {
            data[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        data[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        data[k] = R[j];
        j++;
        k++;
    }
}

//Serial Merge sort algorithm
void seq_mergeSortHelper(Point * data, int left, int right)
{
    if (left < right)
    {
        int middle = left+(right-left)/2;
        seq_mergeSortHelper(data, left, middle);
        seq_mergeSortHelper(data, middle+1, right);
        seq_merge(data, left, middle, right);
    }
}

//Merge sort tasks layer recursion
void parT_mergeSortHelper(Point * data, int left, int right, int low_limit)
{
    if (left < right)
    {
        int mid = left+(right-left)/2;
        if (right - left <= low_limit) {
            seq_mergeSortHelper(data, left, mid);
            seq_mergeSortHelper(data, mid+1, right);
        } else {
            #pragma omp parallel num_threads(2)
            {
                #pragma omp single
                {
                    #pragma omp task
                        parT_mergeSortHelper(data, left, mid, low_limit);
                    #pragma omp task
                        parT_mergeSortHelper(data, mid+1, right, low_limit);
                }
            }
        }
        seq_merge(data, left, mid, right);
    }
}

//Merge sort sections layer recursion
void parS_mergeSortHelper(Point * data, int left, int right, int low_limit)
{
    if (left < right)
    {
        int mid = left+(right-left)/2;
        if ((right - left) <= low_limit) {
            seq_mergeSortHelper(data, left, right);
        } else {
            #pragma omp parallel sections num_threads(2) shared(data)
            {
                #pragma omp section
                    parS_mergeSortHelper(data, left, mid,low_limit);
                #pragma omp section
                    parS_mergeSortHelper(data, mid+1, right,low_limit);
            }
            seq_merge(data, left, mid, right);
        }
    }
}

//Serial merge sort caller
double seq_mergeSort(Point * data, int size)
{
    double time = omp_get_wtime();
    seq_mergeSortHelper(data, 0, size-1);
    return omp_get_wtime() - time;
}

//Merge sort sections caller
double parS_mergeSort(Point * data, int size, int low_limit)
{
    double time = omp_get_wtime();
    //seq_mergeSortHelper(data, 0, size-1);
    parS_mergeSortHelper(data, 0, size-1, low_limit);
    return omp_get_wtime() - time;
}

double parT_mergeSort(Point * data, int size, int low_limit)
{
    double time = omp_get_wtime();
    int middle = (size-1)/2;
    #pragma omp parallel num_threads(2) firstprivate(middle,size,low_limit) shared(data)
    {
        #pragma omp single
        {
            #pragma omp task
                parT_mergeSortHelper(data, 0, middle, low_limit);
            #pragma omp task
                parT_mergeSortHelper(data, middle+1, size-1, low_limit);
        }
        seq_merge(data, 0, middle, size-1);
    }
    return omp_get_wtime() - time;
}

int selectGroupByK (Point * data, int groups, int k) {
    int groupCount[groups];
    for (size_t i = 0; i < groups; i++) {
        groupCount[i] = 0;
    }
    for (size_t i = 0; i < k; i++) {
        groupCount[data[i].val]++;
    }
    int maxGroup = 0;
    for (size_t i = 1; i < groups; i++) {
        if (groupCount[i] > groupCount[maxGroup]) maxGroup = i;
    }
    return maxGroup;
}

void printArray(Point * arr, int size)
{
    int i;
    for (i=0; i < size; i++)
        printf("%f, %d \n", arr[i].distance, arr[i].val);
    printf("\n");
}

void copyArray(Point * og, Point * copy, int m, int d) {
    for (size_t i = 0; i < m; i++) {
        copy[i].val = og[i].val;
        for (size_t j = 0; j < d; j++) {
            copy[i].dimensions[j] = og[i].dimensions[j];
        }
        copy[i].distance = og[i].distance;
    }
}

int checkArray(Point * og, Point * copy, int m, int d) {
    for (size_t i = 0; i < m; i++) {
        if (og[i].val != copy[i].val) return 0;
        for (size_t j = 0; j < d; j++) {
            if (og[i].dimensions[j] != copy[i].dimensions[j]) return 0;
        }
        if (og[i].distance != copy[i].distance) return 0;
        if (&og[0] == &copy[0]) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]){
	int m = 32768*4, n = 400, d = 128, groups = 5;
    int low_limit= 4096;
    //reference = malloc(sizeof(Point) * m);
    //query = malloc(sizeof(Point) * n);
    omp_set_nested(1);

    Point * referenceOG = malloc(sizeof(Point) * m);
    Point * query = malloc(sizeof(Point) * n);
    Point * reference = malloc(sizeof(Point) * m);

    srand(time(NULL));
    for (size_t i = 0; i < m; i++) {
        referenceOG[i].val = rand() % groups;
        referenceOG[i].dimensions = malloc(sizeof(float) * d);
        reference[i].dimensions = malloc(sizeof(float) * d);
        for (size_t j = 0; j < d; j++) {
            referenceOG[i].dimensions[j] = rand();
        }
    }

    //srand(10);
    //for (size_t i = 0; i < n; i++) {
    //    query[i].dimensions = malloc(sizeof(float) * d);
    //    for (size_t j = 0; j < d; j++) {
    //        query[i].dimensions[j] = rand();
    //    }
    //}

    copyArray(referenceOG, reference, m, d);

    double sTime=0, sTime2=0, sTime3=0, sTime4=0;
    double sTime5 = 0, sTime6 = 0, sTime7 = 0, sTime8 = 0, sTime9 = 0;
    double dTime = 0, dTime2 = 0, dTime3 = 0, dTime4 = 0;
    /*for (size_t i = 0; i < n; i++) {
        dTime += seq_EuclidianDistance(referenceOG, m, query[i], d);
        dTime2 += seq_ManhattanDistance(referenceOG, m, query[i], d);
        dTime4 += par_ManhattanDistance(referenceOG, m, query[i], d);
        dTime3 += par_EuclidianDistance(referenceOG, m, query[i], d);
        for (size_t i = 0; i < m; i++) {
            free(reference[i].dimensions);
        }
        free(reference);
        reference = malloc(sizeof(Point) * m);
        for (size_t i = 0; i < m; i++) {
            reference[i].dimensions = malloc(sizeof(float) * d);
        }
        copyArray(referenceOG, reference, m, d);
        sTime += seq_qsort(reference, m);
        for (size_t i = 0; i < m; i++) {
            free(reference[i].dimensions);
        }
        free(reference);
        reference = malloc(sizeof(Point) * m);
        for (size_t i = 0; i < m; i++) {
            reference[i].dimensions = malloc(sizeof(float) * d);
        }
        copyArray(referenceOG, reference, m, d);
        sTime2 += parT_qsort(reference, m, low_limit);
        for (size_t i = 0; i < m; i++) {
            free(reference[i].dimensions);
        }
        free(reference);
        reference = malloc(sizeof(Point) * m);
        for (size_t i = 0; i < m; i++) {
            reference[i].dimensions = malloc(sizeof(float) * d);
        }
        copyArray(referenceOG, reference, m, d);
        sTime3 += parS_qsort(reference, m, m/4);
        for (size_t i = 0; i < m; i++) {
            free(reference[i].dimensions);
        }
        free(reference);
        reference = malloc(sizeof(Point) * m);
        for (size_t i = 0; i < m; i++) {
            reference[i].dimensions = malloc(sizeof(float) * d);
        }
        copyArray(referenceOG, reference, m, d);
        sTime4 += seq_mergeSort(reference, m);
        for (size_t i = 0; i < m; i++) {
            free(reference[i].dimensions);
        }
        free(reference);
        reference = malloc(sizeof(Point) * m);
        for (size_t i = 0; i < m; i++) {
            reference[i].dimensions = malloc(sizeof(float) * d);
        }
        copyArray(referenceOG, reference, m, d);
        sTime5 += parS_mergeSort(reference, m, low_limit);
        for (size_t i = 0; i < m; i++) {
            free(reference[i].dimensions);
        }
        free(reference);
        reference = malloc(sizeof(Point) * m);
        for (size_t i = 0; i < m; i++) {
            reference[i].dimensions = malloc(sizeof(float) * d);
        }
        copyArray(referenceOG, reference, m, d);
        sTime6 += parT_mergeSort(reference, m, m/4);
        for (size_t i = 0; i < m; i++) {
            free(reference[i].dimensions);
        }
        free(reference);
        reference = malloc(sizeof(Point) * m);
        for (size_t i = 0; i < m; i++) {
            reference[i].dimensions = malloc(sizeof(float) * d);
        }
        copyArray(referenceOG, reference, m, d);
        sTime7 += seq_bitonicSort(reference, m);
        for (size_t i = 0; i < m; i++) {
            free(reference[i].dimensions);
        }
        free(reference);
        reference = malloc(sizeof(Point) * m);
        for (size_t i = 0; i < m; i++) {
            reference[i].dimensions = malloc(sizeof(float) * d);
        }
        copyArray(referenceOG, reference, m, d);
        sTime8 += parS_bitonicSort(reference, m, m);
        for (size_t i = 0; i < m; i++) {
            free(reference[i].dimensions);
        }
        free(reference);
        reference = malloc(sizeof(Point) * m);
        for (size_t i = 0; i < m; i++) {
            reference[i].dimensions = malloc(sizeof(float) * d);
        }
        copyArray(referenceOG, reference, m, d);
        sTime9 += parT_bitonicSort(reference, m, low_limit);

        printf("first run %zu\n", i);
    }*/


    FILE *fp;
    /*fp = fopen("Distance.csv", "w+");
    fprintf(fp, "Euclidean, EuclideanPar, Manhattan, ManhattanPar\n");
    fprintf(fp, "%f,%f,%f,%f", dTime, dTime3, dTime2, dTime4);
    fclose(fp);

    fp = fopen("Qsort.csv", "w+");
    fprintf(fp, "Serial, Section, Task\n");
    fprintf(fp, "%f,%f,%f", sTime, sTime3, sTime2);
    fclose(fp);

    */double qsortSerialTotal =sTime+dTime,qsortTaskTotal = sTime2+dTime3, qsortSectionTotal = sTime3+dTime3;/*
    fp = fopen("QsortTotal.csv", "w+");
    fprintf(fp, "Serial, Section, Task, SerialDistPerc, SectionDistPerc, TaskDistPerc, SerialSortPerc, SectionSortPerc, TaskSortPerc\n");
    fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f", sTime+dTime, sTime3+dTime3, sTime2+dTime3,
    dTime/qsortSerialTotal*100, dTime3/qsortSectionTotal*100, dTime3/qsortTaskTotal*100,
    sTime/qsortSerialTotal*100, sTime3/qsortSectionTotal*100, sTime2/qsortTaskTotal*100);
    fclose(fp);

    fp = fopen("Merge.csv", "w+");
    fprintf(fp, "Serial, Section, Task\n");
    fprintf(fp, "%f,%f,%f", sTime4, sTime5, sTime6);
    fclose(fp);

    */double mergeSerialTotal =sTime4+dTime, mergeTaskTotal = sTime6+dTime3, mergeSectionTotal = sTime5+dTime3;/*
    fp = fopen("MergeTotal.csv", "w+");
    fprintf(fp, "Serial, Section, Task, SerialDistPerc, SectionDistPerc, TaskDistPerc, SerialSortPerc, SectionSortPerc, TaskSortPerc\n");
    fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f", sTime4+dTime, sTime5+dTime3, sTime6+dTime3,
    dTime/mergeSerialTotal*100, dTime3/mergeSectionTotal*100, dTime3/mergeTaskTotal*100,
    sTime4/mergeSerialTotal*100, sTime5/mergeSectionTotal*100, sTime6/mergeTaskTotal*100);
    fclose(fp);

    fp = fopen("Bitonic.csv", "w+");
    fprintf(fp, "Serial, Section, Task\n");
    fprintf(fp, "%f,%f,%f", sTime7, sTime8, sTime9);
    fclose(fp);

    */double bitonicSerialTotal =sTime7+dTime, bitonicTaskTotal = sTime9+dTime3, bitonicSectionTotal = sTime8+dTime3;/*
    fp = fopen("BitonicTotal.csv", "w+");
    fprintf(fp, "Serial, Section, Task, SerialDistPerc, SectionDistPerc, TaskDistPerc, SerialSortPerc, SectionSortPerc, TaskSortPerc\n");
    fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f", sTime7+dTime, sTime8+dTime3, sTime9+dTime3,
    dTime/bitonicSerialTotal*100, dTime3/bitonicSectionTotal*100, dTime3/bitonicTaskTotal*100,
    sTime7/bitonicSerialTotal*100, sTime8/bitonicSectionTotal*100, sTime9/bitonicTaskTotal*100);
    fclose(fp);

    dTime=dTime2=dTime3=dTime4=sTime=sTime2=sTime3=sTime4=sTime5=sTime6=sTime7=sTime8=sTime9=0;*/

    //fp = fopen("nQsort.csv", "w+");
    FILE * fp2;
    //fp2 = fopen("nMerge.csv", "w+");
    FILE * fp3;
    /*fp3 = fopen("nBitonic.csv", "w+");
    fprintf(fp, "n, Section, Task, SectionDistPerc, TaskDistPerc, SectionSortPerc, TaskSortPerc\n");
    fprintf(fp2, "n, Section, Task, SectionDistPerc, TaskDistPerc, SectionSortPerc, TaskSortPerc\n");
    fprintf(fp3, "n, Section, Task, SectionDistPerc, TaskDistPerc, SectionSortPerc, TaskSortPerc\n");

    for (size_t i = 0; i < n; i++) {
        free(query[i].dimensions);
    }
    free(query);

    d = 128;
    for (size_t j = 200; j <= 1000; j+=200) {
        if (j==1000) break;
        query = malloc(sizeof(Point) * j);
        srand(10);
        for (size_t i = 0; i < j; i++) {
            query[i].dimensions = malloc(sizeof(float) * d);
            for (size_t k = 0; k < d; k++) {
                query[i].dimensions[k] = rand();
            }
        }
        for (size_t i = 0; i < j; i++) {
            dTime3 += par_EuclidianDistance(referenceOG, m, query[i], d);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * d);
            }
            copyArray(referenceOG, reference, m, d);
            sTime2 += parT_qsort(reference, m, low_limit);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * d);
            }
            copyArray(referenceOG, reference, m, d);
            sTime3 += parS_qsort(reference, m, m/4);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * d);
            }
            copyArray(referenceOG, reference, m, d);
            sTime5 += parS_mergeSort(reference, m, low_limit);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * d);
            }
            copyArray(referenceOG, reference, m, d);
            sTime6 += parT_mergeSort(reference, m, m/4);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * d);
            }
            copyArray(referenceOG, reference, m, d);
            sTime8 += parS_bitonicSort(reference, m, m);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * d);
            }
            copyArray(referenceOG, reference, m, d);
            sTime9 += parT_bitonicSort(reference, m, low_limit);
            printf("nQsort %zu:%zu\n", j, i);
        }
        qsortTaskTotal = sTime2+dTime3;
        qsortSectionTotal = sTime3+dTime3;
        fprintf(fp, "%zu,%f,%f,%f,%f,%f,%f\n", j, sTime3+dTime3, sTime2+dTime3,
        dTime3/qsortSectionTotal*100, dTime3/qsortTaskTotal*100,
        sTime3/qsortSectionTotal*100, sTime2/qsortTaskTotal*100);
        mergeTaskTotal = sTime6+dTime3;
        mergeSectionTotal = sTime5+dTime3;
        fprintf(fp2, "%zu,%f,%f,%f,%f,%f,%f\n", j, sTime5+dTime3, sTime6+dTime3,
        dTime3/mergeSectionTotal*100, dTime3/mergeTaskTotal*100,
        sTime5/mergeSectionTotal*100, sTime6/mergeTaskTotal*100);
        bitonicTaskTotal = sTime9+dTime3;
        bitonicSectionTotal = sTime8+dTime3;
        fprintf(fp3, "%zu,%f,%f,%f,%f,%f,%f\n", j, sTime8+dTime3, sTime9+dTime3,
        dTime3/bitonicSectionTotal*100, dTime3/bitonicTaskTotal*100,
        sTime8/bitonicSectionTotal*100, sTime9/bitonicTaskTotal*100);
        for (size_t i = 0; i < j; i++) {
            free(query[i].dimensions);
        }
        free(query);
        dTime=dTime2=dTime3=dTime4=sTime=sTime2=sTime3=sTime4=sTime5=sTime6=sTime7=sTime8=sTime9=0;
    }
    fclose(fp2);
    fclose(fp3);
    fclose(fp);
    */
    fp = fopen("dQsort.csv", "w+");
    fp2 = fopen("dMerge.csv", "w+");
    fp3 = fopen("dBitonic.csv", "w+");
    fprintf(fp, "d, Section, Task, SectionDistPerc, TaskDistPerc, SectionSortPerc, TaskSortPerc\n");
    fprintf(fp2, "d, Section, Task, SectionDistPerc, TaskDistPerc, SectionSortPerc, TaskSortPerc\n");
    fprintf(fp3, "d, Section, Task, SectionDistPerc, TaskDistPerc, SectionSortPerc, TaskSortPerc\n");

    for (size_t i = 0; i < m; i++) {
        free(referenceOG[i].dimensions);
    }
                    //free(query);
                    //query = malloc(sizeof(Point) * n);

    int increase = 16;
    for (size_t j = 32; j <= 256; j+=increase) {

        srand(time(NULL));
        for (size_t i = 0; i < m; i++) {
            referenceOG[i].dimensions = malloc(sizeof(float) * j);
            for (size_t k = 0; k < j; k++) {
                referenceOG[i].dimensions[k] = rand();
            }
        }

        srand(10);
        for (size_t i = 0; i < n; i++) {
            query[i].dimensions = malloc(sizeof(float) * j);
            for (size_t k = 0; k < j; k++) {
                query[i].dimensions[k] = rand();
            }
        }

        for (size_t i = 0; i < j; i++) {
            dTime3 += par_EuclidianDistance(referenceOG, m, query[i], j);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * j);
            }
            copyArray(referenceOG, reference, m, j);
            sTime2 += parT_qsort(reference, m, low_limit);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * j);
            }
            copyArray(referenceOG, reference, m, j);
            sTime3 += parS_qsort(reference, m, m/4);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * j);
            }
            copyArray(referenceOG, reference, m, j);
            sTime5 += parS_mergeSort(reference, m, low_limit);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * j);
            }
            copyArray(referenceOG, reference, m, j);
            sTime6 += parT_mergeSort(reference, m, m/4);
            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * j);
            }
            copyArray(referenceOG, reference, m, j);
            sTime8 += parS_bitonicSort(reference, m, m);

            for (size_t i = 0; i < m; i++) {
                free(reference[i].dimensions);
            }
            free(reference);
            reference = malloc(sizeof(Point) * m);
            for (size_t i = 0; i < m; i++) {
                reference[i].dimensions = malloc(sizeof(float) * j);
            }
            copyArray(referenceOG, reference, m, j);
            sTime9 += parT_bitonicSort(reference, m, low_limit);
            printf("dQsort %zu:%zu\n", j, i);
        }

        double qsortTaskTotal = sTime2+dTime3, qsortSectionTotal = sTime3+dTime3;
        fprintf(fp, "%zu,%f,%f,%f,%f,%f,%f\n", j, sTime3+dTime3, sTime2+dTime3,
        dTime3/qsortSectionTotal*100, dTime3/qsortTaskTotal*100,
        sTime3/qsortSectionTotal*100, sTime2/qsortTaskTotal*100);
        double mergeTaskTotal = sTime6+dTime3, mergeSectionTotal = sTime5+dTime3;
        fprintf(fp2, "%zu,%f,%f,%f,%f,%f,%f\n", j, sTime5+dTime3, sTime6+dTime3,
        dTime3/mergeSectionTotal*100, dTime3/mergeTaskTotal*100,
        sTime5/mergeSectionTotal*100, sTime6/mergeTaskTotal*100);
        double bitonicTaskTotal = sTime9+dTime3, bitonicSectionTotal = sTime8+dTime3;
        fprintf(fp3, "%zu,%f,%f,%f,%f,%f,%f\n", j, sTime8+dTime3, sTime9+dTime3,
        dTime3/bitonicSectionTotal*100, dTime3/bitonicTaskTotal*100,
        sTime8/bitonicSectionTotal*100, sTime9/bitonicTaskTotal*100);
        increase *=2;

        for (size_t i = 0; i < n; i++) {
            free(query[i].dimensions);
        }
        for (size_t i = 0; i < m; i++) {
            free(referenceOG[i].dimensions);
        }
        dTime=dTime2=dTime3=dTime4=sTime=sTime2=sTime3=sTime4=sTime5=sTime6=sTime7=sTime8=sTime9=0;
    }
    fclose(fp2);
    fclose(fp3);
    fclose(fp);


    for (size_t i = 0; i < m; i++) {
        free(reference[i].dimensions);
    }
    free(reference);

    // for (size_t i = 0; i < m; i++) {
    //     free(referenceOG[i].dimensions);
    // }
    free(referenceOG);
    //
    // for (size_t i = 0; i < n; i++) {
    //     free(query[i].dimensions);
    // }
    free(query);

	return 0;
}
