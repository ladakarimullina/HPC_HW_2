#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

int main()
{
    const int n = 10;
    double x = 0.;
    double dx = 0.01;
    double * y;
    y = (double *) malloc(n*sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        x = i*dx;
        printf("Thread %d: %f, dx= %f\n", omp_get_thread_num(), x, dx); 
        y[i] = exp(x)*cos(x)*sqrt(6*x+5);
    }
    printf("after calculations x = %f\n", x);
    return 0;
}
