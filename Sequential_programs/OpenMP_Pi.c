
#include <stdio.h>
#include <omp.h>

int main()
{
    double start, stop;

    const size_t N = 100000;
    double step;

    double x, pi, sum = 0.;

    step = 1. / (double)N;

    start = omp_get_wtime();

    #pragma omp parallel for private(x) reduction(+:sum)
    for (int i = 0; i < N; ++i)
    {
        x = (i + 0.5) * step;
        sum += 4.0 / (1. + x * x);
    }

    pi = step * sum;

    stop = omp_get_wtime();

    printf("pi = %.16f\n", pi);
    printf("total number of threads: %d\n", omp_get_max_threads());
    printf("total time: %f\n", stop - start);

    return 0;
}
