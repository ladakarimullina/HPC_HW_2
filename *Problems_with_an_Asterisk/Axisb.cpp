
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

const int ITERATION_LIMIT = 1000;

int main() {
    // Initialize the matrix
    std::vector<std::vector<double>> A = {{10., -1., 2., 0.},
                                          {-1., 11., -1., 3.},
                                          {2., -1., 10., -1.},
                                          {0.0, 3., -1., 8.}};
    
    // Initialize the RHS vector
    std::vector<double> b = {6., 25., -11., 15.};

    // Print the system
    std::cout << "System:" << std::endl;
    for (size_t i = 0; i < A.size(); ++i) {
        std::cout << A[i][0] << "*x1";
        for (size_t j = 1; j < A[i].size(); ++j) {
            std::cout << " + " << A[i][j] << "*x" << j + 1;
        }
        std::cout << " = " << b[i] << std::endl;
    }
    std::cout << std::endl;

    std::vector<double> x(b.size(), 0.0);

    double start_time = omp_get_wtime(); // Start timing



    for (int it_count = 0; it_count < ITERATION_LIMIT; ++it_count) {
        
        std::vector<double> x_new(x.size(), 0.0);
        bool converged = true;

        #pragma omp parallel for shared(x, x_new, A, b) reduction(&&:converged)
        for (size_t i = 0; i < A.size(); ++i) {

            double s1 = 0.0, s2 = 0.0;
            for (size_t j = 0; j < i; ++j) {
                s1 += A[i][j] * x[j];
            }
            for (size_t j = i + 1; j < A[i].size(); ++j) {
                s2 += A[i][j] * x[j];
            }
            x_new[i] = (b[i] - s1 - s2) / A[i][i];

            // Check for convergence
            if (std::abs(x_new[i] - x[i]) > 1e-10) {
                converged = false;
            }
        }

        if (converged) {
            break;
        }

        x = x_new;
    }
    double end_time = omp_get_wtime(); // End timing

    std::cout << "Solution: " << std::endl;
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Error:" << std::endl;
    for (size_t i = 0; i < A.size(); ++i) {
        double error = 0.0;
        for (size_t j = 0; j < A[i].size(); ++j) {
            error += A[i][j] * x[j];
        }
        error -= b[i];
        std::cout << error << " ";
    }
    std::cout << std::endl;

    // Print elapsed time
    std::cout << "total time: " << end_time - start_time << " seconds" << std::endl;

    return 0;
}
