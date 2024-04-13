
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

const int N = 1000;
const double alpha = 0.001;

// Function f(x,a,b)
double f(double x, double a, double b) {
    return a * x + b;
}

// Compute the sum of squared residuals
/*double computeResidualSum(const std::vector<double>& x, const std::vector<double>& y, double a, double b) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < x.size(); ++i) {
        double residual = y[i] - f(x[i], a, b);
        sum += residual * residual;
    }
    return sum;
}*/

// Gradient descent to optimize parameters a and b
void gradientDescent(const std::vector<double>& x, const std::vector<double>& y, double& a, double& b) {
    for (int iter = 0; iter < N; ++iter) {
        double gradient_a = 0.0;
        double gradient_b = 0.0;

        #pragma omp parallel for reduction(+:gradient_a, gradient_b)
        for (size_t i = 0; i < x.size(); ++i) {
            double residual = y[i] - f(x[i], a, b);
            gradient_a += -2.0 * x[i] * residual;
            gradient_b += -2.0 * residual;
        }

        a -= alpha * gradient_a;
        b -= alpha * gradient_b;
    }
}

int main() {

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y;

    double a = 2.0; // True parameter value for 'a'
    double b = 3.0; // True parameter value for 'b'

    for (double xi : x) {
        y.push_back(a * xi + b + ((double)rand() / RAND_MAX - 0.5)); // Add noise
    }

    // Initial guess for parameters a and b
    double initial_a = 1.0;
    double initial_b = 1.0;

    double start_time = omp_get_wtime(); // Start timing

    // Perform gradient descent to find optimal parameters
    gradientDescent(x, y, initial_a, initial_b);

    double end_time = omp_get_wtime(); // End timing

    // Output the optimized parameters
    std::cout << "Optimized parameters:" << std::endl;
    std::cout << "a = " << initial_a << std::endl;
    std::cout << "b = " << initial_b << std::endl;

    // Print elapsed time
    std::cout << "total time: " << end_time - start_time << " seconds" << std::endl;

    return 0;
}
