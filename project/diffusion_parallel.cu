#include <omp.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "utils.cuh"

bool isin(std::vector<double> v, double x)
{
    return std::find(v.begin(), v.end(), x) != v.end();
}

std::ostream& filewrite(std::ostream& os, double* arr1, int len1, double* arr2, int len2)
{
    char buf[10];
    for (int i = 0; i < len1; ++i)
    {
       sprintf(buf, "%.3f", arr1[i]);
       os << std::string(buf) <<" ";
    }

	os << std::endl;

    for (int j = 0; j < len2; ++j)
    {
       sprintf(buf, "%.3f", arr2[j]);
       os << std::string(buf) <<" ";
    }

    return os;
}

// Check errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Function to initialize concentration profiles
void initialize_profiles(double* n_left, double* n_right, int ny_left, int ny_right, double y_max, double dy, double init_prof_shift, double C) {
    double p = -y_max;
    for (int j = 0; j < ny_left; j++) {
        n_left[j] = (1 - tanhf((p - init_prof_shift) * 10)) / 2;
        p += dy;
    }

    p = -dy;
    for (int j = 0; j < ny_right; j++) {
        n_right[j] = (1 - tanhf((p - init_prof_shift) * 10)) / 2;
        p += dy;
    }
}

__global__ void first_pass(double* lower, double* diag, 
                                 double* upper, double* rhs, int n, int  m) {
    int gid = getGid();
    //printf("hi from %d", gid);
    if ((gid >= n) || ((int)(gid / m) % 2 == 0)) {
        return;
    }
    //int i = gid / (2*m);
    int j = gid;
    double factor = lower[j] / diag[j - m];
    diag[j] -= factor * upper[j - m];
    rhs[j] -= factor * rhs[j - m];
    lower[j] = -factor * lower[j - m];
}

__global__ void second_pass(double* lower, double* diag, 
                                  double* upper, double* rhs, int n, int m) {
    int gid = getGid();
    if ((gid >= n) || ((int)(gid / m) % 2 != 0)) {
        return;
    }
    //int i = gid / (2*m);
    int j = gid;
    double factor = upper[j - m] / diag[j - m];
    diag[j - m] -= factor * lower[j];
    rhs[j - m] -= factor * rhs[j];
    upper[j - m] = -factor * upper[j];
}

__global__ void finalize_rhs(double* rhs, double* diag, int n) {
    int gid = getGid();
    if (gid < n) {
        rhs[gid] /= diag[gid];
    }
}

// Function to solve tridiagonal system using Parallel Cyclic Reduction (PCR)
__device__ void parallel_cyclic_reduction(double* lower, double* diag, double* upper, double* rhs, int n) {
    int log2n = __log2f(n);
    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << (log2n - s);
        
        first_pass<<<n/(2*m*1024)+1, 1024>>>(lower, diag, upper, rhs, n, m);
        second_pass<<<n/(2*m*1024)+1, 1024>>>(lower, diag, upper, rhs, n, m);
    }

    for (int s = log2n; s > 0; --s) {
        int m = 1 << (log2n - s);
        first_pass<<<n/(2*m*1024)+1, 1024>>>(lower, diag, upper, rhs, n, m);
        second_pass<<<n/(2*m*1024)+1, 1024>>>(lower, diag, upper, rhs, n, m);
    }
    finalize_rhs<<<(n/1024)+1, 1024>>>(rhs, diag, n);
}


__global__ void run_iteration(double* B_left, double* lower_left, double* upper_left, double* diag_left, double* n_left, 
                            double* B_right, double* lower_right, double* upper_right, double* diag_right, double* n_right, 
                            int ny_left, float s_left, float D_left, 
                            int ny_right, float s_right, float D_right, float a) {
    // Prepare right-hand side for left sub-domain
    for (int i = 1; i < ny_left - 1; i++) {
        B_left[i - 1] = n_left[i];
    }

    // Solve using PCR
    parallel_cyclic_reduction(lower_left, diag_left, upper_left, B_left, ny_left - 2);

    for (int i = 1; i < ny_left - 1; i++) {
        n_left[i] = B_left[i - 1];
    }

    // Update interface boundary condition
    n_right[0] = (n_right[1] - D_left / D_right * (n_left[ny_left - 1] - n_left[ny_left - 2])) / (1 - 2 * a);
    n_left[ny_left - 1] = n_right[1];

    // Prepare right-hand side for right sub-domain
    for (int i = 1; i < ny_right - 1; i++) {
        B_right[i - 1] = n_right[i];
    }

    // Solve using PCR
    parallel_cyclic_reduction(lower_right, diag_right, upper_right, B_right, ny_right - 2);

    for (int i = 1; i < ny_right - 1; i++) {
        n_right[i] = B_right[i - 1];
    }

    // Update interface boundary condition
    n_left[ny_left - 1] = n_left[ny_left - 2] + D_right / D_left * (n_right[1] - n_right[0] * (1 - 2 * a));
    n_right[1] = n_left[ny_left - 1];
}


// Main function
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " D_left a path_to_init_prof use_precomputed" << std::endl;
        return -1;
    }
    double y_max = 5.0;
    double dy = 1e-3;
    double dt = 1e-6;
    double D_left = std::atof(argv[1]);
    double a = std::atof(argv[2]);
    double D_right = 2e-2;
    double s_left = D_left * dt / (dy * dy);
    double s_right = D_right * dt / (dy * dy);
    double init_prof_shift = 0;
    double curr_time;
    double arr[] = {100.0};
    int n = sizeof(arr) / sizeof(arr[0]);
    std::vector<double> steps_to_dump(arr, arr + n);
    double sim_time = arr[n - 1] + 2 * dt;
    int nt = (int)(sim_time / dt);


    std::cout << "D_left = " << D_left << ", num of timesteps: " << nt << ", a = " << a << std::endl;

    int ny_left = (int)(y_max / dy) + 1;
    int ny_right = (int)(2 * y_max / dy) + 2;
    double* n_left = new double[ny_left];
    double* n_right = new double[ny_right];

    bool use_precomputed = (std::string(argv[4]) == "true");

    if (use_precomputed) {
        std::ifstream in(argv[3]);
        std::string line;
        double val;
        int ind1 = 0, ind2 = 0;
        std::getline(in, line);
        std::stringstream ss1(line);
        while (ss1 >> val) {
            n_left[ind1++] = val;
        }

        std::getline(in, line);
        std::stringstream ss2(line);
        while (ss2 >> val) {
            n_right[ind2++] = val;
        }
    } else {
        initialize_profiles(n_left, n_right, ny_left, ny_right, y_max, dy, init_prof_shift, 10.0);
    }

    // Initialize tridiagonal system coefficients

    double *d_lower_left;
    double *d_diag_left;
	double *d_upper_left;
    double *d_B_left;
    cudaMalloc(&d_lower_left, sizeof(double)*(ny_left-2));
    cudaMalloc(&d_upper_left, sizeof(double)*(ny_left-2));
    cudaMalloc(&d_diag_left, sizeof(double)*(ny_left-2));
    cudaMalloc(&d_B_left, sizeof(double)*(ny_left-2));
    cudaMemset(d_lower_left, -0.5 * s_left, (ny_left-2));
    cudaMemset(d_diag_left, 1 + s_left, (ny_left-2));
    cudaMemset(d_upper_left, -0.5 * s_left, (ny_left-2));

    double *d_lower_right;
    double *d_diag_right;
	double *d_upper_right;
    double *d_B_right;
    cudaMalloc(&d_lower_right, sizeof(double)*(ny_right-2));
    cudaMalloc(&d_upper_right, sizeof(double)*(ny_right-2));
    cudaMalloc(&d_diag_right, sizeof(double)*(ny_right-2));
    cudaMalloc(&d_B_right, sizeof(double)*(ny_right-2));
    cudaMemset(d_lower_right, -0.5 * s_right, (ny_right-2));
    cudaMemset(d_diag_right, 1 + s_right, (ny_right-2));
    cudaMemset(d_upper_right, -0.5 * s_right, (ny_right-2));
    
    // Move these variables as well
    double* d_n_left;
    cudaMalloc(&d_n_left, sizeof(double)*ny_left);
    cudaMemcpy(d_n_left, n_left, sizeof(double)*ny_left, cudaMemcpyHostToDevice);
    double* d_n_right;
    cudaMalloc(&d_n_right, sizeof(double)*ny_right);
    cudaMemcpy(d_n_right, n_right, sizeof(double)*ny_right, cudaMemcpyHostToDevice);

    for (int j = 1; j < nt; j++) {
        run_iteration<<<1, 1>>>(d_B_left, d_lower_left, d_upper_left, d_diag_left, d_n_left, 
                      d_B_right, d_lower_right, d_upper_right, d_diag_right, d_n_right, 
                      ny_left, s_left, D_left, 
                      ny_right, s_right, D_right, a);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        curr_time = j * dt;
        if (isin(steps_to_dump, curr_time)) {
            char filename[130];
            sprintf(filename, "profiles/impulses_volt/5_10_sim_time_%.6f_D_left_%.3f_a_%.4f.txt", curr_time, D_left, a);
            std::ofstream of(filename);

            cudaMemcpy(n_left,  d_n_left,  sizeof(double)*ny_left,  cudaMemcpyDeviceToHost);
            cudaMemcpy(n_right, d_n_right, sizeof(double)*ny_right, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            if (of.is_open()) {
                filewrite(of, n_left, ny_left, n_right, ny_right);
                of.close();
            }
            steps_to_dump.erase(std::remove(steps_to_dump.begin(), steps_to_dump.end(), curr_time), steps_to_dump.end());
        }
    }

    return 0;
}


