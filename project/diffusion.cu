#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "utils.cuh"

bool isin(std::vector<float> v, float x)
{
    return std::find(v.begin(), v.end(), x) != v.end();
}

std::ostream& filewrite(std::ostream& os, double* arr1, int len1, double* arr2, int len2)
{
    char buf[5];
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

// --- GPU code ---

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

__global__ void mulByTridiagKernel(double* res, double* arr, int len_arr, float lower_d, float d, float upper_d)
{
    int gid = getGid();
    if (gid < len_arr) {
        if (gid == 0) {
            res[0] = d * arr[0] + upper_d * arr[1];
        } else if (gid == len_arr-1) {
            res[len_arr-1] = lower_d * arr[len_arr-2] + d * arr[len_arr-1];
        } else {
            res[gid] = lower_d * arr[gid-1] + d * arr[gid] + upper_d * arr[gid+1];
	    }
    }
}


__device__ void interDiffusion(double* B, double* alpha, double* beta, double* n, int ny, float s, float a)
{
    // Need to check, that the number of blocks is not smaller than 1
    int num_blocks = 0.5*s*(1-a);
    num_blocks = num_blocks/1024 + (num_blocks % 1024 > 0);
	mulByTridiagKernel<<<num_blocks, 1024>>>(B, &n[1], ny - 2, 0.5 * s * (1 - a), 1 - s, 0.5 * s * (1 + a));
	
    B[0] = B[0] + 0.5 * s * (n[0] + n[0]) * (1 - a); // changed
	B[ny-3] = B[ny-3] + 0.5 * s * (n[ny-1] + n[ny-1]) * (1 + a); // changed

	alpha[0] = (s * (1 + a) / 2) / (1 + s);
	beta[0] = B[0] / (1 + s);

	for (int i = 1; i < ny - 3; i++){
		alpha[i] = (s * (1 + a) / 2) / (1 + s - s * (1 - a) / 2 * alpha[i - 1]);
		beta[i] = (s * (1 - a) / 2 * beta[i - 1] + B[i]) / (1 + s - s * (1 - a) / 2 * alpha[i - 1]);
	}

	n[ny - 2] = (B[ny - 3] + s * (1 - a) / 2 * beta[ny - 4]) / (1 + s - s * (1 - a) / 2 * alpha[ny - 4]);
        
	for (int j = ny - 3; j > 0; j--){
            n[j] = alpha[j - 1] * n[j + 1] + beta[j - 1];
	}
}

__global__ void diffusionIterationKernel(double* B_left, double* alpha_left, double* beta_left, double* n_left, 
                                         double* B_right, double* alpha_right, double* beta_right, double* n_right, 
                                         int ny_left, float s_left, float D_left, 
                                         int ny_right, float s_right, float D_right, float a) {
    interDiffusion(B_left, alpha_left, beta_left, n_left, ny_left, s_left, 0.);
	n_right[0] = (n_right[1] - D_left / D_right * (n_left[ny_left-1] - n_left[ny_left-2])) / (1 - 2 * a);
	n_left[ny_left-1] = n_right[1];

    interDiffusion(B_right, alpha_right, beta_right, n_right, ny_right, s_right, a);
    n_left[ny_left-1] = n_left[ny_left-2] + D_right / D_left * (n_right[1] - n_right[0] * (1 - 2 * a));
	n_right[1] = n_left[ny_left-1];
	// n_right[ny_right-1] = (1 - 2 * a) * n_right[ny_right-2]; // zero outflux
}


int main(int argc, char *argv[])
{
	if (argc < 5){
		printf("%s D_left a path_to_init_prof use_precomputed\n", argv[0]);
		return -1;
	}
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

	float y_max = 5.;
	float dy = std::atof(argv[5]);;
	float dt = 1e-3;
	float D_left = std::atof(argv[1]);
	float a = std::atof(argv[2]);
	float D_right = 2e-2;
	float s_left = D_left * dt / (dy * dy);  	
	float s_right = D_right * dt / (dy * dy);
	float init_prof_shift = 0;
	char filename[130]; // make sure it's big enough
	float curr_time;
	float arr[] = {1.};
	// float arr[] = {60};
    	int n = sizeof(arr) / sizeof(arr[0]);
	std::vector<float> steps_to_dump(arr, arr + n);
	float sim_time = arr[n-1] + 2 * dt;
	int nt = (int) (sim_time / dt);

	for (int i=0; i < steps_to_dump.size(); i++){
		std::cout << steps_to_dump[i] << ' ';
	}
	std::cout << '\n';

	printf("D_left = %.3f, num of timesteps: %d, a = %.3f\n", D_left, nt, a);
	
	int ny_left = (int) (y_max / dy) + 1;
	int ny_right = (int) (2 * y_max / dy) + 2; // one for zero and one for type change
	double n_left[ny_left];
	double n_right[ny_right];

	bool use_precomputed;
	std::string action(argv[4]);
	if (action == "true") {
		use_precomputed = true;
	}
	else {
		use_precomputed = false;
	}


	if (use_precomputed){
		printf("Using precomputed profile\n");

		std::ifstream in(argv[3]);
		std::string line;
		float val;
		int ind1 = 0;
		int ind2 = 0;
		std::getline(in, line);
		std::stringstream ss1(line);
		while (ss1 >> val) {
			n_left[ind1] = val;
			ind1++;
		}

		std::getline(in, line);
		std::stringstream ss2(line);
		while (ss2 >> val) {
			n_right[ind2] = val;
			ind2++;
		}
	}
	else{
		printf("Using new initial profile\n");

		float p = -y_max;
		for (int j = 0; j < ny_left; j++){
			n_left[j] = (1 - tanhf((p - init_prof_shift) * 10)) / 2;
			// n_left[j] = (y_max - p) / y_max;
			p += dy;
		}

		p = - dy;
		for (int j = 0; j < ny_right; j++){
			n_right[j] = (1 - tanhf((p - init_prof_shift) * 10)) / 2;
			// n_right[j] = 0;
			p += dy;
		}

	}



    double *d_alpha_left;
	double *d_beta_left;
    double *d_B_left;
    cudaMalloc(&d_alpha_left, sizeof(double)*(ny_left-3));
    cudaMalloc(&d_B_left, sizeof(double)*(ny_left-2));
    cudaMalloc(&d_beta_left, sizeof(double)*(ny_left-3));
	//alpha_left = new double[ny_left - 3];
	//B_left = new double[ny_left - 2];
	//beta_left = new double[ny_left - 3];

    double *d_alpha_right;
	double *d_beta_right;
    double *d_B_right;
    cudaMalloc(&d_alpha_right, sizeof(double)*(ny_right-3));
    cudaMalloc(&d_B_right, sizeof(double)*(ny_right-2));
    cudaMalloc(&d_beta_right, sizeof(double)*(ny_right-3));
	//alpha_right = new double[ny_right - 3];
	//B_right = new double[ny_right - 2];
	//beta_right = new double[ny_right - 3];
    
    // Move these variables as well
    double* d_n_left;
    cudaMalloc(&d_n_left, sizeof(double)*ny_left);
    cudaMemcpy(d_n_left, n_left, sizeof(double)*ny_left, cudaMemcpyHostToDevice);
    double* d_n_right;
    cudaMalloc(&d_n_right, sizeof(double)*ny_right);
    cudaMemcpy(d_n_right, n_right, sizeof(double)*ny_right, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    
	for (int j = 1; j < nt; j++){
		diffusionIterationKernel<<<1, 1>>>(d_B_left, d_alpha_left, d_beta_left, d_n_left, 
                                           d_B_right, d_alpha_right, d_beta_right, d_n_right, 
                                           ny_left, s_left, D_left, ny_right, s_right, D_right, a);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
		curr_time = j * dt;
//		if (isin(steps_to_dump, curr_time)){
//			sprintf(filename, "profiles/impulses_volt/5_10_sim_time_%.6f_D_left_%.3f_a_%.4f.txt", curr_time, D_left, a);
//			std::fstream of(filename, std::ios::out);
//            
//            cudaMemcpy(n_left, d_n_left, sizeof(double)*ny_left, cudaMemcpyDeviceToHost);
//			cudaMemcpy(n_right, d_n_right, sizeof(double)*ny_right, cudaMemcpyDeviceToHost);
//            cudaDeviceSynchronize();
//            if (of.is_open()){
//					filewrite(of, n_left, ny_left, n_right, ny_right);
//					of.close();
//			}
//			printf("Dumped in a file a profile for %.2f seconds\n", curr_time);
//			steps_to_dump.erase(std::remove(steps_to_dump.begin(), steps_to_dump.end(), curr_time), steps_to_dump.end());
//		}
	}
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << milliseconds << std::endl;

	cudaFree(d_alpha_left);
	cudaFree(d_B_left);
	cudaFree(d_beta_left);
    cudaFree(d_alpha_right);
	cudaFree(d_beta_right);
    cudaFree(d_B_right);
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);

	return 0;
}	
