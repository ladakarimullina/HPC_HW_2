# g++ parallel_diffusion.cc -o run_diffusion_parallel -lm -fopenmp
nvcc diffusion.cu  -use_fast_math -rdc=true -o run_diffusion_parallel

dy=$1
# num_threads=$2

D_left=0.4
use_precomputed=true
init_prof_path=profiles/over_time/5_10_sim_time_40.00_D_left_0.400_a_-0.0017.txt
a=-0.0016525538567301913
./run_diffusion_parallel $D_left $a $init_prof_path $use_precomputed $dy
