#include "Halide.h"
#include "function000026_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>
using namespace std::chrono;
using namespace std;                
int main(int, char **argv)
{
	double *c_buf00 = (double*)malloc(1280*768* sizeof(double));
	parallel_init_buffer(c_buf00, 1280*768, (double)10);
	Halide::Buffer<double> buf00(c_buf00, 1280,768);
	double *c_buf02 = (double*)malloc(512*768* sizeof(double));
	parallel_init_buffer(c_buf02, 512*768, (double)29);
	Halide::Buffer<double> buf02(c_buf02, 512,768);
	double *c_buf03 = (double*)malloc(128*512* sizeof(double));
	parallel_init_buffer(c_buf03, 128*512, (double)6);
	Halide::Buffer<double> buf03(c_buf03, 128,512);

	double *c_buf01 = (double*)malloc(768*513* sizeof(double));
	parallel_init_buffer(c_buf01, 768*513, (double)67);
	Halide::Buffer<double> buf01(c_buf01, 768,513);

    bool nb_runs_dynamic = is_nb_runs_dynamic();
    
    if (!nb_runs_dynamic){ 
        
        int nb_exec = get_max_nb_runs();    
        for (int i = 0; i < nb_exec; i++) 
        {  
            auto begin = std::chrono::high_resolution_clock::now(); 
            function000026(buf00.raw_buffer(),buf01.raw_buffer(),buf02.raw_buffer(),buf03.raw_buffer());
            auto end = std::chrono::high_resolution_clock::now(); 

            std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
        }
    }
    
    else{ // Adjust the number of runs depending on the measured time on the firs runs
    
        std::vector<double> duration_vector;
        double duration;
        int nb_exec = get_min_nb_runs();    
        
        for (int i = 0; i < nb_exec; i++) 
        {  
            auto begin = std::chrono::high_resolution_clock::now(); 
            function000026(buf00.raw_buffer(),buf01.raw_buffer(),buf02.raw_buffer(),buf03.raw_buffer());
            auto end = std::chrono::high_resolution_clock::now(); 

            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000;
            std::cout << duration << " "<< std::flush; 
            duration_vector.push_back(duration);
        }

        int nb_exec_remaining = choose_nb_runs(duration_vector);

        for (int i = 0; i < nb_exec_remaining; i++) 
        {  
            auto begin = std::chrono::high_resolution_clock::now(); 
            function000026(buf00.raw_buffer(),buf01.raw_buffer(),buf02.raw_buffer(),buf03.raw_buffer());
            auto end = std::chrono::high_resolution_clock::now(); 

            std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
        }
    }
    std::cout << std::endl;

	return 0; 
}

        