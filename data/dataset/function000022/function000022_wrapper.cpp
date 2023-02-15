#include "Halide.h"
#include "function000022_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>
using namespace std::chrono;
using namespace std;                
int main(int, char **argv)
{
	double *c_buf00 = (double*)malloc(65*65* sizeof(double));
	parallel_init_buffer(c_buf00, 65*65, (double)11);
	Halide::Buffer<double> buf00(c_buf00, 65,65);
	double *c_buf03 = (double*)malloc(66*66* sizeof(double));
	parallel_init_buffer(c_buf03, 66*66, (double)90);
	Halide::Buffer<double> buf03(c_buf03, 66,66);

	double *c_buf01 = (double*)malloc(65* sizeof(double));
	parallel_init_buffer(c_buf01, 65, (double)99);
	Halide::Buffer<double> buf01(c_buf01, 65);

	double *c_buf02 = (double*)malloc(65* sizeof(double));
	parallel_init_buffer(c_buf02, 65, (double)19);
	Halide::Buffer<double> buf02(c_buf02, 65);

	double *c_buf04 = (double*)malloc(65* sizeof(double));
	parallel_init_buffer(c_buf04, 65, (double)69);
	Halide::Buffer<double> buf04(c_buf04, 65);

    bool nb_runs_dynamic = is_nb_runs_dynamic();
    
    if (!nb_runs_dynamic){ 
        
        int nb_exec = get_max_nb_runs();    
        for (int i = 0; i < nb_exec; i++) 
        {  
            auto begin = std::chrono::high_resolution_clock::now(); 
            function000022(buf00.raw_buffer(),buf01.raw_buffer(),buf02.raw_buffer(),buf03.raw_buffer(),buf04.raw_buffer());
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
            function000022(buf00.raw_buffer(),buf01.raw_buffer(),buf02.raw_buffer(),buf03.raw_buffer(),buf04.raw_buffer());
            auto end = std::chrono::high_resolution_clock::now(); 

            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000;
            std::cout << duration << " "<< std::flush; 
            duration_vector.push_back(duration);
        }

        int nb_exec_remaining = choose_nb_runs(duration_vector);

        for (int i = 0; i < nb_exec_remaining; i++) 
        {  
            auto begin = std::chrono::high_resolution_clock::now(); 
            function000022(buf00.raw_buffer(),buf01.raw_buffer(),buf02.raw_buffer(),buf03.raw_buffer(),buf04.raw_buffer());
            auto end = std::chrono::high_resolution_clock::now(); 

            std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
        }
    }
    std::cout << std::endl;

	return 0; 
}

        