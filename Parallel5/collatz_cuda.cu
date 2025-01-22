/*
Compile in [Developer Command Prompt for VS 2022]: nvcc -o collatz_cuda collatz_cuda.cu --compiler-bindir="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64"

Run in [Developer Command Prompt for VS 2022]: collatz_cuda.exe 3 4000000
                                               collatz_cuda.exe 7 40000000

Once installed and PATH variable is set, restart PC and nvcc should be available within VScode terminal

Compile in terminal: nvcc -o collatz_cuda collatz_cuda.cu --compiler-bindir="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64"

Run in terminal: ./collatz_cuda 3 4000000
                 ./collatz_cuda 7 40000000
*/


#include <cstdio>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static __global__ void collatz( const long long start, const long long stop, int* const maxlen ){

  // compute sequence lengths
  const long long idx = threadIdx.x + ( blockIdx.x * (long long)blockDim.x );
  long long i = start + ( 4 * idx );
  if (i < stop) {
      long long val = i;
      int len = 1;
      do {
          len++;
          if ((val % 2) != 0) {
              val = val * 3 + 1;  // odd
          } else {
              val = val / 2;  // even
          }
      } while (val != 1);
      atomicMax( maxlen, len );
  }
}

static void CheckCuda(const int line){
    cudaError_t error;
    cudaDeviceSynchronize();
    if ( cudaSuccess != ( error = cudaGetLastError() ) ){
        std::cerr << "CUDA error " << error << " on line " << line << ": " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

int main(int argc, char* argv []){
  std::cout << "Collatz" << std::endl;

  // check command line
  if (argc != 3) {
    std::cerr << "USAGE: " << argv[0] << " start_value stop_value" << std::endl; 
    exit(-1);
  }
  const long long start = std::atoll(argv[1]);
  const long long stop = std::atoll(argv[2]);
  if (start >= stop) {
    std::cerr << "ERROR: start_value must be smaller than stop_value" << std::endl; 
    exit(-1);
  }
  std::cout << "start value: " << start << std::endl;
  std::cout << "stop value: " << stop << std::endl;

  // initialize maxlen for host
  int maxlen = 0;

  // allocate memory for maxlen on device
  int* d_maxlen;
  cudaMalloc( (void**)&d_maxlen, sizeof(int) );
  CheckCuda(__LINE__);

  // copy value to device
  cudaMemcpy( d_maxlen, &maxlen, sizeof(int), cudaMemcpyHostToDevice );
  CheckCuda(__LINE__);

  // start time
  auto beg = std::chrono::high_resolution_clock::now();

  // execute timed code
  collatz<<< ( (stop / 4) + ThreadsPerBlock - 1) / (ThreadsPerBlock), ThreadsPerBlock >>>( start, stop, d_maxlen );
  cudaDeviceSynchronize();

  // end time
  auto end = std::chrono::high_resolution_clock::now();
  CheckCuda(__LINE__);

  // calc
  std::chrono::duration<double> runtime = end - beg;
  std::cout << "compute time: " << runtime.count() << " s" << std::endl;

  // copy result from device to host
  cudaMemcpy( &maxlen, d_maxlen, sizeof(int), cudaMemcpyDeviceToHost );
  CheckCuda(__LINE__);

  // print result
  std::cout << "max sequence length: " << maxlen << std::endl;

  // clean up
  cudaFree( d_maxlen );

  return 0;
}
