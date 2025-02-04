/*
First. Install NVIDIA CUDA Toolkit:
   - Download the NVIDIA CUDA Toolkit from the official website: https://developer.nvidia.com/cuda-downloads
   - Follow the installation instructions for your operating system and ensure to install the necessary drivers.

Second. Install Visual Studio Build Tools (VS Build Tools):
   - Download and install VS Build Tools from the official Microsoft website: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - During installation, ensure to select "Desktop development with C++" and "MSVC v14.x" components.

Third. Set Up PATH Variables (this will allow nvcc to be referenced in VScode terminal):
   - Add the following directory to your system PATH (adjust paths for your installation):
     - Microsoft Visual Studio Build Tools:
       - `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<version>\bin\Hostx64\x64`

Forth. Restart Your System:
   - After modifying the PATH variables, restart your computer to ensure the changes take effect.

Fifth. Compile!
   - Compile: nvcc -o collatz_multiGPU collatz_multiGPU.cu
   - Compile explicitly: nvcc -o collatz_multiGPU collatz_multiGPU.cu --compiler-bindir="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64"

Sixth. Run! (argv[3] can be modified to match the number of GPUs your machine has)
   - If still using [Developer Command Prompt for VS 2022]
     - collatz_multiGPU.exe 3 4000000 1
     - collatz_multiGPU.exe 7 40000000 1
   - If using terminal
     - ./collatz_multiGPU 3 4000000 1
     - ./collatz_multiGPU 7 40000000 1
*/


#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <chrono>


static const int ThreadsPerBlock = 512;


static __global__ void collatz(const long long start, const long long stop, const long long increment, int* const maxlen){
  // todo: insert solution from previous project and change it to support an arbitrary loop-counter 
  // increment (not just "4") *
  // compute sequence lengths
  const long long idx = threadIdx.x + ( blockIdx.x * (long long)blockDim.x );
  long long i = start + ( increment * idx );
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
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}


int main(int argc, char* argv []){
  std::cout << "Collatz multi GPU" << std::endl;

  // check command line
  if (argc != 4) {
    std::cerr << "USAGE: " << argv[0] << " start_value stop_value number_of_GPUs" << std::endl;
    exit(-1);
  }
  const long long start = atoll(argv[1]);
  const long long stop = atoll(argv[2]);
  if (start >= stop) {
    std::cerr << "ERROR: start_value must be smaller than stop_value" << std::endl;
    exit(-1);
  }
  const int gpus = atoi(argv[3]);
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpus > device_count) {
    std::cerr << "ERROR: Requested " << gpus << " GPUs, but only " << device_count << " available." << std::endl;
    exit(-1);
  }
  if ((gpus < 1) || (gpus > 4)) {
    std::cerr << "ERROR: number of GPUs must be between 1 and 4" << std::endl; 
    exit(-1);
  }
  
  std::cout << "Start value: " << start << "\nStop value: " << stop << "\nNumber of GPUs: " << gpus << std::endl;

  // allocate and initialize GPU memory
  int maxlen = 0;
  int* d_maxlen [4];
  for (int d = 0; d < gpus; d++) {
    cudaSetDevice(d);
    cudaMalloc((void **)&d_maxlen[d], sizeof(int));
    CheckCuda(__LINE__);
    // todo: set the maximum length on the selected GPU to zero by copying maxlen to the GPU *
    cudaMemcpy( d_maxlen[d], &maxlen, sizeof(int), cudaMemcpyHostToDevice );
    CheckCuda(__LINE__);
  }

  // start time
  auto beg = std::chrono::high_resolution_clock::now();

  // execute timed code
  for (int d = 0; d < gpus; d++) {
    cudaSetDevice(d);
    const long long begin = start + (4 * d); // todo: compute the start value for each GPU * 
    const long long incr = 4 * gpus; // todo: compute the loop-counter increment * 
    collatz<<<( (stop / incr) + ThreadsPerBlock - 1) / (ThreadsPerBlock)/* todo: compute number of TB*/, ThreadsPerBlock>>>(begin, stop, incr, d_maxlen[d]);
  }
  for (int d = 0; d < gpus; d++) {
    cudaSetDevice(d);
    cudaDeviceSynchronize();  // wait for kernel to finish
  }

  // end time
  auto end = std::chrono::high_resolution_clock::now();
  CheckCuda(__LINE__);
  
  // calc
  std::chrono::duration<double> runtime = end - beg;
  std::cout << "Compute time: " << runtime.count() << " s" << std::endl;

  // todo: get the result from each GPU and reduce the values on the CPU into a final global result *
  for( int d = 0; d < gpus; d++ ){
    cudaSetDevice(d);
    int templen = 0;
    cudaMemcpy( &templen, d_maxlen[d], sizeof(int), cudaMemcpyDeviceToHost );
    CheckCuda(__LINE__);
    maxlen = std::max( maxlen, templen );
  }
  std::cout << "Max sequence length: " << maxlen << std::endl;

  for (int d = 0; d < gpus; d++) {
    cudaSetDevice(d);
    cudaFree(d_maxlen[d]);
    CheckCuda(__LINE__);
  }
  return 0;
}
