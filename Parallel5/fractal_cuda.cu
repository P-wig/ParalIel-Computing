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
   - Compile: nvcc -o fractal_cuda fractal_cuda.cu
   - Compile explicitly: nvcc -o fractal_cuda fractal_cuda.cu --compiler-bindir="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64"

Sixth. Run!
   - If still using [Developer Command Prompt for VS 2022]
     - fractal_cuda.exe 4096
   - If using terminal
     - ./fractal_cuda 4096
*/


#include <cstdio>
#include <iostream>
#include <chrono>
#include <algorithm>
#include "BMP24.h"
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static __global__ void fractal( const int width, unsigned char* const pic ){

  const long long i = threadIdx.x + ( blockIdx.x * (long long)blockDim.x );

  const int col = i % width;  // columns
  const int row = i / width;  // rows

  if( i < (width * width) ) {
      const double scale = 0.003;
      const double xCenter = -0.663889302;
      const double yCenter =  0.353461972;

      // compute pixels of image
      const double xMin = xCenter - scale;
      const double yMin = yCenter - scale;
      const double dw = 2.0 * scale / width;

      double cy = yMin + (row * dw);
      double cx = xMin + (col * dw);

      double x = cx;
      double y = cy;
      double x2, y2;
      int count = 256;
      do {
          x2 = x * x;
          y2 = y * y;
          y = 2.0 * x * y + cy;
          x = x2 - y2 + cx;
          count--;
      } while ((count > 0) && ((x2 + y2) < 5.0));

      pic[row * width + col] = (unsigned char) count;
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
  std::cout << "Fractal with double precision";

  // check command line
  if (argc != 2) {
    std::cerr << "USAGE: " << argv[0] << " image_width" << std::endl; 
    exit(-1);
  }
  const int width = atoi(argv[1]);
  if (width < 12) {
    std::cerr << "ERROR: image_width must be at least 12 pixels" << std::endl; 
    exit(-1);
  }
  std::cout << "Image width: " << width << std::endl;

  // allocate memory for host
  // makue_unique creates a smart pointer with automatic memory management
  auto pic = std::make_unique<unsigned char[]>(width * width);

  // allocate memory for pic on device
  unsigned char* d_pic;
  cudaMalloc( (void**)&d_pic, (width * width) * sizeof(unsigned char) );
  CheckCuda(__LINE__);

  // start time
  auto start = std::chrono::high_resolution_clock::now();

  // execute timed code
  fractal<<<( (width * width) + ThreadsPerBlock - 1) / (ThreadsPerBlock), ThreadsPerBlock>>>( width, d_pic );
  cudaDeviceSynchronize();

  // end time
  auto end = std::chrono::high_resolution_clock::now();
  CheckCuda(__LINE__);

  // calc
  std::chrono::duration<double> runtime = end - start;
  std::cout << "Compute time: " << runtime.count() << " s" << std::endl;

  // copy result from device to host
  cudaMemcpy( pic.get(), d_pic, (width * width) * sizeof(unsigned char), cudaMemcpyDeviceToHost );
  CheckCuda(__LINE__);

  // write image to BMP file
  if (width <= 1024) {
    BMP24 bmp(0, 0, width, width);
    for (int y = 0; y < width; y++) {
      for (int x = 0; x < width; x++) {
        bmp.dot(x, y, 0x0000ff - pic[y * width + x] * 0x000001 + pic[y * width + x] * 0x010100);
      }
    }
    bmp.save("fractal.bmp");
  }

  // clean up
  cudaFree( d_pic );
  return 0;
}
