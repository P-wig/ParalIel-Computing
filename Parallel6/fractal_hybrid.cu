/*
Not a executable code!!!
Only serves as a module for GPU computation called by fractal_hybrid.cpp
*/


#include <cstdio>
#include <iostream>
#include <cuda.h>


static const int ThreadsPerBlock = 512;


static __global__ void fractal(const int start, const int stop, const int width, unsigned char* const pic){
  // todo: compute the pixels in the range using double-precision arithmetic and store them at the 
  // beginning of the pic array
  const long long i = start + threadIdx.x + ( blockIdx.x * (long long)blockDim.x );

  const int col = i % width;  // columns
  const int row = i / width;  // rows

  if( i < stop ) {
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


unsigned char* GPU_Init(const int size){
  unsigned char* d_pic;
  cudaMalloc((void **)&d_pic, size);
  CheckCuda(__LINE__);
  return d_pic;
}


void GPU_Exec(const int start, const int stop, const int width, unsigned char* d_pic){
  // todo: launch the kernel with just the right number of blocks and ThreadsPerBlock threads per block 
  // and do nothing else
  fractal<<<( (width * width) + ThreadsPerBlock - 1) / (ThreadsPerBlock), ThreadsPerBlock>>>\
  ( start, stop, width, d_pic );
}


void GPU_Fini(const int size, unsigned char* pic, unsigned char* d_pic){
  // todo: copy the result from the device to the host and free the device memory
  cudaMemcpy( pic, d_pic, size * sizeof(unsigned char), cudaMemcpyDeviceToHost );
  //CheckCuda(__LINE__);
  cudaFree( d_pic );
}
