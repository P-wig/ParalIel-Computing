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

Fifth. Compile! <still under configuration!!!>
   

Sixth. Run!
   
*/


#include <cstdio>
#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <chrono>
#include "BMP24.h"


unsigned char* GPU_Init(const int size);
void GPU_Exec(const int start, const int stop, const int width, unsigned char* const d_pic);
void GPU_Fini(const int size, unsigned char* const pic, unsigned char* const d_pic);


static void fractal(const int start, const int stop, const int width, unsigned char* const pic){
  // todo: use OpenMP to compute the pixels in the range in parallel and store them at 
  // the beginning of the pic array with default(none) and do not specify a schedule *
  #pragma omp parallel for default( none ) shared( start, stop, width, pic )
  for(int i = start; i < stop; i++){
    const int col = i % width;  // columns
    const int row = i / width;  // rows

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


int main(int argc, char* argv []){
  // set up MPI
  int comm_sz, my_rank;
  // todo: initialize MPI *
  MPI_Init( NULL, NULL );
  MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );
  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

  if (my_rank == 0){ 
    printf("Fractal hybrid parallelization\n");
  }

  // check command line
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <image_width> <cpu_percentage>\n";
    MPI_Finalize();
    exit(-1);
  }
  const int width = atoi(argv[1]);
  if (width < 12) {
    std::cerr << "Error: image_width must be at least 12 pixels\n";
    MPI_Finalize(); 
    exit(-1);
  }
  const int size = width * width;
  if ((size % comm_sz) != 0) {
    std::cerr << "Error: image size must be a multiple of the number of processes\n";
    MPI_Finalize();
    exit(-1);
  }
  const int percentage = atoi(argv[2]);
  if ((percentage < 0) || (percentage > 100)) {
    std::cerr << "Error: cpu_percentage must be between 0 and 100\n";
    MPI_Finalize();
    exit(-1);
  }

  // distribute work
  const int cpu_start = my_rank * (long long)size / comm_sz; 
  const int gpu_stop = (my_rank + 1) * (long long)size / comm_sz;
  const int my_range = gpu_stop - cpu_start;
  const int cpu_stop = cpu_start + my_range * percentage / 100;
  const int gpu_start = cpu_stop;

  if (my_rank == 0) {
    std::cout << "Image width: " << width << "\nCPU percentage: " << percentage << "\nMPI tasks: " << comm_sz << "\n";
  }

  // allocate image memory
  unsigned char* const pic = new unsigned char [size];
  unsigned char* const d_pic = GPU_Init(gpu_stop - gpu_start);
  unsigned char* full_pic = NULL;
  if (my_rank == 0) full_pic = new unsigned char [size];

  // start time
  MPI_Barrier(MPI_COMM_WORLD);  // for better timing
  auto beg = std::chrono::high_resolution_clock::now();

  // asynchronously compute the requested pixels on the GPU
  if (gpu_start < gpu_stop) GPU_Exec(gpu_start, gpu_stop, width, d_pic);

  // compute the remaining pixels on the CPU
  if (cpu_start < cpu_stop) fractal(cpu_start, cpu_stop, width, pic);

  // copy the GPU's result into the appropriate location of the CPU's pic array
  if (gpu_start < gpu_stop) GPU_Fini(gpu_stop - gpu_start, &pic[cpu_stop - cpu_start], d_pic);

  // todo: gather the results into full_pic on compute node 0 *
  // MPI is only used to launch the code on multiple compute nodes and to communicate data between them
  MPI_Gather( &pic[my_range], 
            my_range, 
            MPI_UNSIGNED_CHAR, 
            full_pic, 
            my_range, 
            MPI_UNSIGNED_CHAR, 
            0, 
            MPI_COMM_WORLD);


  if (my_rank == 0) {
    auto end = std::chrono::high_resolution_clock::now();

    // calc
    std::chrono::duration<double> runtime = end - beg;
    std::cout << "Compute time: " << runtime.count() << " s\n";

    // write image to BMP file
    if (width <= 1024) {
      BMP24 bmp(0, 0, width, width);
      for (int y = 0; y < width; y++) {
        for (int x = 0; x < width; x++) {
          bmp.dot(x, y, 0x0000ff - full_pic[y * width + x] * 0x000001 + full_pic[y * width + x] * 0x010100);
        }
      }
      bmp.save("fractal.bmp");
    }

    delete [] full_pic;
  }

  MPI_Finalize();
  delete [] pic;
  return 0;
}
