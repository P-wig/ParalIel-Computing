/*
Compile: g++ -fopenmp -o fractal1_omp fractal1_omp.cpp

Run: ./fractal1_omp 1024 4
     ./fractal1_omp 1024 8
*/


#include <cstdio>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "BMP24.h"


static void fractal( const int width, unsigned char* const pic, const int threads ){
  const double scale = 0.003;
  const double xCenter = -0.663889302;
  const double yCenter =  0.353461972;

  // compute pixels of image
  const double xMin = xCenter - scale;
  const double yMin = yCenter - scale;
  const double dw = 2.0 * scale / width;
  //double cy = yMin;

# pragma omp parallel for default( none ) num_threads( threads ) shared( width, pic, dw )
  for (int row = 0; row < width; row++) {  // rows
    double cx = xMin;
    double cy = yMin + ( dw * row );
    for (int col = 0; col < width; col++) {  // columns
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
      pic[row * width + col] = (unsigned char)count;
      cx += dw;
    }
  }
}


int main(int argc, char* argv []){
  std::cout << "Fractal" << std::endl;

  // check command line
  if (argc != 3) {
    std::cerr << "USAGE: " << argv[0] << " image_width thread_count" << std::endl;
    exit(-1);
  }
  const int width = atoi(argv[1]);
  if (width < 12) {
    std::cerr << "ERROR: image_width must be at least 12 pixels" << std::endl;
    exit(-1);
  }
  std::cout << "Image width: " << width << std::endl;

  const int threads = atoi(argv[2]);
  if (threads < 1) {
    std::cerr << "ERROR: thread_count must be at least 1" << std::endl;
    exit(-1);
  }
  std::cout << "Thread count: " << threads << std::endl;

  // allocate image memory
  unsigned char* pic = new unsigned char [width * width];

  // start time
  auto start_time = std::chrono::high_resolution_clock::now();

  // execute timed code
  fractal(width, pic, threads);

  // end time
  auto end_time = std::chrono::high_resolution_clock::now();

  // calc
  std::chrono::duration<double> runtime = end_time - start_time;
  std::cout << "Compute time: " << runtime.count() << " seconds" << std::endl;

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
  delete [] pic;
  return 0;
}
