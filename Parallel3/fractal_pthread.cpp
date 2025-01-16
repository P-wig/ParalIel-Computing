/*
Very C style pthreading, can be more modernized with -std=c++17
compile: g++ -pthread -fpermissive -o fractal fractal_pthread.cpp


run: ./fractal 4096 8
     ./fractal 4096 16

This code is actually slower than the MPI version because of the critial section in fractal function. MPI has no 
syncronization cost as it avoids shared ememory access through message passing.
*/


#include <cstdio>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "BMP24.h"
#include <pthread.h>

using namespace std;

static long threads;
pthread_mutex_t mutex;
static int width;
static unsigned char* pic;

static void* fractal( void* arg ){
  const long my_rank = (long)arg;

  // Block partitioning
  const int begin_row = my_rank * width / threads;
  const int end_row = (my_rank + 1) * width / threads;

  const double scale = 0.003;
  const double xCenter = -0.663889302;
  const double yCenter =  0.353461972;

  // compute pixels of image
  const double xMin = xCenter - scale;
  const double yMin = yCenter - scale;
  const double dw = 2.0 * scale / width;
  double cy = yMin;

  for ( int row = begin_row; row < end_row; row++ ) {  // rows
    double cx = xMin;
    cy = yMin + ( dw * row );
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

      pthread_mutex_lock(&mutex);
      pic[ (row * width) + col ] = (unsigned char)count;
      pthread_mutex_unlock(&mutex);

      cx += dw;
    }
  }



  return NULL;
}


int main(int argc, char* argv []){
  cout << "Fractal" << endl;

  // check command line
  if (argc != 3) {
    cerr << "USAGE: " << argv[0] << " image_width threads" << endl;
    exit(-1);
  }
  width = atoi(argv[1]);
  if (width < 12) {
    cerr << "ERROR: image_width must be at least 12 pixels" << endl; 
    exit(-1);
  }
  // check thread count
  threads = atol(argv[2]);
  if (threads < 1) {
    cerr << "ERROR: threads must be at least 1" << endl;
    exit(-1);
  }
  
  cout << "Image width: " << width << endl;
  cout << "Threads: " << threads << endl;

  // allocate image memory
  pic = new unsigned char [width * width];

  // initialize pthread variables and mutex
  pthread_t* const handle = new pthread_t[threads - 1];
  pthread_mutex_init(&mutex, NULL);

  // start time
  auto start_time = chrono::high_resolution_clock::now();

  // launch threads
  for (long thread = 0; thread < threads - 1; thread++) {
      pthread_create(&handle[thread], NULL, fractal, (void*)thread);
  }

  // work for master
  fractal((void*)(threads - 1));

  // join threads
  for (long thread = 0; thread < threads - 1; thread++) {
      pthread_join(handle[thread], NULL);
  }

  // end time
  auto end_time = chrono::high_resolution_clock::now();

  // calc
  chrono::duration<double> runtime = end_time - start_time;
  
  cout << "Compute time: " << runtime.count() << " seconds" << endl;

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
  pthread_mutex_destroy( &mutex );
  delete[] handle;
  delete [] pic;
  return 0;
}
