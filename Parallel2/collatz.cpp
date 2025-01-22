/*
Compile: g++ -g collatz.cpp -o collatz

Run: ./collatz 3 4000000
*/

#include <cstdio>
#include <iostream>
#include <chrono>
#include <algorithm>


static int collatz(const long long start, const long long stop){
  int maxlen = 0;

  // compute sequence lengths
  for (long long i = start; i < stop; i += 1) {
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
    maxlen = std::max(maxlen, len);
  }

  return maxlen;
}


int main(int argc, char* argv []){
  std::cout << "Collatz Conjecture" << std::endl;

  // check command line
  if (argc != 3) {
    std::cerr << "USAGE: " << argv[0] << " start_value stop_value" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const long long start = atoll(argv[1]);
  const long long stop = atoll(argv[2]);
  if (start >= stop) {
    std::cerr << "ERROR: start_value must be smaller than stop_value" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  printf("start value: %lld\n", start);
  printf("stop value: %lld\n", stop);

  // start time
  auto begin = std::chrono::high_resolution_clock::now();

  // execute timed code
  const int maxlen = collatz(start, stop);

  // end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calc
  std::chrono::duration<double> elapsed = end - begin;
  std::cout << "Compute time: " << elapsed.count() << " seconds\n";

  // print result
  std::cout << "Max sequence length: " << maxlen << std::endl;
  return 0;
}
