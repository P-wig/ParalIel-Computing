

#ifndef BMP_24
#define BMP_24

#include <cstdio>

class BMP24 {
  private:
    int wo, ho;
    int w, h;
    int* bmp;

  public:
    BMP24(int xmin = 0, int ymin = 0, int xmax = 512, int ymax = 512){
      if ((xmin >= xmax) || (ymin >= ymax)) exit(-2);
      wo = xmin;
      ho = ymin;
      w = xmax - xmin;
      h = ymax - ymin;
      bmp = new int[w * h];
    }

    ~BMP24(){
      delete [] bmp;
    }

    void clear(int col){
      for (int i = 0; i < w * h; i++) bmp[i] = col;
    }

    void dot(int x, int y, const int col){
      x -= wo;
      y -= ho;
      if ((0 <= x) && (0 <= y) && (x < w) && (y < h)) {
        bmp[y * w + x] = col;
      }
    }

    void save(const char* const name){
      const int pad = ((w * 3 + 3) & ~3) - (w * 3);
      FILE* f = fopen(name, "wb");
      int d;

      d = 0x4d42;  fwrite(&d, 1, 2, f);
      d = 14 + 40 + h * w * 3 + pad * h;  fwrite(&d, 1, 4, f);
      d = 0;  fwrite(&d, 1, 4, f);
      d = 14 + 40;  fwrite(&d, 1, 4, f);

      d = 40;  fwrite(&d, 1, 4, f);
      d = w;  fwrite(&d, 1, 4, f);
      d = h;  fwrite(&d, 1, 4, f);
      d = 1;  fwrite(&d, 1, 2, f);
      d = 24;  fwrite(&d, 1, 2, f);
      d = 0;  fwrite(&d, 1, 4, f);
      d = h * w * 3 + pad * h;  fwrite(&d, 1, 4, f);
      d = 0;  fwrite(&d, 1, 4, f);
      d = 0;  fwrite(&d, 1, 4, f);
      d = 0;  fwrite(&d, 1, 4, f);
      d = 0;  fwrite(&d, 1, 4, f);

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          fwrite(&bmp[y * w + x], 1, 3, f);
        }
        fwrite(&d, 1, pad, f);
      }

      fclose(f);
    }
};

#endif
