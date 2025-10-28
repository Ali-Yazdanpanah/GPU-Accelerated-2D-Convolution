// conv2d_serial.cpp
// Simple CPU reference convolution for benchmarking GPU speedup.
// Compile: g++ -O3 -march=native -o conv2d_serial conv2d_serial.cpp
// Run: ./conv2d_serial [H W K]  (defaults: H=1024, W=1024, K=7)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

static inline int clamp(int x, int a, int b) { return std::min(std::max(x, a), b); }

void conv2d_serial(const std::vector<float>& in, std::vector<float>& out,
                   const std::vector<float>& kernel, int H, int W, int K) {
  int R = K / 2;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float acc = 0.f;
      for (int ky = -R; ky <= R; ++ky) {
        int iy = clamp(y + ky, 0, H - 1);
        for (int kx = -R; kx <= R; ++kx) {
          int ix = clamp(x + kx, 0, W - 1);
          acc += in[iy * W + ix] * kernel[(ky + R) * K + (kx + R)];
        }
      }
      out[y * W + x] = acc;
    }
  }
}

void make_gaussian(std::vector<float>& k, int K) {
  float sigma = K / 3.0f; float sum = 0.f; int R = K/2;
  for (int y = -R; y <= R; ++y) for (int x = -R; x <= R; ++x) {
    float v = expf(-(x*x + y*y) / (2*sigma*sigma));
    k[(y+R)*K + (x+R)] = v; sum += v;
  }
  for (auto &v: k) v /= sum;
}

void init_input(std::vector<float>& a) {
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = (float)((i * 1315423911u) & 0xFFFF) / 65535.f;
}

int main(int argc, char** argv) {
  int H = (argc>1)? atoi(argv[1]): 1024;
  int W = (argc>2)? atoi(argv[2]): 1024;
  int K = (argc>3)? atoi(argv[3]): 7;
  if (K % 2 == 0) {
    std::cerr << "Kernel size must be odd!\n";
    return 1;
  }

  std::vector<float> in(H*W), out(H*W), kernel(K*K);
  init_input(in);
  make_gaussian(kernel, K);

  std::cout << "Running serial conv2d on CPU...\n";
  auto t0 = std::chrono::high_resolution_clock::now();
  conv2d_serial(in, out, kernel, H, W, K);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << "CPU conv2d completed in " << ms << " ms" << std::endl;
  return 0;
}
