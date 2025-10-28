// conv2d_optimized.cu
// Compile: nvcc -O3 -use_fast_math -arch=sm_70 -o conv2d conv2d_optimized.cu
// Run: ./conv2d [H W K]  (defaults: H=4096, W=4096, K=7)
//
// Demonstrates: memory coalescing, occupancy-aware launch config, and shared-memory tiling
// for 2D convolution on heterogeneous NVIDIA GPUs. Includes a naive baseline for comparison.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

#ifndef TILE_X
#define TILE_X 32
#endif
#ifndef TILE_Y
#define TILE_Y 16
#endif

static_assert(TILE_X > 0 && TILE_Y > 0, "Tile dimensions must be positive.");

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                                    \
  do {                                                                                      \
    cudaError_t _e = (call);                                                                \
    if (_e != cudaSuccess) {                                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));\
      exit(1);                                                                              \
    }                                                                                       \
  } while (0)
#endif

// --- Constant memory for small kernels (<= 15x15)
__constant__ float cKernel[15*15];

// Utility to round up
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Baseline (naive, global loads)
__global__ void conv2d_naive(const float* __restrict__ in, float* __restrict__ out,
                             int H, int W, const float* __restrict__ kernel, int K) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int R = K / 2;
  if (y >= H || x >= W) return;
  float acc = 0.f;
  for (int ky = -R; ky <= R; ++ky) {
    int iy = min(max(y + ky, 0), H - 1);
    for (int kx = -R; kx <= R; ++kx) {
      int ix = min(max(x + kx, 0), W - 1);
      float w = kernel[(ky + R) * K + (kx + R)];
      acc += w * in[iy * W + ix];
    }
  }
  out[y * W + x] = acc;
}

// Optimized: tiled shared memory + coalesced loads + optional constant-memory weights
// Template tile dims are OUTPUT tile size handled per block
// Halo is added for K radius.
template<int TX, int TY, int K>
__launch_bounds__(TX * TY, 2) // encourage higher occupancy: 2 blocks/SM when possible
__global__ void conv2d_tiled(const float* __restrict__ in, float* __restrict__ out,
                             int H, int W, const float* __restrict__ kernel_gmem, bool useConst) {
  constexpr int R = K / 2;
  // Shared memory tile: (TILE_Y + 2R) x (TILE_X + 2R)
  extern __shared__ float smem[];
  float* tile = smem;

  // Global coordinates for the output pixel this thread will compute
  int ox = blockIdx.x * TX + threadIdx.x;
  int oy = blockIdx.y * TY + threadIdx.y;

  // Each block loads a larger tile with halo. We'll cooperatively load in stripes.
  // Number of threads per block is TILE_X * TILE_Y; we map 2D thread indices.
  // Compute the top-left of the shared tile in global coords
  int tile_gx0 = blockIdx.x * TX - R;
  int tile_gy0 = blockIdx.y * TY - R;

  // The shared tile dimensions
  const int SH_W = TX + 2 * R;
  const int SH_H = TY + 2 * R;

  // Load shared tile with coalesced accesses by iterating rows with x-threads
  for (int sy = threadIdx.y; sy < SH_H; sy += TY) {
    int gy = tile_gy0 + sy;
    gy = min(max(gy, 0), H - 1);

    // vectorized load when possible (float4) for better coalescing
    int sx = threadIdx.x * 4; // 4 floats per iteration if aligned enough
    for (; sx + 3 < SH_W; sx += TX * 4) {
      int gx = tile_gx0 + sx;
      int gx0 = min(max(gx + 0, 0), W - 1);
      int gx1 = min(max(gx + 1, 0), W - 1);
      int gx2 = min(max(gx + 2, 0), W - 1);
      int gx3 = min(max(gx + 3, 0), W - 1);
      float4 v;
      // manual gather to avoid alignment pitfalls
      v.x = in[gy * W + gx0];
      v.y = in[gy * W + gx1];
      v.z = in[gy * W + gx2];
      v.w = in[gy * W + gx3];
      int base = sy * SH_W + sx;
      tile[base + 0] = v.x;
      tile[base + 1] = v.y;
      tile[base + 2] = v.z;
      tile[base + 3] = v.w;
    }
    // tail
    for (; sx < SH_W; ++sx) {
      int gx = tile_gx0 + sx;
      gx = min(max(gx, 0), W - 1);
      tile[sy * SH_W + sx] = in[gy * W + gx];
    }
  }

  __syncthreads();

  if (ox < W && oy < H) {
    float acc = 0.f;
#pragma unroll
    for (int ky = 0; ky < K; ++ky) {
#pragma unroll
      for (int kx = 0; kx < K; ++kx) {
        float w = useConst ? cKernel[ky * K + kx] : kernel_gmem[ky * K + kx];
        float v = tile[(threadIdx.y + ky) * SH_W + (threadIdx.x + kx)];
        acc += w * v;
      }
    }
    out[oy * W + ox] = acc;
  }
}

// Choose a near-peak occupancy configuration for the current device
struct LaunchCfg {
  dim3 grid, block; size_t smem = 0; bool useConst = false; int K = 0;
};

template<int TX, int TY>
LaunchCfg pick_cfg(int H, int W, int K, int dev) {
  cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  LaunchCfg cfg; cfg.K = K;
  int R = K / 2;
  // dynamic shared memory per block
  size_t shmem = (TX + 2*R) * (TY + 2*R) * sizeof(float);
  cfg.smem = shmem;
  cfg.block = dim3(TX, TY, 1);
  cfg.grid  = dim3(ceil_div(W, TX), ceil_div(H, TY), 1);
  // Prefer constant memory if kernel fits
  cfg.useConst = (K <= 15);
  // Guard shared mem limit
  if (shmem > prop.sharedMemPerBlockOptin) {
    // fall back to smaller tile if too big (user can recompile with different tile)
    fprintf(stderr, "[warn] Shared memory tile too big for this GPU; consider smaller TILE or K.\n");
  }
  return cfg;
}

// Host utility: upload kernel to constant memory if small enough
void uploadKernelConst(const std::vector<float>& h, int K) {
  CHECK_CUDA(cudaMemcpyToSymbol(cKernel, h.data(), K*K*sizeof(float), 0, cudaMemcpyHostToDevice));
}

// Initialize input and kernel
void init_input(std::vector<float>& a) {
  for (size_t i = 0; i < a.size(); ++i) a[i] = (float)((i * 1315423911u) & 0xFFFF) / 65535.f;
}

void make_gaussian(std::vector<float>& k, int K) {
  float sigma = K / 3.0f; float sum = 0.f; int R = K/2;
  for (int y= -R; y<=R; ++y) for (int x=-R; x<=R; ++x){
    float v = expf(-(x*x + y*y) / (2*sigma*sigma));
    k[(y+R)*K + (x+R)] = v; sum += v;
  }
  for (auto &v: k) v /= sum;
}

float run_and_time(void(*kernel)(const float*, float*, int, int, const float*, int),
                   const float* dIn, float* dOut, int H, int W, const float* dK, int K,
                   dim3 grid, dim3 block, size_t smem) {
  cudaEvent_t s, e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s));
  kernel<<<grid, block, smem>>>(dIn, dOut, H, W, dK, K);
  CHECK_CUDA(cudaEventRecord(e)); CHECK_CUDA(cudaEventSynchronize(e));
  float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
  CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
  return ms;
}

int main(int argc, char** argv){
  int H = (argc>1)? atoi(argv[1]): 4096;
  int W = (argc>2)? atoi(argv[2]): 4096;
  int K = (argc>3)? atoi(argv[3]): 7; // must be odd
  if (K % 2 == 0 || K < 1 || K > 15) {
    fprintf(stderr, "K must be odd and 1..15 for this demo.\n");
    return 1;
  }

  int dev=0; CHECK_CUDA(cudaGetDevice(&dev));
  cudaDeviceProp prop{}; CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  printf("Device %d: %s (SM %d.%d, %d MPs)\n", dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  size_t N = (size_t)H * W;
  std::vector<float> hIn(N), hOut(N), hOut2(N);
  std::vector<float> hK(K*K);
  init_input(hIn); make_gaussian(hK, K);

  float *dIn=nullptr, *dOut=nullptr, *dOut2=nullptr, *dK=nullptr;
  CHECK_CUDA(cudaMalloc(&dIn,  N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dOut, N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dOut2,N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dK,   K*K*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK,  hK.data(),  K*K*sizeof(float), cudaMemcpyHostToDevice));
  if (K <= 15) uploadKernelConst(hK, K);

  // --- Baseline launch (no shared memory)
  dim3 block0(16,16,1);
  dim3 grid0(ceil_div(W, block0.x), ceil_div(H, block0.y), 1);
  float t0 = run_and_time(conv2d_naive, dIn, dOut, H, W, dK, K, grid0, block0, 0);
  printf("Naive conv:  %.3f ms\n", t0);

  // --- Optimized launch (shared-memory tiling)
  // Choose tile size depending on device capability for heterogeneous fleet
  // You can build multiple variants; we'll pick TILE=32x16 as a good default
  using KernelT = void(*)(const float*, float*, int, int, const float*, bool);

  LaunchCfg cfg = pick_cfg<TILE_X, TILE_Y>(H, W, K, dev);
  float ms1=0;
  // Dispatch on K at compile time for best unrolling
  switch (K) {
    case 3: {
      ms1 = [&]{
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        conv2d_tiled<TILE_X, TILE_Y,3><<<cfg.grid, cfg.block, cfg.smem>>>(dIn, dOut2, H, W, dK, cfg.useConst);
        cudaEventRecord(e); cudaEventSynchronize(e); float ms=0; cudaEventElapsedTime(&ms,s,e);
        cudaEventDestroy(s); cudaEventDestroy(e); return ms; }(); break; }
    case 5: { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
      conv2d_tiled<TILE_X, TILE_Y,5><<<cfg.grid, cfg.block, cfg.smem>>>(dIn, dOut2, H, W, dK, cfg.useConst);
      cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms1,s,e); cudaEventDestroy(s); cudaEventDestroy(e); break; }
    case 7: { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
      conv2d_tiled<TILE_X, TILE_Y,7><<<cfg.grid, cfg.block, cfg.smem>>>(dIn, dOut2, H, W, dK, cfg.useConst);
      cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms1,s,e); cudaEventDestroy(s); cudaEventDestroy(e); break; }
    case 9: { cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
      conv2d_tiled<TILE_X, TILE_Y,9><<<cfg.grid, cfg.block, cfg.smem>>>(dIn, dOut2, H, W, dK, cfg.useConst);
      cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms1,s,e); cudaEventDestroy(s); cudaEventDestroy(e); break; }
    case 11:{ cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
      conv2d_tiled<TILE_X, TILE_Y,11><<<cfg.grid, cfg.block, cfg.smem>>>(dIn, dOut2, H, W, dK, cfg.useConst);
      cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms1,s,e); cudaEventDestroy(s); cudaEventDestroy(e); break; }
    case 13:{ cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
      conv2d_tiled<TILE_X, TILE_Y,13><<<cfg.grid, cfg.block, cfg.smem>>>(dIn, dOut2, H, W, dK, cfg.useConst);
      cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms1,s,e); cudaEventDestroy(s); cudaEventDestroy(e); break; }
    case 15:{ cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
      conv2d_tiled<TILE_X, TILE_Y,15><<<cfg.grid, cfg.block, cfg.smem>>>(dIn, dOut2, H, W, dK, cfg.useConst);
      cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms1,s,e); cudaEventDestroy(s); cudaEventDestroy(e); break; }
    default: fprintf(stderr, "Unsupported K for this demo.\n"); return 1;
  }

  printf("Tiled conv:  %.3f ms  (tile=%dx%d, K=%d, shared=%.1f KB, const=%s)\n",
         ms1, cfg.block.x, cfg.block.y, K, cfg.smem/1024.0, cfg.useConst?"yes":"no");

  // Validate correctness (L2 error)
  CHECK_CUDA(cudaMemcpy(hOut.data(),  dOut,  N*sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hOut2.data(), dOut2, N*sizeof(float), cudaMemcpyDeviceToHost));
  double err2=0, ref2=0; for (size_t i=0;i<N;++i){ double d=hOut[i]-hOut2[i]; err2+=d*d; ref2+=hOut[i]*hOut[i]; }
  printf("Relative L2 error: %.3e\n", sqrt(err2/(ref2+1e-30)));

  CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut)); CHECK_CUDA(cudaFree(dOut2)); CHECK_CUDA(cudaFree(dK));
  return 0;
}
