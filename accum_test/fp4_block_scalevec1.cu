#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s in %s at line %d\n",                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void test_sm120_fp4_mma_strict_packing() {
  // 1. C 矩阵初始化
  // (1.0 * 2^23)
  // uint32_t t = 0b0'10010110'00000000000000000000000;
  // (1.0 * 2^24)
  uint32_t t = 0b0'10010111'00000000000000000000000;
  float t_f = *((float *)&t);
  float c0 = t_f;
  float c1 = t_f;
  float c2 = t_f;
  float c3 = t_f;

  // 2. 输出变量
  float d0, d1, d2, d3;

  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  uint32_t b0 = 0, b1 = 0;

  // 4个0.5
  if (threadIdx.x == 0) {
    a0 = 0b00001000'00001000'00001000'00001000;
    a1 = 0b00000000'00000000'00000000'00000000;
    a2 = 0b00000000'00000000'00000000'00000000;
    a3 = 0b00000000'00000000'00000000'00000000;

    b0 = 0b00000100'00000100'00000100'00000100;
    b1 = 0b00000000'00000000'00000000'00000000;
  }

  // 8个0.25
  // if (threadIdx.x == 0 || threadIdx.x == 1) {
  //   a0 = 0b00000100'00000100'00000100'00000100;
  //   a1 = 0b00000000'00000000'00000000'00000000;
  //   a2 = 0b00000000'00000000'00000000'00000000;
  //   a3 = 0b00000000'00000000'00000000'00000000;

  //   b0 = 0b00000100'00000100'00000100'00000100;
  //   b1 = 0b00000000'00000000'00000000'00000000;
  // }

  // UE8M0 格式：8-bit 指数，Bias 127。 127(0x7F) -> 2^0 = 1.0
  // uint32_t sfa0 = 0x7F7F7F7F;
  // uint32_t sfb0 = 0x7F7F7F7F;
  uint32_t sfa0 = 0x7F;
  uint32_t sfb0 = 0x7F;

  // 5. Metadata (SM120 必须参数)
  // 用于索引缩放向量，这里设为0即可
  uint16_t tidA = 0;
  uint16_t bidA = 0;
  uint16_t tidB = 0;
  uint16_t bidB = 0;

  /*
  scale_vec::1X表示整个K维度的向量共享一个缩放因子
  2X就是K维度切成两份，每份有自己的sf
  */
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
               "m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
               "{%0,  %1,  %2,  %3},"  // D (4 floats)
               "{%4,  %5,  %6,  %7},"  // A (4 x b32) - 对应 a0, a1, a2, a3
               "{%8,  %9},"            // B (2 x b32) - 对应 b0, b1
               "{%10, %11, %12, %13}," // C (4 floats)
               "{%14},"                // Scale A
               "{%15, %16},"           // Metadata A (bid, tid)
               "{%17},"                // Scale B
               "{%18, %19};\n"         // Metadata B (bid, tid)
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
                 "f"(c1), "f"(c2), "f"(c3), "r"(sfa0), "h"(bidA), "h"(tidA),
                 "r"(sfb0), "h"(bidB), "h"(tidB));

  // 7. 打印输出
  if (threadIdx.x == 0) {
    // printf("\nThread 0 Result:\n");
    printf("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32."
           "row.col.f32.e2m1.e2m1.f32.ue8m0\n");
    printf("d0: %f\n", d0);
    printf("d1: %f\n", d1);
    // printf("d2: %f\n", d2);
    // printf("d3: %f\n", d3);
  }
}

int main() {
  printf("\nfp4 block scaled mma:\n");

  test_sm120_fp4_mma_strict_packing<<<1, 32>>>();

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  return 0;
}