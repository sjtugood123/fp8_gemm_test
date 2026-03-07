#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_fp8.h> // 引入FP8支持
#include <cute/arch/mma_sm120.hpp>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s in %s at line %d\n",                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// __global__ void test_fp16_mma_kernel_k32(float c) {
//   float c0 = c;
//   float c1 = c;
//   float c2 = c;
//   float c3 = c;

//   float d0, d1, d2, d3;

//   // k64: A=8 regs, B=4 regs (每个寄存器打包2个half)
//   uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
//   uint32_t a4 = 0, a5 = 0, a6 = 0, a7 = 0;

//   uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;

//   // 这里不做测试逻辑，你可以自行填充a/b寄存器

//   asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
//                "{%0,  %1,  %2,  %3},"
//                "{%4,  %5,  %6,  %7,  %8,  %9,  %10, %11},"
//                "{%12, %13, %14, %15},"
//                "{%16, %17, %18, %19};\n"
//                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
//                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5),
//                "r"(a6),
//                  "r"(a7), "r"(b0), "r"(b1), "r"(b2), "r"(b3), "f"(c0),
//                  "f"(c1), "f"(c2), "f"(c3));
//   printf("k32right\n");
// }

__global__ void test_fp16_mma_kernel_k16(float c) {
  float c0 = c;
  float c1 = c;
  float c2 = c;
  float c3 = c;

  float d0, d1, d2, d3;

  // k32: A=4 regs, B=2 regs (每个寄存器打包2个half)
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;

  uint32_t b0 = 0, b1 = 0;

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,  %1,  %2,  %3},"
               "{%4,  %5,  %6,  %7},"
               "{%8,  %9},"
               "{%10, %11, %12, %13};\n"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
                 "f"(c1), "f"(c2), "f"(c3));
  if (threadIdx.x == 0) {
    printf("k16right\n");
  }
}

// __global__ void test_fp16_mma_kernel_k16(float c) {
//   float c0 = c;
//   float c1 = c;
//   float c2 = c;
//   float c3 = c;

//   float d0, d1, d2, d3;

//   // k16: A=2 regs, B=1 reg (每个寄存器打包2个half)
//   uint32_t a0 = 0, a1 = 0;

//   uint32_t b0 = 0;

//   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
//                "{%0,  %1,  %2,  %3},"
//                "{%4,  %5},"
//                "{%6},"
//                "{%7,  %8,  %9,  %10};\n"
//                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
//                : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2),
//                "f"(c3));
//   printf("k16right\n");
// }

int main() {

  uint32_t c_init = 0b0'10010111'00000000000000000000000; // 2^24
  float c = *((float *)&c_init);

  //   test_fp16_mma_kernel_k32<<<1, 32>>>(c);
  test_fp16_mma_kernel_k16<<<1, 32>>>(c);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}