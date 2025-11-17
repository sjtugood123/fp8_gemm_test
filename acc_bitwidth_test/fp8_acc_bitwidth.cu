#include <cstdio>
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

__global__ void test_fp8_mma_kernel() {

  uint32_t t = 0b01000000010000000111001111111111;
  float t_f = *((float *)&t);
  float c0 = t_f;
  float c1 = t_f;
  float c2 = t_f;
  float c3 = t_f;

  // 2. 准备输出寄存器 (D矩阵)
  float d0, d1, d2, d3;

  // 3. 准备输入 A (e4m3)
  uint32_t a0 = 0x00000000;
  uint32_t a1 = 0x00000000;
  uint32_t a2 = 0x00000000;
  uint32_t a3 = 0x00000000;

  // 4. 准备输入 B (e4m3)
  uint32_t b0 = 0x00000000;
  uint32_t b1 = 0x00000000;

  // 5. 执行内联汇编
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,  %1,  %2,  %3},"    // 输出 D (4个 float)
      "{%4,  %5,  %6,  %7},"    // 输入 A (4个 b32/packed fp8)
      "{%8,  %9},"              // 输入 B (2个 b32/packed fp8)
      "{%10, %11, %12, %13};\n" // 输入 C (4个 float Accumulator)
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
        "f"(c2), "f"(c3));

  if (threadIdx.x == 0) {
    // printf("\nThread 0 Result:\n");
    printf("d0: %f\n", d0);
    // printf("d1: %f\n", d1);
    // printf("d2: %f\n", d2);
    // printf("d3: %f\n", d3);
  }
}

int main() {
  uint32_t t = 0b01000000010000000111001111111111;
  float t_f = *((float *)&t);
  printf("ref:%f\n", t_f);
  test_fp8_mma_kernel<<<1, 32>>>();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    return -1;
  }

  return 0;
}