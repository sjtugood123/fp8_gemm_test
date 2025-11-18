#include <cstdint>
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

__global__ void test_fp4_mma_kernel() {
  // (1.0 * 2^23)
  // uint32_t t = 0b0'10010110'00000000000000000000000;
  // (1.0 * 2^24)
  uint32_t t = 0b0'10010111'00000000000000000000000;
  float t_f = *((float *)&t);
  float c0 = t_f;
  float c1 = t_f;
  float c2 = t_f;
  float c3 = t_f;

  // 输出
  float d0, d1, d2, d3;

  // a是16*32的，每个线程是16个元素，即64 bits，应该是两个寄存器
  // 但mma要求是4个，也就是说可能按照f8的精度来传了
  // 打包细节:fp4放在8位的中间4位，比如00011100表示的是0111即6
  // e2m1的bias是1，但是0001是非规格化数，0.M*2^(1-bias)，即0.5
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;

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
  // fragment划分:https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832
  if (threadIdx.x == 0 || threadIdx.x == 1) {
    a0 = 0b00000100'00000100'00000100'00000100;
    a1 = 0b00000000'00000000'00000000'00000000;
    a2 = 0b00000000'00000000'00000000'00000000;
    a3 = 0b00000000'00000000'00000000'00000000;

    b0 = 0b00000100'00000100'00000100'00000100;
    b1 = 0b00000000'00000000'00000000'00000000;
  }

  // 5. 执行内联汇编
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
        "f"(c2), "f"(c3));

  if (threadIdx.x == 0) {
    // printf("\nThread 0 Result:\n");
    printf("d0: %f\n", d0);
    printf("d1: %f\n", d1);
    // printf("d2: %f\n", d2);
    // printf("d3: %f\n", d3);
  }
}

int main() {
  printf("\nfp4:\n");
  test_fp4_mma_kernel<<<1, 32>>>();

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