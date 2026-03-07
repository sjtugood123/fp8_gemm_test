/*
假设W=4或者8
目标其实就是用k=16测出两个W的结果应该不同
不同的原因：
1.在前面的节点进行了舍入，这好像和lossless是矛盾的，前面已经证明了fp16相乘加到fp32上不会造成loss
2.但是不太一样，lossless是乘的过程会保留22位所以对于乘是无损的，如果C很大，比如2^128，那么加的过程一定会有损失，所以和lossless并不矛盾

那方案比较明确，C设成一个大数，如果W是8，那么4个1/4ULP不管在八个mul
node哪个位置都应该能加上，而如果是4则只有全在前四个才能加上。
*/

#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s in %s at line %d\n",                     \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void test_w() {

  unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  unsigned int b0 = 0, b1 = 0;
  b0 = b1 = 0b0'01111'0000000000'0'01111'0000000000; // b全设成1.0
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  // 顺序安排见：https://docs.nvidia.com/cuda/parallel-thread-execution/_images/mma-16816-A-f16.png

  // 这一条能加上，W>8，于是W应该是16，和之前做的accumTree结果相符

  if (threadIdx.x == 0 || threadIdx.x == 2) {
    a0 = a2 = 0b0'01101'0000000000'0'01110'0000000000; // 0.25
    // b0 = b1 = 0b0'01111'0000000000'0'01111'0000000000; // 1.0
    unsigned int c_init = 0b0'10010111'00000000000000000000000;
    float c = *((float *)&c_init);
    c0 = c;
  }

  float d0, d1, d2, d3;

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%10, %11, %12, %13};\n"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
                 "f"(c1), "f"(c2), "f"(c3));

  if (threadIdx.x == 0) {
    unsigned int u;
    // 将float的字节复制到unsigned int中
    memcpy(&u, &d0, sizeof(float));

    printf("Float: %f\n", d0);
    printf("Binary: ");
    for (int i = 31; i >= 0; i--) {
      unsigned int bit = (u >> i) & 1;
      printf("%u", bit);
      if (i == 31 || i == 23)
        printf(" ");
    }
    printf("\n");
  }
}

int main() {
  // 2^24 + 8*0.25，调整这8个0.25的位置

  test_w<<<1, 32>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  return 0;
}

/*
a100
2^24+2*1.0

// 这一条加不上，说明W<=8
    // if (threadIdx.x == 0) {
    //   a0 = a2 = 0b0'01111'0000000000'0'00000'0000000000; // 1.0||0.0
    //   b0 = 0b0'01111'0000000000'0'01111'0000000000; // 1.0
    //   unsigned int c_init = 0b0'10010111'00000000000000000000000;
    //   float c = *((float *)&c_init);
    //   c0 = c;
    // }
//这一条能加上，说明W>4
    if (threadIdx.x == 0 || threadIdx.x == 3) {
      a0 = 0b0'01111'0000000000'0'00000'0000000000; // 1.0||0.0
      b0 = 0b0'01111'0000000000'0'01111'0000000000; // 1.0
      unsigned int c_init = 0b0'10010111'00000000000000000000000;
      float c = *((float *)&c_init);
      c0 = c;
    }
//所以A100的W=8
*/