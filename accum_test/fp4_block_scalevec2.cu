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

// Pack 8 FP4 nibbles into one .b32 (two nibbles per byte: low[3:0], high[7:4])
__host__ __device__ static inline uint32_t
pack_fp4_2perbyte(uint8_t n0, uint8_t n1, uint8_t n2, uint8_t n3, uint8_t n4,
                  uint8_t n5, uint8_t n6, uint8_t n7) {
  return ((uint32_t)((n0 & 0xF) | ((n1 & 0xF) << 4))) |
         ((uint32_t)((n2 & 0xF) | ((n3 & 0xF) << 4)) << 8) |
         ((uint32_t)((n4 & 0xF) | ((n5 & 0xF) << 4)) << 16) |
         ((uint32_t)((n6 & 0xF) | ((n7 & 0xF) << 4)) << 24);
}

__global__ void test_sm120_fp4_mma_strict_packing() {
  // 1. C 矩阵初始化
  // (1.0 * 2^23)
  // uint32_t t23 = 0b0'10010110'00000000000000000000000;
  // (1.0 * 2^24)
  uint32_t t24 = 0b0'10010111'00000000000000000000000;
  // (1.0 * 2^25)
  uint32_t t25 = 0b0'10011000'00000000000000000000000;
  uint32_t t26 = 0b0'10011001'00000000000000000000000;
  float t_f = *((float *)&t26);
  float d0 = t_f;
  float d1 = t_f;
  float d2 = t_f;
  float d3 = t_f;

  // 2. 输出变量
  // float d0, d1, d2, d3;

  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  uint32_t b0 = 0, b1 = 0;
  // UE8M0 格式：8-bit 指数，Bias 127。 127(0x7F) -> 2^0 = 1.0
  uint32_t sfa0 = 0x7F7F7F7F;
  uint32_t sfb0 = 0x7F7F7F7F;

  // 4个0.5
  // if (threadIdx.x == 0) {
  //   a0 = pack_fp4_2perbyte(0x1, 0x1, 0x1, 0x1, 0x0, 0x0, 0x0, 0x0); // 0.5
  //   b0 = pack_fp4_2perbyte(0x2, 0x2, 0x2, 0x2, 0x0, 0x0, 0x0, 0x0); // 1.0
  // }

  // 2^24+8*0.25 PASS
  // if (threadIdx.x == 0) {
  //   a0 = pack_fp4_2perbyte(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1);
  //   b0 = pack_fp4_2perbyte(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1);
  // }

  // 2^25 + 16*0.25 PASS
  // if (threadIdx.x == 0 || threadIdx.x == 1) {
  //   a0 = pack_fp4_2perbyte(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1);
  //   b0 = pack_fp4_2perbyte(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1);
  // }

  // 测试一下修改scale可不可以让fp4表示更低的数，8个0.5，Ascale设成0.5，Bscale保持1不变
  // sfa0 = 0x7E7E7E7E; // 0.5
  // if (threadIdx.x == 0) {
  //   a0 = pack_fp4_2perbyte(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1);
  //   b0 = pack_fp4_2perbyte(0x2, 0x2, 0x2, 0x2, 0x2, 0x2, 0x2, 0x2);
  // }
  // 上面的猜测成立

  // 所以2^26 + 64*0.125就是64个0.5*0.5*0.5
  // 这一组precision也lost，这不符合预期
  // 猜测1：scale参与的0.0625和单纯的ab参与的不一样
  // 猜测2：fragment布局超过16失效？
  if (threadIdx.x < 4) {
    a0 = pack_fp4_2perbyte(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1);
    a2 = a0;
    b0 = pack_fp4_2perbyte(0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1);
    b1 = b0;
    sfa0 = 0x7E7E7E7E;
  }
  // 猜测2成立，每个线程负责的位置要看准，调整之后也PASS了
  // 具体布局：https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16864
  // 但实际上64个元素是上限了，因为m16n8k64

  // 结论是至少29 bit(不包含隐藏位)
  // 下一步试试k128看一下精度到哪，还有mxf8f6f4能否也能达到这个精度

  /*
  scale_vec::1X表示整个K维度的向量共享一个缩放因子
  2X就是K维度切成两份，每份有自己的sf
  */
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
               "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
               "{%0,  %1,  %2,  %3}," // D (4 floats)
               "{%4,  %5,  %6,  %7}," // A (4 x b32) - 对应 a0, a1, a2, a3
               "{%8,  %9},"           // B (2 x b32) - 对应 b0, b1
               "{%0,  %1,  %2,  %3}," // C (4 floats)
               "%10, {0, 0}, "        // scaleA 及其选择器
               "%11, {0, 0};"         // scaleB 及其选择器
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "r"(sfa0), "r"(sfb0));

  // 7. 打印输出
  if (threadIdx.x == 0) {
    // printf("\nThread 0 Result:\n");
    printf("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64."
           "row.col.f32.e2m1.e2m1.f32.ue8m0 \n");
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