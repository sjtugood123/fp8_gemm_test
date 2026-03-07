#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// 编译命令参考: nvcc -arch=sm_120 test_mxfp4.cu -o test_mxfp4

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err),  \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void test_mxfp4_block_scale_kernel(float c_init_val) {
  // 1. 初始化累加器 C 为大数 (2^24)
  float c0 = c_init_val;
  float c1 = c_init_val;
  float c2 = c_init_val;
  float c3 = c_init_val;

  // 输出寄存器
  float d0, d1, d2, d3;

  // 2. 准备数据 A 和 B (FP4 E2M1)
  // 这里的位模式 0x22222222 对应每4位是 0010
  // 在 FP4 E2M1 中，0010 通常表示 1.0 (根据 Nvidia 映射)
  // 我们填满所有寄存器，意味着 K=64 维度的每一个元素都是 1.0
  uint32_t a_reg = 0x22222222;
  uint32_t b_reg = 0x22222222;

  // A 需要 4 个寄存器 (128 bits / 32 threads = 4 bits * 16 rows * 64 k / 32?
  // 实际上 m16n8k64 A是4个寄存器，B是2个寄存器)
  uint32_t a0 = a_reg, a1 = a_reg, a2 = a_reg, a3 = a_reg;
  uint32_t b0 = b_reg, b1 = b_reg;

  // 3. 准备 Scale (Block Scaling) - 这是核心魔法
  // 格式 UE8M0 (Unsigned Exponent 8-bit, Bias 127)
  // 目标：我们希望 64 个元素累加的总和刚好是 2.0
  // 当前：Sum_Raw = 64 * (1.0 * 1.0) = 64.0
  // 需要 Scale 因子 = 2.0 / 64.0 = 1/32 = 2^-5
  // Scale A 编码: Bias 127 + (-5) = 122 = 0x7A
  // Scale B 编码: Bias 127 + (0)  = 127 = 0x7F (即 1.0)

  uint32_t scale_a_val = 0x7F7F7F7F;
  uint32_t scale_b_val = 0x7F7F7F7F; // 每个字节都是 1.0

  // Triton PTX 中使用了 block_scale 指令，我们需要传入 scale 寄存器
  // 指令:
  // mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X...

  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale."
               "scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
               "{%0,  %1,  %2,  %3}, "  // Output D (4x f32)
               "{%4,  %5,  %6,  %7}, "  // Input A  (4x b32, packed fp4)
               "{%8,  %9}, "            // Input B  (2x b32, packed fp4)
               "{%10, %11, %12, %13}, " // Accumulator C (4x f32)
               "%14, "                  // Scale A (b32, packed ue8m0)
               "{0, 0}, "               // Scale A reserved/immediate structure
               "%15, "                  // Scale B (b32, packed ue8m0)
               "{0, 0};\n"              // Scale B reserved
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
                 "f"(c1), "f"(c2), "f"(c3), "r"(scale_a_val), "r"(scale_b_val));

  // 4. 检查结果
  // 只需要线程0打印即可
  if (threadIdx.x == 0) {
    printf("--------------------------------------------------\n");
    printf("Reference C_init : %.1f (2^24)\n", c_init_val);
    printf("Expected Add Val : 2.0 (From 64 * 1.0 * 1.0 * 1/32)\n");
    printf("Expected Result  : %.1f\n", c_init_val + 2.0f);
    printf("Actual Output d0 : %.1f\n", d0);

    float diff = d0 - c_init_val;
    printf("Actual Increment : %.1f\n", diff);

    if (diff > 1.0f) {
      printf("CONCLUSION: [Precision KEPT] Atomic accumulation worked!\n");
    } else {
      printf("CONCLUSION: [Precision LOST] Split accumulation occurred.\n");
    }
    printf("--------------------------------------------------\n");
  }
}

int main() {
  printf("=== SM120 MXFP4 (mxf4nvf4) Atomic K-Accumulation Test ===\n");

  // 2^24 = 16777216.0
  // 在此数值下，FP32 精度 gap 为 2.0
  uint32_t c_init_bits = 0x4B800000;
  float c_init = *((float *)&c_init_bits);

  // 启动 1 个 Warp
  test_mxfp4_block_scale_kernel<<<1, 32>>>(c_init);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  return 0;
}