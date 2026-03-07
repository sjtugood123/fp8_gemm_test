#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>

/*
1.00000 00001 * 1.00000 00001
=
1.00000 00010 00000 00001 不够测22，因为前面没有进位



1.11111 11111 * 1.11111 11111 = 11.1111 1111 0000 0000 0001

换成fp32就是：
0 10000000 11111111100000000000100
那么c是
0 10000000 00000000011111111111100
如果是lossless,结果应该是6
如果有loss，结果应该小于6

*/

__global__ void test_mul_lossless(float *out_real) {

  unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  unsigned int b0 = 0, b1 = 0;
  unsigned int c_init = 0b0'10000000'00000000011111111111100;
  float c = *((float *)&c_init);
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  if (threadIdx.x == 0) {
    // 0'01111'11111 11111
    a0 = b0 = 0b0'01111'1111111111'0'00000'0000000000;
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
    out_real[0] = d0;
  }
}

int main() {
  float *h_real, *d_real;
  h_real = new float[1];
  cudaMalloc(&d_real, sizeof(float));

  test_mul_lossless<<<1, 32>>>(d_real);

  cudaMemcpy(h_real, d_real, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << std::fixed << std::setprecision(10);
  std::cout << "Real TC Output (d0):     " << h_real[0] << std::endl;

  // if (1) {
  //   std::cout << "\nRESULT: Hardware MUL is LOSSLESS!" << std::endl;
  //   std::cout
  //       << "Reason: The extra 2^-20 precision was preserved and added to C."
  //       << std::endl;
  // } else {
  //   std::cout << "\nRESULT: Hardware MUL is TRUNCATED." << std::endl;
  // }

  return 0;
}