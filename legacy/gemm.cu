// #include "c10/util/Exception.h"
// // #include <../include>
// // #include <ATen/ATen.h>
// // #include <cuda_runtime.h>
// // #include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAException.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <c10/cuda/CUDAStream.h>
// #include <cstdio>
// #include <cuda_bf16.h>
// #include <cuda_fp16.h>
// #include <iostream>
// #include <torch/all.h>

// // static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// __global__ void groupwise_gemm_fp8_kernel(
//     const c10::Float8_e4m3fn *__restrict__ a, // [M, K]
//     const float *__restrict__ a_scale,        // [M, K/group_size]
//     const c10::Float8_e4m3fn *__restrict__ w, // [K, N],已转置
//     const float *__restrict__ w_scale,        // [K/group_size, N]
//     __nv_bfloat16 *__restrict__ out,          // [M, N]
//     int M, int N, int K, int group_size, int tile_m, int tile_n) {
//   // 先测试一下能否直接相乘
//   // 测试每个线程负责output[block_x*tile_m+tid/4,block_y*tile_n+tid%4+j]
//   int block_x = blockIdx.x;
//   int block_y = blockIdx.y;
//   int tid = threadIdx.x;
//   int row = block_x * tile_m + tid / 4;
//   int col = block_y * tile_n + (tid % 4) * 8;
//   int print = 30;

//   float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
// #pragma unroll
//   for (int j = 0; j < 8; j++) {
//     for (int i = 0; i < K; i++) {
//       float a_s = a_scale[row * (K / group_size) + i / group_size];
//       float w_s = w_scale[(i / group_size) * N + col + j]; // 这里要先除G
//       c10::Float8_e4m3fn temp_a = a[row * K + i];
//       c10::Float8_e4m3fn temp_w = w[i * N + col + j];
//       // float temp = a[row * K + i] * w[i * N + col + j];
//       float temp = temp_a * temp_w;
//       c10::Float8_e4m3fn temp_2 = temp_a * temp_w;
//       c10::Float8_e4m3fn temp_3 = temp_a + temp_w;
//       float temp_4 = (float)temp_a * (float)temp_w;
//       if (block_x == 0 && block_y == 0 && tid == 0 && print > 0) {
//         printf("temp:%f\ttemp_2:%f\ttemp_3:%f\ttemp_4:%f\ttemp_a:%f\ttemp_w:%"
//                "f\t\n",
//                temp, (float)temp_2, (float)temp_3, (float)temp_4,
//                (float)temp_a, (float)temp_w);
//         print--;
//       }
//       temp = temp * a_s * w_s;
//       // out[row * N + col + j] += __float2bfloat16(temp);
//       acc[j] += temp;
//     }
//   }
// #pragma unroll
//   for (int j = 0; j < 8; j++) {
//     int current_col = col + j;
//     if (current_col < N) {
//       // 使用=，而不是+=
//       out[row * N + current_col] = __float2bfloat16(acc[j]);
//       // torch::BFloat16(acc[j]);
//     }
//   }
// };

// void groupwise_gemm_fp8(
//     // 只接受2维，reshape在调用之前做，w可以保证是2维
//     // w和w_scale必须要转置之后送进来
//     // output_bf16必须初始化为0
//     const torch::Tensor &a, // [m, k]
//     const torch::Tensor &a_scale,
//     const torch::Tensor &w, // [k, n]
//     const torch::Tensor &w_scale,
//     torch::Tensor &output_bf16, // [m, n]
//     int64_t group_size) {
//   TORCH_CHECK(a.is_contiguous());
//   TORCH_CHECK(a_scale.is_contiguous());
//   TORCH_CHECK(w.is_contiguous());
//   TORCH_CHECK(w_scale.is_contiguous());
//   TORCH_CHECK(output_bf16.is_contiguous());
//   TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn);
//   TORCH_CHECK(w.scalar_type() == torch::kFloat8_e4m3fn);
//   TORCH_CHECK(a_scale.scalar_type() == torch::kFloat32);
//   TORCH_CHECK(w_scale.scalar_type() == torch::kFloat32);
//   TORCH_CHECK(output_bf16.scalar_type() == torch::kBFloat16);
//   //   TORCH_CHECK(input.dim() == 2);
//   TORCH_CHECK(a_scale.dim() == 2);
//   TORCH_CHECK(w_scale.dim() == 2);
//   TORCH_CHECK(output_bf16.dim() == 2);
//   TORCH_CHECK(group_size % 16 == 0);
//   // 必须在同一个CUDA设备上
//   TORCH_CHECK(a.is_cuda() && w.is_cuda() && a_scale.is_cuda() &&
//                   w_scale.is_cuda() && output_bf16.is_cuda(),
//               "所有张量必须在CUDA上");
//   TORCH_CHECK(a.device() == w.device() && a.device() == a_scale.device() &&
//                   a.device() == w_scale.device() &&
//                   a.device() == output_bf16.device(),
//               "所有张量必须在同一设备上");

//   const int a_rows = a.size(0);
//   const int a_cols = a.size(1);
//   const int w_rows = w.size(0);
//   const int w_cols = w.size(1);

//   TORCH_CHECK(a_cols % group_size == 0, "a_cols必须被 group_size整除");
//   TORCH_CHECK(a_scale.size(0) == a_rows, "a_scale 行数不匹配");
//   TORCH_CHECK(a_scale.size(1) == a_cols / group_size, "a_scale列数不匹配");

//   TORCH_CHECK(w_rows == a_cols, "a@w不匹配,是不是忘了转置w?");
//   TORCH_CHECK(w_rows % group_size == 0, "w_rows必须被 group_size整除");
//   TORCH_CHECK(w_scale.size(0) == w_rows / group_size, "w_scale 行数不匹配");
//   TORCH_CHECK(w_scale.size(1) == w_cols, "w_scale 列数不匹配");

//   at::cuda::CUDAGuard device_guard(a.device());
//   auto stream = at::cuda::getCurrentCUDAStream(a.device().index());

//   int tile_m = 32;
//   int tile_n = 32;
//   TORCH_CHECK(a_rows % tile_m == 0);
//   TORCH_CHECK(w_cols % tile_n == 0);
//   dim3 grid((a_rows / tile_m), (w_cols / tile_n), 1);
//   dim3 block(128, 1, 1);

//   const c10::Float8_e4m3fn *a_ptr =
//       reinterpret_cast<const c10::Float8_e4m3fn *>(a.data_ptr());
//   const float *a_scale_ptr = a_scale.data_ptr<float>();
//   const c10::Float8_e4m3fn *w_ptr =
//       reinterpret_cast<const c10::Float8_e4m3fn *>(w.data_ptr());
//   const float *w_scale_ptr = w_scale.data_ptr<float>();
//   // at::BFloat16 *out_ptr =
//   //     reinterpret_cast<at::BFloat16 *>(output_bf16.data_ptr());
//   __nv_bfloat16 *out_ptr =
//       reinterpret_cast<__nv_bfloat16 *>(output_bf16.data_ptr());

//   groupwise_gemm_fp8_kernel<<<grid, block, 0, stream>>>(
//       a_ptr, a_scale_ptr, w_ptr, w_scale_ptr, out_ptr, a_rows, w_cols,
//       a_cols, static_cast<int>(group_size), tile_m, tile_n);
// }