import torch
import scaled_fp8_ops as ops 

FLOAT8_E4M3_MAX = 448.0
FLOAT8_E4M3_MIN = -448.0

# 内核的块粒度
# 由于5090的smem大小限制，mmatile最大是128*128*128
# 粒度最大也是如此，因为底层写死每个tile里至少有一个粒度
M_GRANULARITY = 128
K_GRANULARITY = 128
N_GRANULARITY = 128

# 尺寸
# M = 1024
# K = 2048
# N = 4096
M = 128
K = 128
N = 128

c_init_bin = 0b01001011000000000000000000000000; # 8388608.0f (1.0 * 2^23)

a_bf16 = torch.zeros((M, K), dtype=torch.bfloat16, device="cuda")
b_bf16 = torch.zeros((N, K), dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()

# A
a_q_fp8 = torch.zeros((M, K), dtype=torch.float8_e4m3fn, device="cuda")
a_s_f32 = torch.zeros((M // M_GRANULARITY, K // K_GRANULARITY), dtype=torch.float32, device="cuda")
a_s_f32[0,0] = 1.0
# B
b_q_fp8 = torch.zeros((N, K), dtype=torch.float8_e4m3fn, device="cuda")
b_s_f32 = torch.zeros((N // N_GRANULARITY, K // K_GRANULARITY), dtype=torch.float32, device="cuda")
b_s_f32[0,0] = 1.0
#
a_q_fp8[0,0] = 1.0
b_q_fp8[0,0] = 1.0

a_s_f32_t = a_s_f32.T.contiguous()
# out_cutlass = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
out_cutlass = torch.zeros((M, N), dtype=torch.float32, device="cuda")
torch.cuda.synchronize()

ops.cutlass_scaled_mm_blockwise_sm120_fp8(
        out_cutlass,
        a_q_fp8, 
        b_q_fp8,
        a_s_f32_t, 
        b_s_f32
)

print(f"\nCUTLASS [{out_cutlass.shape}, {out_cutlass.dtype}]:")
print(out_cutlass[:4, :4])
