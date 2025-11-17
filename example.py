import torch
import scaled_fp8_ops as ops 
import time
import fp8_pseudo_quantize


# --- 0. 配置 ---
FLOAT8_E4M3_MAX = 448.0
FLOAT8_E4M3_MIN = -448.0

# 内核的块粒度
# 由于5090的smem大小限制，mmatile最大是128*128*128
# 粒度最大也是如此，因为底层写死每个tile里至少有一个粒度
M_GRANULARITY = 128
K_GRANULARITY = 128
N_GRANULARITY = 128

# 尺寸
M = 1024
K = 2048
N = 4096

# 计时配置
WARMUP_ITERS = 3
N_ITERS = 5

# 检查维度是否有效
assert K % K_GRANULARITY == 0, f"K 必须是 {K_GRANULARITY} 的倍数"
assert K % K_GRANULARITY == 0, f"K 必须是 {K_GRANULARITY} 的倍数"
assert N % N_GRANULARITY == 0, f"N 必须是 {N_GRANULARITY} 的倍数"

def calculate_diff(a, b, name_a, name_b):
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    
    abs_diff = (a_f32 - b_f32).abs()
    max_abs_diff = abs_diff.max()
    
    print(f"\n比较: {name_a} vs {name_b}")
    print(f"  最大绝对差异: {max_abs_diff.item()}")

print(f"运行 GEMM: M={M}, K={K}, N={N}")
print(f"A 缩放粒度: ({M_GRANULARITY}, {K_GRANULARITY})")
print(f"B 缩放粒度: ({K_GRANULARITY}, {N_GRANULARITY})")
print(f"预热迭代: {WARMUP_ITERS}, 计时迭代: {N_ITERS}")
print("-" * 30)

# --- 1. 创建输入张量 ---
a_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
b_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()

# ===================================================================
# 方法 1: 伪量化 (fp8_pseudo_quantize_groupwise) + BF16 GEMM
# ===================================================================

# 预热
for _ in range(WARMUP_ITERS):
    a_pseudo_q = fp8_pseudo_quantize.fp8_pseudo_quantize_groupwise(a_bf16)
    b_pseudo_q = fp8_pseudo_quantize.fp8_pseudo_quantize_groupwise(b_bf16)
    # print(a_pseudo_q.dtype)
    out_pseudo_temp = a_pseudo_q @ b_pseudo_q.T
torch.cuda.synchronize()

# 计时
start_time = time.time()
for _ in range(N_ITERS):
    a_pseudo_q = fp8_pseudo_quantize.fp8_pseudo_quantize_groupwise(a_bf16)
    b_pseudo_q = fp8_pseudo_quantize.fp8_pseudo_quantize_groupwise(b_bf16)
    out_pseudo_temp = a_pseudo_q @ b_pseudo_q.T
torch.cuda.synchronize()
end_time = time.time()

# 保存最后一次运行的结果
out_pseudo = out_pseudo_temp
print(f"pseudo quant & bf16 gemm: {(end_time - start_time) / N_ITERS * 1000:.4f} ms")


# ===================================================================
# 方法 2 & 3 的共同准备工作：执行逐块量化
# ===================================================================
# A
a_q_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device="cuda")
a_s_f32 = torch.empty((M // M_GRANULARITY, K // K_GRANULARITY), dtype=torch.float32, device="cuda")
# B
b_q_fp8 = torch.empty((N, K), dtype=torch.float8_e4m3fn, device="cuda")
b_s_f32 = torch.empty((N // N_GRANULARITY, K // K_GRANULARITY), dtype=torch.float32, device="cuda")

# 量化 A (粒度 128 x 128)
for m_block_idx in range(M // M_GRANULARITY):
    m_start = m_block_idx * M_GRANULARITY
    m_end = (m_block_idx + 1) * M_GRANULARITY

    for k_block_idx in range(K // K_GRANULARITY):
        k_start = k_block_idx * K_GRANULARITY
        k_end = (k_block_idx + 1) * K_GRANULARITY
        
        a_chunk = a_bf16[m_start:m_end, k_start:k_end]
        abs_max = a_chunk.abs().max()
        scale = abs_max / FLOAT8_E4M3_MAX
        scale = torch.clamp(scale, min=1e-8) 
        
        a_s_f32[m_block_idx, k_block_idx] = scale
        a_q_fp8[m_start:m_end, k_start:k_end] = torch.clamp(
            a_chunk / scale, min=FLOAT8_E4M3_MIN, max=FLOAT8_E4M3_MAX
        ).to(torch.float8_e4m3fn)

# 量化 B (粒度 128 x 128)
for n_block_idx in range(N // N_GRANULARITY):
    n_start = n_block_idx * N_GRANULARITY
    n_end = (n_block_idx + 1) * N_GRANULARITY

    for k_block_idx in range(K // K_GRANULARITY):
        k_start = k_block_idx * K_GRANULARITY
        k_end = (k_block_idx + 1) * K_GRANULARITY

        b_chunk = b_bf16[n_start:n_end, k_start:k_end]
        abs_max = b_chunk.abs().max()
        scale = abs_max / FLOAT8_E4M3_MAX
        scale = torch.clamp(scale, min=1e-8)
            
        b_s_f32[n_block_idx, k_block_idx] = scale
        b_q_fp8[n_start:n_end, k_start:k_end] = torch.clamp(
            b_chunk / scale, min=FLOAT8_E4M3_MIN, max=FLOAT8_E4M3_MAX
        ).to(torch.float8_e4m3fn)


# 准备 CUTLASS 内核所需的转置输入
# b_q_fp8_t = b_q_fp8.T.contiguous()
# b_s_f32_t = b_s_f32.T.contiguous()
# a_q_fp8_t = a_q_fp8.T.contiguous()
a_s_f32_t = a_s_f32.T.contiguous()
out_cutlass = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()

# ===================================================================
# 方法 2: CUTLASS FP8 内核 (cutlass_scaled_mm_blockwise_sm120_fp8)
# ===================================================================

# 预热
for _ in range(WARMUP_ITERS):
    ops.cutlass_scaled_mm_blockwise_sm120_fp8(
        out_cutlass, a_q_fp8, b_q_fp8, a_s_f32_t, b_s_f32
    )
torch.cuda.synchronize()

start_time = time.time()
for _ in range(N_ITERS):
    ops.cutlass_scaled_mm_blockwise_sm120_fp8(
        out_cutlass,
        a_q_fp8, 
        b_q_fp8,
        a_s_f32_t, 
        b_s_f32
    )
torch.cuda.synchronize()
end_time = time.time()
print(f"real quant & cutlass fp8 gemm: {(end_time - start_time) / N_ITERS * 1000:.4f} ms")


# ===================================================================
# 方法 3 的准备工作：执行逐块反量化
# ===================================================================
# print("\n--- 3. 准备: Python 逐块反量化 (仅执行一次) ---")
a_dequant_f32 = torch.empty((M, K), dtype=torch.float32, device="cuda")
b_dequant_f32 = torch.empty((N, K), dtype=torch.float32, device="cuda")

# 反量化 A
for m_block_idx in range(M // M_GRANULARITY):
    ms = m_block_idx * M_GRANULARITY
    me = (m_block_idx + 1) * M_GRANULARITY

    for k_block_idx in range(K // K_GRANULARITY):
        ks = k_block_idx * K_GRANULARITY
        ke = (k_block_idx + 1) * K_GRANULARITY
        scale = a_s_f32[m_block_idx, k_block_idx].to(torch.float32)
        a_dequant_f32[ms:me, ks:ke] = a_q_fp8[ms:me, ks:ke].to(torch.float32) * scale

# 反量化 B
for n_block_idx in range(N // N_GRANULARITY):
    ns = n_block_idx * N_GRANULARITY
    ne = (n_block_idx + 1) * N_GRANULARITY
    for k_block_idx in range(K // K_GRANULARITY):
        ks = k_block_idx * K_GRANULARITY
        ke = (k_block_idx + 1) * K_GRANULARITY

    
        scale = b_s_f32[n_block_idx, k_block_idx].to(torch.float32)
        b_dequant_f32[ns:ne, ks:ke] = b_q_fp8[ns:ne, ks:ke].to(torch.float32) * scale
# print("反量化完成。")
torch.cuda.synchronize()

# ===================================================================
# 方法 3: 参考 FP32 GEMM (来自反量化的数据)
# ===================================================================
# print("\n--- 3. 计时: 参考 FP32 GEMM (来自反量化) ---")

# 预热
for _ in range(WARMUP_ITERS):
    out_ref_temp = (a_dequant_f32 @ b_dequant_f32.T).to(torch.bfloat16)
torch.cuda.synchronize()

# 计时
start_time = time.time()
for _ in range(N_ITERS):
    out_ref_temp = (a_dequant_f32 @ b_dequant_f32.T).to(torch.bfloat16)
torch.cuda.synchronize()
end_time = time.time()

# 保存最后一次运行的结果
out_ref = out_ref_temp
print(f"real quant & fp32 gemm: {(end_time - start_time) / N_ITERS * 1000:.4f} ms")


# ===================================================================
# 结果比较
# ===================================================================
print("\n" + "=" * 30)
print(" 结果比较 ")
print("=" * 30)

print(f"pseudo [{out_pseudo.shape}, {out_pseudo.dtype}]:")
print(out_pseudo[:4, :4])

print(f"\nCUTLASS [{out_cutlass.shape}, {out_cutlass.dtype}]:")
print(out_cutlass[:4, :4])

print(f"\nref [{out_ref.shape}, {out_ref.dtype}]:")
print(out_ref[:4, :4])


calculate_diff(out_pseudo, out_cutlass, "pseudo", "CUTLASS")
calculate_diff(out_pseudo, out_ref, "pseudo", "ref")
calculate_diff(out_cutlass, out_ref, "CUTLASS", "ref")