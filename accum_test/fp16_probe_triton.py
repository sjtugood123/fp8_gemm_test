import torch
import triton
import triton.language as tl

@triton.jit
def matmul_precision_kernel(
    a_ptr, b_ptr, output_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 新增基准值
    BASE_VAL: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, 
    PRECISION: tl.constexpr = "ieee",
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # --- 关键修改点：累加器初始化为基准大数 ---
    # output_ptr.dtype.element_ty 通常是 fp32
    accumulator = tl.full((BLOCK_M, BLOCK_N), BASE_VAL, dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # 执行 dot 累加
        accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32, input_precision=PRECISION)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 结果存回
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator)

def run_precision_probe(base_val, total_k, a_val, b_val):
    # 配置
    M, N, K = 16, 16, total_k
    A = torch.full((M, K), a_val, device='cuda', dtype=torch.float16)
    B = torch.full((K, N), b_val, device='cuda', dtype=torch.float16)
    C = torch.zeros((M, N), device='cuda', dtype=torch.float32)

    grid = (1,) # 只跑一个 block
    matmul_precision_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BASE_VAL=float(base_val),
        BLOCK_M=16, BLOCK_N=8, BLOCK_K=64,
        NUM_STAGES=1
    )

    expected = base_val + (total_k * a_val * b_val)
    actual = C[0, 0].item()
    
    
    print(f"Probe: {base_val} + ({total_k} * {a_val * b_val})")
    print(f"  Result: {actual} (Exp: {expected})")

    if actual == expected:
        return True
    else:
        return False

# run_precision_probe(2**24, 1, 1.0, 1.0)#rz

def probe():
    acc_bitwdth=-1
    if not run_precision_probe(2**23, 1, 1.0, 1.0):
        print(f"acc_bitwidth:{acc_bitwdth}")
        return 
    acc_bitwdth=24

    if not run_precision_probe(2**24, 4, 0.5, 1.0):
        print(f"acc_bitwidth:{acc_bitwdth}")
        return 
    acc_bitwdth=25

    if not run_precision_probe(2**24, 8, 0.25, 1.0):
        print(f"acc_bitwidth:{acc_bitwdth}")
        return 
    acc_bitwdth=26

    if not run_precision_probe(2**24, 16, 0.125, 1.0):
        print(f"acc_bitwidth:{acc_bitwdth}")
        return 
    acc_bitwdth=27

    if not run_precision_probe(2**24, 32, 0.125, 0.5):
        print(f"acc_bitwidth:{acc_bitwdth}")
        return 
    acc_bitwdth=28

    if not run_precision_probe(2**24, 64, 0.125, 0.25):
        print(f"acc_bitwidth:{acc_bitwdth}")
        return 
    acc_bitwdth=29
    print(f"acc_bitwidth:{acc_bitwdth}")

probe()




