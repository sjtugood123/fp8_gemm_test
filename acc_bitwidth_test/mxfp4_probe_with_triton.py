import pytest
import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from triton._internal_testing import is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4, is_hip_cdna

@triton.jit
def block_scale_fp4_matmul(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_scale,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,
        PACK_ALONG_K: tl.constexpr,
        c_init: tl.constexpr):  #
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    PACKING_ALONG_M_N: tl.constexpr = 1 if PACK_ALONG_K else 2
    offs_am_packed = (pid_m * (BLOCK_M // PACKING_ALONG_M_N) + tl.arange(0, BLOCK_M // PACKING_ALONG_M_N))
    offs_bn_packed = (pid_n * (BLOCK_N // PACKING_ALONG_M_N) + tl.arange(0, BLOCK_N // PACKING_ALONG_M_N))
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2 if PACK_ALONG_K else BLOCK_K

    # Two e2m1 values per K
    offs_k = tl.arange(0, BLOCK_K_PACKED)
    offs_scale_k = tl.arange(0, BLOCK_K // VEC_SIZE)
    if a_scale is not None:
        a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    if b_scale is not None:
        b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + (offs_am_packed[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn_packed[None, :] * stride_bn)
    # accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    accumulator = tl.full((BLOCK_M, BLOCK_N), c_init, dtype=output_ptr.dtype.element_ty)
    tl.static_print(f"accumulator dtype: ", accumulator.dtype)
    
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if a_scale is not None:
            scale_a = tl.load(a_scale_ptr)
        else:
            scale_a = None
        if b_scale is not None:
            scale_b = tl.load(b_scale_ptr)
        else:
            scale_b = None
        accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator, lhs_k_pack=PACK_ALONG_K,
                                    rhs_k_pack=PACK_ALONG_K)
        a_ptrs += (BLOCK_K_PACKED) * stride_ak
        b_ptrs += (BLOCK_K_PACKED) * stride_bk
        if a_scale is not None:
            a_scale_ptr += BLOCK_K // VEC_SIZE
        if b_scale is not None:
            b_scale_ptr += BLOCK_K // VEC_SIZE
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


def test_block_scale_fp4(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, VEC_SIZE, with_a_scale, with_b_scale, pack_along_k,
                         scale_type, nonKDim, device, c_init, num_add, add_val):
    
    assert M % BLOCK_M == 0,"M error"
    assert N % BLOCK_N == 0,"N error"
    assert K % BLOCK_K == 0,"K error"
    assert num_add <= K,"num_add error"
    if is_cuda():
        if scale_type == "float8_e4m3fn" and not pack_along_k:
            pytest.skip("Packing along K is required for float8_e4m3fn")
        if torch.cuda.get_device_capability()[0] != 10 and torch.cuda.get_device_capability()[0] != 12:
            pytest.skip("Requires compute capability == 10 or 12")
        if torch.cuda.get_device_capability()[0] == 12 and pack_along_k is False:
            pytest.skip("Packing along M, N is not supported on SM120")
        if not (with_a_scale and with_b_scale):
            pytest.skip("None aScale/bScale is only tested on AMD backend for now")
    elif is_hip():
        if not is_hip_cdna4():
            pytest.skip("Scaled fp4 matmul is only natively supported on CDNA4")
        if scale_type != 'float8_e8m0fnu':
            pytest.skip("CDNA4 only supports E8M0 scale")
        if (nonKDim == 16 and BLOCK_K < 128) or (nonKDim == 32 and BLOCK_K < 64):
            pytest.skip(f"CDNA4 does not support {BLOCK_K=} for scaled mfma {nonKDim=} variants")

    NUM_STAGES = 1
    torch.manual_seed(42)
    packing_dim = 1 if pack_along_k else 0

    
    # 直接传tensor
    test_tensor_a = torch.zeros((M, K), dtype=float, device=device)
    for i in range(num_add):
        test_tensor_a[0][i]=0.5 if add_val<=0.25 else add_val
    a = MXFP4Tensor(size=(M, K), device=device, data = test_tensor_a).to_packed_tensor(dim=packing_dim)
    # a_mxfp4 = MXFP4Tensor(size=(M, K), device=device).random()
    # a = a_mxfp4.to_packed_tensor(dim=packing_dim)
    # print(type(a))
    # print(a.device)
    # print(a_mxfp4.device)
    # Generate b with k-major layout, pack two e2m1 along k or n, then logical transpose to K, N
    test_tensor_b = torch.zeros((N, K), dtype=float, device=device)
    for i in range(num_add):
        test_tensor_b[0][i] = 0.5 if add_val<=0.25 else 1.0#后面有转置
    b = MXFP4Tensor(size=(N, K), device=device, data = test_tensor_b).to_packed_tensor(dim=packing_dim).T
    # b_mxfp4 = MXFP4Tensor(size=(N, K), device=device).random()
    # b = b_mxfp4.to_packed_tensor(dim=packing_dim).T
    # No need to pack along K since we convert each e2m1 to f32 directly for the reference matmul
    # b_ref = b_mxfp4.to(torch.float32).T

    #https://docs.nvidia.com/cuda/parallel-thread-execution/_images/mma-block-scaling.png
    a_size = (M, (K + VEC_SIZE - 1) // VEC_SIZE)#[M,2]
    b_size = (N, (K + VEC_SIZE - 1) // VEC_SIZE)#[N,2]
    # a_scale = torch.rand(a_size, device=device)
    # b_scale = torch.rand(b_size, device=device)
    a_scale = torch.full(a_size, 0.5 if add_val<=0.125 else 1.0, device=device)# 1.0，这里不能传0b01111111，因为MXScaleTensor的init构造方式
    b_scale = torch.full(b_size, 0.5 if add_val<=0.0625 else 1.0, device=device)
    if scale_type == "float8_e8m0fnu":
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data#[M,2]  
        b_scale = b_scale_ref.data#[N,2]
        # print(b_scale.shape)
    elif scale_type == "float8_e4m3fn":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale
    a_scale_ref = a_scale_ref.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_ref = b_scale_ref.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    stride_scale = a_scale.stride(0)

    # ref_out = torch.matmul(a_mxfp4.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)

    output = a.new_empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    kernel_kwargs = {}
    if is_hip():
        kernel_kwargs["matrix_instr_nonkdim"] = nonKDim


    # with torch.cuda.nvtx.range("MyFP4_MMA_Range"):
    #     k = block_scale_fp4_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, stride_scale, a.stride(0), a.stride(1),
    #                                  b.stride(0), b.stride(1), output.stride(0), output.stride(1), VEC_SIZE, BLOCK_M,
    #                                  BLOCK_N, BLOCK_K, NUM_STAGES=NUM_STAGES, PACK_ALONG_K=pack_along_k, c_init=c_init,
    #                                  **kernel_kwargs)
    #     torch.cuda.synchronize()
    # torch.cuda.nvtx.range_pop()

    k = block_scale_fp4_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, stride_scale, a.stride(0), a.stride(1),
                                     b.stride(0), b.stride(1), output.stride(0), output.stride(1), VEC_SIZE, BLOCK_M,
                                     BLOCK_N, BLOCK_K, NUM_STAGES=NUM_STAGES, PACK_ALONG_K=pack_along_k, c_init=c_init,
                                     **kernel_kwargs)
    # torch.testing.assert_close(ref_out, output, atol=1e-2, rtol=1e-2)
    expected = float(c_init) + add_val * num_add
    # print(expected)
    abs_error = abs(output[0,0].double() - expected)
    # print(abs_error)
    to_print = "Precision LOST." if abs_error > 1e-5 else "Precision KEPT."
    print(f"output:{output[0,0]} {to_print}\n")
    # if is_cuda():
    #     ptx = k.asm["ptx"]
    #     if pack_along_k:
    #         assert "mxf4nvf4" in ptx
    #     else:
    #         assert "kind::mxf8f6f4" in ptx

    if is_cuda():
        ptx = k.asm["ptx"]
        filename = f"kernel_{num_add}_{add_val}_{c_init}.ptx"
        with open(filename, 'w') as f:
            f.write(ptx)
        print(f"PTX 已保存到: {filename}")
        # for i, line in enumerate(ptx.split('\n'), 1):
        #     # if 'mma' in line:
        #         print(f"{i}: {line}")



if __name__ == "__main__":
    """
    test_rz:
    2^23      + 1*0.5 = 2^23
    (2^23+1)  + 1*0.5 = 2^23+1


    test_acc_bitwidth
    2^23 + 2*1   = 2^23 + 2 Pass
    2^23 + 8*0.25= 2^23 + 2 Pass

    2^24 + 16*0.125= 2^24 + 2 Pass
    2^24 + 32*0.0625= 2^24 + 2 Pass

    2^25 + 64*0.0625=2^25 + 4 Pass

    2^26 + 64*0.125=2^26 + 8 Pass
    2^26 + 128*0.0625=2^26      FAIL
    """
    #0.5:a->0.5，其余是1.0
    #0.25:a->0.5,b->0.5 其余是1.0
    #0.125:a,b,as->0.5,bs->1.0
    #0.0625:全是0.5

    param_combinations = []
    param_combinations = [
        # (num_add, add_val, c_init_value)
        # (1, 0.5, 1 << 23),
        # (1, 0.5, (1 << 23) + 1),
        # (2, 0.5, 1 << 23),
        # (4, 0.5, 1 << 23),
        # (8, 0.5, 1 << 23),
        # (2, 1.0, 1 << 23),
        # (3, 1.0, 1 << 23),
        # (4, 1.0, 1 << 23),
        # (8, 0.25, 1 << 23),
        # (4, 0.5, 1 << 24),
        # (16, 0.125, 1 << 24),
        # (32, 0.0625, 1 << 24),
        # (64, 0.0625, 1 << 25),
        # (128, 0.0625, 1 << 25),
        # (128, 0.125, 1 << 26),
        (256, 0.0625, 1 << 26),
    ]
    
    print("=" * 60)
    print("开始测试不同参数组合")
    print("=" * 60)
    
    # 遍历所有参数组合运行测试
    for i, (num_add, add_val, c_init_value) in enumerate(param_combinations, 1):
        print(f"参数: num_add={num_add}, add_val={add_val}, c_init={c_init_value}(2^{c_init_value.bit_length()-1})")
        try:
            test_block_scale_fp4(
                # MNK不能降到64以下，为了ptx长度易读，可以降到64
                # 但是如果要做后面的测试还是需要扩展一下，后面的num_add太长
                M=1024,
                N=1024,
                K=1024,
                # M=64,  
                # N=64,
                # K=64,
                BLOCK_M=64,
                BLOCK_N=64,
                BLOCK_K=64,
                VEC_SIZE=32,
                with_a_scale=True,
                with_b_scale=True,
                pack_along_k=True,
                scale_type="float8_e8m0fnu",
                nonKDim=0,
                device="cuda",
                c_init=c_init_value,
                num_add=num_add,
                add_val=add_val
            )
        except Exception as e:
            print(f"测试组合 {i} 失败{e}\n\n")
    print(f"Triton: {triton.__version__}")
    print("\nConclusion: Internal accumulator for mxfp4 has 29 bits precision.\n")
    