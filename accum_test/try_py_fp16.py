import sys
import torch
import numpy as np
import argparse

def probe_gemm_with_sparse_logic(api, c_init, num, add_val):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 理论参考值
    ref_result = float(c_init) + (int(num) * float(add_val))
    
    # 为了触发 Tensor Core，设置标准维度 (M, N, K)
    # K 取 num 和 16 的最大值，并向上取 16 的倍数
    M, N, K = 256, 256, 256
    # K = max(16, ((int(num) + 15) // 16) * 16)
    assert(K>=num)
    
    # 构造 A, B 矩阵并填充 0
    a_torch = torch.zeros((M, K), dtype=torch.float16, device=device)
    b_torch = torch.zeros((K, N), dtype=torch.float16, device=device)
    # try_torch = torch.zeros((K, N), dtype=torch.float4_e2m1fn_x2, device=device)
    
    # 模仿 CUDA 逻辑：只给前 num 个位置赋值
    # 这样 A[0,:] 和 B[:,0] 的点积就是 num * add_val
    a_torch[0, :int(num)] = float(add_val)
    b_torch[:int(num), 0] = 1.0
    
    # C 矩阵
    c_torch = torch.full((M, N), float(c_init), dtype=torch.float32, device=device)

    print(f"\n[ 实验配置 ] API: {api} | C: {c_init} | 累加次数 (num): {num} | 步长 (add): {add_val}")
    print(f"矩阵维度: M={M}, N={N}, K={K} (实际有效 K={num})")

    # 执行计算
    if api == 'mm':
        res = torch.mm(a_torch, b_torch) + c_torch
    elif api == 'matmul':
        res = torch.matmul(a_torch, b_torch) + c_torch
    elif api == 'at':
        res = (a_torch @ b_torch) + c_torch
    elif api == 'addmm':
        res = torch.addmm(c_torch, a_torch, b_torch, out_dtype=torch.float32)
    elif api == 'einsum':
        res = torch.einsum('mk,kn->mn', a_torch, b_torch) + c_torch
    
    actual = res[0, 0].item() if api != 'np' else res[0, 0]
    
    print("-" * 40)
    print(f"理论值: {ref_result}")
    print(f"实际值: {actual}")
    
    # 精度判定
    if actual == ref_result:
        print("结论: 精度保留 (Precision KEPT)")
    elif actual == float(c_init):
        print("结论: 精度完全丢失 (Result == C_init)")
    else:
        # 计算尾数位宽差异
        print(f"结论: 精度部分丢失 (误差: {abs(actual - ref_result)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', type=str, default='einsum', choices=['mm', 'matmul', 'at', 'addmm', 'einsum'])
    parser.add_argument('--c', type=float, default=(1<<24), help='初始值, 如 2^24')
    parser.add_argument('--num', type=int, default=4, help='有效累加个数')
    parser.add_argument('--add', type=float, default=0.5, help='每次累加的增量')
    
    args = parser.parse_args()
    probe_gemm_with_sparse_logic(args.api, args.c, args.num, args.add)