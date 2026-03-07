import sys
import torch
import numpy as np
import argparse

def probe_gemm_with_sparse_logic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 理论参考值
    # ref_result = float(c_init) + (int(num) * float(add_val))
    M, N, K = 256, 256, 256

    
    a_torch = torch.zeros((M, K), dtype=torch.float16, device=device)
    b_torch = torch.zeros((K, N), dtype=torch.float16, device=device)
    # try_torch = torch.zeros((K, N), dtype=torch.float4_e2m1fn_x2, device=device)
    
    # 模仿 CUDA 逻辑：只给前 num 个位置赋值
    # 这样 A[0,:] 和 B[:,0] 的点积就是 num * add_val
    # a_torch[0, :int(num)] = float(add_val)
    # b_torch[:int(num), 0] = 1.0
    a_torch[0, 0] = 32.0
    b_torch[0, 0] = 2048.0

    res = torch.matmul(a_torch, b_torch)
    print(res.dtype)
    res = torch.mm(a_torch,b_torch).to(torch.float32)
    print(res.dtype)

    actual = res[0, 0].item()
    print(f"\n{actual}\n")
    
    # print("-" * 40)
    # print(f"理论值: {ref_result}")
    # print(f"实际值: {actual}")
    
    # # 精度判定
    # if actual == ref_result:
    #     print("结论: 精度保留 (Precision KEPT)")
    # elif actual == float(c_init):
    #     print("结论: 精度完全丢失 (Result == C_init)")
    # else:
    #     # 计算尾数位宽差异
    #     print(f"结论: 精度部分丢失 (误差: {abs(actual - ref_result)})")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--api', type=str, default='einsum', choices=['mm', 'matmul', 'at', 'addmm', 'einsum'])
    # parser.add_argument('--c', type=float, default=(1<<24), help='初始值, 如 2^24')
    # parser.add_argument('--num', type=int, default=4, help='有效累加个数')
    # parser.add_argument('--add', type=float, default=0.5, help='每次累加的增量')
    
    # args = parser.parse_args()
    probe_gemm_with_sparse_logic()