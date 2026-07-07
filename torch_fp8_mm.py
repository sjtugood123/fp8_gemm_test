"""
- For TensorWise scaling, a and b should be float8, scales should be float and singletons.
- For RowWise scaling, a and b should be float8, scales should be float, scale_a should be (1024, 1) and scale_b should be (1, 1024), and both should be contiguous.
- For BlockWise 1x128 scaling, a and b should be float8, scales should be float, scale_a should be (1024, 16) and scale_b should be (16, 1024), and both should be outer-dim-major.
- For BlockWise 128x128 scaling, a and b should be float8, scales should be float, scale_a should be (8, 16) and scale_b should be (16, 8), and both should be near-inner-dim-major (with 16-byte aligned strides).
- For Blockwise 1x32 scaling, a and b should be float8, scales should be float8_e8m0fnu, scale_a should have 65536 elements and scale_b should have 65536 elements, and both should be contiguous.
- For Blockwise 1x16 scaling, a and b should be float4 (packed 2x), scales should be float8_e4m3fn, scale_a should have 262144 elements and scale_b should have 262144 elements, and both should be contiguous.
Got a.dtype()=Float8_e4m3fn, scale_a.dtype()=Half, scale_a.size()=[], scale_a.stride()=[], b.dtype()=Float8_e4m3fn, scale_b.dtype()=Half, scale_b.size()=[] and scale_b.stride()=[]
"""




import torch

# --- 准备工作 ----------------------------------------------------
# 确认GPU是否被PyTorch识别，并设置计算精度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compute_dtype = torch.float16  # 可改为 torch.float32 测试

print(f"PyTorch版本: {torch.__version__}")
print(f"使用的设备: {device}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    # sm120a 算力为 9.0，建议升级到 CUDA 12.8 或更高版本以获得最佳支持 [citation:1]
    print(f"GPU 算力: SM {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")

# --- 第一步：生成 FP16 数据 ---------------------------------------
print("\n=== 第一步：生成原始 FP16 数据 ===")
M, K, N = 1024, 2048, 1024
# 生成随机 FP16 矩阵，数据类型为 torch.float16
A_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
B_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)

# --- 第二步：量化到 FP8 (e4m3fn) ---------------------------------
print("\n=== 第二步：量化到 FP8 (e4m3fn) ===")
# 1. 手工计算一个简单的缩放因子 (scale) 
#    这步很关键：为了保证精度，需要根据数据最大值来选scale，避免溢出。
#    下面是一种简化的 scale 计算方式：
#    torch.finfo(torch.float8_e4m3fn).max 能得出 FP8 可表示的最大值
fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 约为 448.0
scale_a = (A_fp16.abs().max() / fp8_max).clamp(min=1e-6)  # 保护除法
scale_b = (B_fp16.abs().max() / fp8_max).clamp(min=1e-6)

# 2. 执行量化：FP16 -> FP8 (使用缩放因子)
#    (A_fp16 / scale_a) 将数值缩放到 FP8 能表示的范围内，再 .to(dtype) 完成类型转换
A_fp8 = (A_fp16 / scale_a).to(torch.float8_e4m3fn)
B_fp8 = (B_fp16 / scale_b).to(torch.float8_e4m3fn)

print(f"原始数据类型: FP16, 缩放后 FP8 数据: shape={A_fp8.shape}, dtype={A_fp8.dtype}")

# --- 第三步：使用 _scaled_mm 进行 FP8 矩阵乘 ------------------------
print("\n=== 第三步：FP8 矩阵乘法 (CUDA Core 或 Tensor Core) ===")
# torch._scaled_mm 是底层接口，需要传入 scale_a 和 scale_b。
# 输出结果的公式为： Out_fp32 = (A_fp8.to(torch.float32) @ B_fp8.to(torch.float32)) * (scale_a * scale_b)
# 因此，尽管计算在 FP8 下进行，结果会通过scale因子还原回更高的精度。
try:
    # 注意：PyTorch 的 _scaled_mm 要求输入必须是 FP8 类型 (e4m3fn 或 e5m2)
    # 并且必须是 2D 矩阵（不能是向量）。
    # out_dtype 参数控制输出结果的类型，可设置为 torch.float16 或 torch.float32
    result_fp16, scale_result = torch._scaled_mm(
        A_fp8, 
        B_fp8, 
        scale_a=scale_a, 
        scale_b=scale_b, 
        out_dtype=compute_dtype
    )
    print(f"矩阵乘法成功！输出 Tensor: shape={result_fp16.shape}, dtype={result_fp16.dtype}")
    
    # --- (可选) 验证结果的正确性 ------------------------------------
    # 使用 FP16 进行一次标准矩阵乘法作为参考
    result_fp16_ref = torch.mm(A_fp16, B_fp16)
    
    # 计算最大误差
    max_diff = (result_fp16 - result_fp16_ref).abs().max().item()
    # 可以容忍的相对误差范围
    print(f"与 FP16 精确乘法比较的最大误差: {max_diff:.6f}")

except RuntimeError as e:
    print(f"FP8 矩阵乘法失败：{e}")
    print("可能原因：")
    print("1. 当前 PyTorch/CUDA 版本对 sm120a 的 FP8 Tensor Core 支持不完整。")
    print("2. GPU 算力低于 8.9 （需要 H100 或 Blackwell 架构）。")