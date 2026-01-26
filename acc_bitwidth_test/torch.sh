# ncu -o gemm_report -f --set full python try_py.py --api mm --c 67108864 --num 256 --add 0.0625

# --page source 表示查看源码/汇编页
# --format text 强制文本输出
# ncu --metrics sm__sass_inst_executed_op_hmma_dot_all.sum \
#     --kernel-name-match REGEX:.*gemm.* \
#     --page source \
#     --format text \
#     python try_py.py --api mm --c 67108864 --num 256 --add 0.0625 | grep -E "HMMA|WGMMA"

#  mkdir -p ./ncu_tmp

# 2. 设置 TMPDIR 环境变量并运行 ncu
# 这样 ncu 会把锁文件创建在 ./ncu_tmp 而不是 /tmp
# TMPDIR=./ncu_tmp ncu -o addmm_report -f --set full python try_py.py --api addmm --c 67108864 --num 256 --add 0.0625
# TMPDIR=./ncu_tmp ncu -o addmm_report -f --set full python try_py_fp16.py --api einsum --c 67108864 --num 256 --add 0.0625

# --nvtx --nvtx-range "MyFP4_MMA_Range"
# TMPDIR=./ncu_tmp ncu --nvtx --nvtx-include "MyFP4_MMA_Range" -o tritonreport -f --set full python fp4_probe_with_triton.py 
TMPDIR=./ncu_tmp ncu -o tritonreport -f --set full python fp4_probe_with_triton.py 