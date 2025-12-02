nvcc -diag-suppress 550 -O3 -std=c++17 -arch=sm_120a \
-I/home/xtzhao/cutlass/include \
-I/home/xtzhao/cutlass/tools/util/include/ \
-I/home/xtzhao/cutlass/examples/common/ \
-I/home/xtzhao/cutlass/include/cute \
-o bin/fp8_acc_bitwidth fp8_acc_bitwidth.cu -lcudart

nvcc -diag-suppress 550 -O3 -std=c++17 -arch=sm_120a \
-I/home/xtzhao/cutlass/include \
-I/home/xtzhao/cutlass/tools/util/include/ \
-I/home/xtzhao/cutlass/examples/common/ \
-I/home/xtzhao/cutlass/include/cute \
-o bin/fp4_acc_bitwidth fp4_acc_bitwidth.cu -lcudart

# nvcc -O3 -std=c++17 -arch=sm_120a \
# -I/home/xtzhao/cutlass/include \
# -I/home/xtzhao/cutlass/tools/util/include/ \
# -I/home/xtzhao/cutlass/examples/common/ \
# -I/home/xtzhao/cutlass/include/cute \
# -o bin/fp4_block_scaled_mma fp4_block_scaled_mma.cu -lcudart

./bin/fp8_acc_bitwidth
./bin/fp4_acc_bitwidth
# ./bin/fp4_block_scaled_mma

cuobjdump -sass ./bin/fp8_acc_bitwidth > fp8.sass
cuobjdump -sass ./bin/fp4_acc_bitwidth > fp4.sass
# cuobjdump -sass ./bin/fp4_block_scaled_mma > fp4_bs_mma.sass